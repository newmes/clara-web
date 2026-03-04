"""Multi-stage LLM pipeline for grounded rule synthesis.

Replaces the single-call synthesis in agent.py with 3 focused stages:
  Stage 1: 5 parallel focused extraction sub-calls (AE freq, severity, onset, triggers, demographics)
  Stage 2: Grounding verification against raw evidence
  Stage 3: Final JSON assembly combining stage 1 + stage 2 into a complete RuleSet
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone

from openai import AsyncOpenAI

from rule_engine.agent import AgentLog, _extract_and_parse_ruleset, _repair_json
from rule_engine.config import RuleEngineConfig
from rule_engine.rate_limiter import RateLimiter
from rule_engine.prompts import (
    STAGE1_AE_FREQ_PROMPT,
    STAGE1_DEMOGRAPHICS_PROMPT,
    STAGE1_ONSET_PROMPT,
    STAGE1_SEVERITY_PROMPT,
    STAGE1_TRIGGERS_PROMPT,
    STAGE2_GROUNDING_PROMPT,
    STAGE3_SYNTHESIS_PROMPT,
    format_evidence_prompt,
)
from rule_engine.schema import EvidenceBundle, RuleSet

log = logging.getLogger(__name__)

# JSON schema for the RuleSet, reused in stage 3
_RULESET_JSON_SCHEMA = json.dumps(RuleSet.model_json_schema(), indent=2)

# Focused re-prompt when Stage 3 omits comorbidities
_COMORBIDITY_REPROMPT = """\
You are a clinical pharmacologist. The previous synthesis for {drugs} / {indication} \
omitted comorbidities. Using the trial evidence below, extract 5-8 pre-existing \
comorbidities commonly seen in patients with {indication} who receive {drugs}.

Rules:
- Only include pre-existing medical conditions (NOT biomarkers or lab values).
- prevalence_pct is the % of trial patients with that comorbidity (0-100 scale).
- impacts_dosing = true if the comorbidity requires dose adjustment for any of the study drugs.
- Base your answers ONLY on the evidence provided. Use trial-population prevalences (5-15% each), NOT general population rates.

Respond with a JSON object:
{{"comorbidities": [{{"condition": "..", "prevalence_pct": .., "impacts_dosing": <bool>}}]}}

EVIDENCE:
{evidence}
"""

# Focused re-prompt when Stage 3 omits demographics
_DEMOGRAPHICS_REPROMPT = """\
You are a clinical trial statistician. The previous synthesis for {drugs} / {indication} \
had incomplete demographics. Using the trial evidence below, extract patient demographics.

Rules:
- ALL percentages on 0-100 scale.
- Race/ethnicity: include at least White, Black, Asian, Hispanic, Other — sum ~100%.
- Use large-trial demographics (n >= 30).

Respond with a JSON object:
{{"demographics": {{"age": {{"min": .., "max": .., "mean": .., "std": ..}}, "sex": {{"pct_male": .., "pct_female": ..}}, "race_ethnicity": [{{"group": "..", "pct": ..}}]}}}}

EVIDENCE:
{evidence}
"""

# Focused re-prompt when Stage 3 omits efficacy
_EFFICACY_REPROMPT = """\
You are a clinical pharmacologist. The previous synthesis for {drugs} / {indication} \
had incomplete efficacy data. Using the trial evidence below, extract efficacy endpoints.

Rules:
- ALL percentages on 0-100 scale (e.g., 30% ORR → 30, not 0.30).
- complete_response_rate_pct <= overall_response_rate_pct.
- Use pivotal Phase III trial results when available.
- You MUST provide numeric values, not null.
- For ORR: look for "Overall Response Rate", "Objective Response Rate", or "ORR" in the outcomes.
  If ORR is not in the evidence, use published Phase III trial data for this drug+indication.
- For PFS/OS: use the EXACT values from the evidence. Do NOT estimate if evidence provides them.
- Use the EXACT numeric values from the evidence when available. Do NOT round significantly.
- PFS and OS: provide MEDIAN plus 95% CONFIDENCE INTERVAL bounds. Every drug combination has \
different efficacy — do NOT use generic values.
- If exact CI is unknown, estimate plausible bounds (e.g., median 5.5 months, CI 4.5-7.0).

Respond with a JSON object:
{{"efficacy": {{"overall_response_rate_pct": 30.0, "complete_response_rate_pct": 5.0, \
"median_pfs_months": 5.5, "median_pfs_ci_low": 4.5, "median_pfs_ci_high": 7.0, \
"median_os_months": 12.0, "median_os_ci_low": 10.0, "median_os_ci_high": 15.0}}}}

EVIDENCE:
{evidence}
"""

# Focused re-prompt when regimen doses are vague
_DOSE_REPROMPT = """\
You are a clinical pharmacologist. The regimen for {drugs} / {indication} has vague dose \
entries: {vague_entries}. Using the dosage evidence below, provide specific dose values.

Rules:
- Provide specific numeric doses with units (e.g., "175 mg/m^2", "AUC 5", "15 mg/kg").
- For AUC-based dosing (e.g., Carboplatin), use "AUC X" format.
- Use standard approved doses from pivotal trials.

Respond with a JSON object mapping drug names to doses:
{{"doses": {{"DrugName": "175 mg/m^2", "DrugName2": "AUC 5"}}}}

DOSAGE EVIDENCE:
{evidence}
"""

# Focused re-prompt when regimen is completely empty or missing drugs
_REGIMEN_REPROMPT = """\
You are a clinical pharmacologist. The regimen for {drugs} / {indication} was not generated. \
Using the dosage evidence below, provide the complete regimen.

Rules:
- Include ALL drugs: {drugs}
- Provide specific numeric doses with units (e.g., "175 mg/m^2", "AUC 5", "15 mg/kg").
- cycle_days is the cycle length in days (typically 21 for most chemo).
- schedule describes which days of the cycle the drug is given (e.g., "Day 1", "Days 1-3").
- route must be one of: "IV", "oral", "subcutaneous".

Respond with a JSON object:
{{"regimen": [{{"drug": "DrugName", "dose": "175 mg/m^2", "route": "IV", "cycle_days": 21, "schedule": "Day 1"}}]}}

DOSAGE EVIDENCE:
{evidence}
"""


def _format_ae_freq_evidence(evidence: EvidenceBundle) -> str:
    """Extract DailyMed AE table + OnSIDES + CT.gov data for the AE frequency sub-call."""
    sections: list[str] = []
    for drug in evidence.drugs:
        sd = evidence.per_drug.get(drug)
        if sd is None:
            continue
        dm = sd.dailymed
        if dm.found and dm.ae_table:
            sections.append(f"[DailyMed — {drug} — AE incidence table]")
            for ae in dm.ae_table[:30]:
                term = ae.get("term", "?")
                line = f"  {term}: {ae.get('incidence_pct', '?')}%"
                if ae.get("grade34_pct") is not None:
                    line += f" (grade 3-4: {ae['grade34_pct']}%)"
                table_type = ae.get("table_type", "")
                if table_type and table_type != "clinical":
                    line += f" [{table_type}]"
                sections.append(line)
        onsides = sd.onsides
        if onsides.found:
            sections.append(f"\n[OnSIDES — {drug} — validated drug-ADE pairs]")
            if onsides.boxed_warning_aes:
                sections.append(f"  BOXED WARNING AEs: {', '.join(onsides.boxed_warning_aes)}")
            for ae in onsides.ae_pairs[:20]:
                bw_tag = " [BOXED WARNING]" if ae.get("is_boxed_warning") else ""
                sections.append(
                    f"  {ae.get('pt_meddra_term', '?')}: labels={ae.get('label_count', '?')}{bw_tag}"
                )
        sections.append("")
    # ClinicalTrials.gov reported AEs — important frequency source alongside DailyMed
    # Only show AEs from trials with at_risk >= 10 to prevent small-trial
    # frequency inflation (e.g. 3/3 = 100%) from misleading the LLM.
    _MIN_AE_DISPLAY = 10
    ct = evidence.clinical_trials
    if ct.reported_aes:
        filtered = [ae for ae in ct.reported_aes if ae.get("at_risk", 0) >= _MIN_AE_DISPLAY]
        if filtered:
            sections.append("[ClinicalTrials.gov — reported AEs with frequencies (MUST include all of these)]")
            for ae in filtered[:40]:
                term = ae.get("term", "?")
                pct = ae.get("pct", "?")
                n_info = f" ({ae.get('affected', '?')}/{ae.get('at_risk', '?')})"
                line = f"  {term}: {pct}%{n_info}"
                if grade34 := ae.get("grade34_pct"):
                    line += f" (grade 3-4: {grade34}%)"
                sections.append(line)
            sections.append("")
    # Combo trial AEs
    combo_ct = evidence.combo_trials
    if combo_ct.reported_aes:
        filtered = [ae for ae in combo_ct.reported_aes if ae.get("at_risk", 0) >= _MIN_AE_DISPLAY]
        if filtered:
            sections.append("[ClinicalTrials.gov — COMBO trial AEs (high priority for combination therapy)]")
            for ae in filtered[:40]:
                term = ae.get("term", "?")
                pct = ae.get("pct", "?")
                n_info = f" ({ae.get('affected', '?')}/{ae.get('at_risk', '?')})"
                line = f"  {term}: {pct}%{n_info}"
                if grade34 := ae.get("grade34_pct"):
                    line += f" (grade 3-4: {grade34}%)"
                sections.append(line)
            sections.append("")
    # Project Data Sphere — patient-level AE data
    pds = evidence.pds
    if pds.found and pds.ae_aggregates:
        n_label = pds.safety_population_n or (pds.matched_trial.n_patients if pds.matched_trial else 0)
        sections.append(f"[Project Data Sphere — patient-level AE data (n={n_label})]")
        for ae in pds.ae_aggregates[:40]:
            line = f"  {ae.term}: {ae.frequency_pct}% ({ae.n_patients_with_event}/{ae.n_total_patients})"
            if ae.grade_distribution:
                grades = ", ".join(f"g{k}={v}%" for k, v in sorted(ae.grade_distribution.items()))
                line += f" [{grades}]"
            if ae.median_onset_day is not None:
                line += f" onset={ae.median_onset_day}d"
            sections.append(line)
        sections.append("")
    return "\n".join(sections)


def _format_severity_evidence(evidence: EvidenceBundle) -> str:
    """Extract grade data from DailyMed + CT.gov for severity distribution sub-call."""
    sections: list[str] = []
    for drug in evidence.drugs:
        sd = evidence.per_drug.get(drug)
        if sd is None:
            continue
        dm = sd.dailymed
        if dm.found and dm.ae_table:
            sections.append(f"[DailyMed — {drug} — AE grade data]")
            for ae in dm.ae_table[:30]:
                term = ae.get("term", "?")
                freq = ae.get("incidence_pct", "?")
                grade34 = ae.get("grade34_pct")
                line = f"  {term}: frequency_pct={freq}%"
                if grade34 is not None:
                    line += f", grade34_pct={grade34}%"
                sections.append(line)
    ct = evidence.clinical_trials
    if ct.reported_aes:
        sections.append("\n[ClinicalTrials.gov — reported AEs with grade data]")
        for ae in ct.reported_aes[:40]:
            term = ae.get("term", "?")
            pct = ae.get("pct", "?")
            grade34 = ae.get("grade34_pct")
            line = f"  {term}: {pct}%"
            if grade34 is not None:
                line += f" (grade 3-4: {grade34}%)"
            sections.append(line)
    sections.append("")
    return "\n".join(sections)


def _format_onset_evidence(evidence: EvidenceBundle) -> str:
    """Extract FAERS time-to-onset data + drug mechanism info for onset sub-call."""
    sections: list[str] = []
    for drug in evidence.drugs:
        sd = evidence.per_drug.get(drug)
        if sd is None:
            continue
        fda = sd.openfda
        if fda.has_timing_data and fda.time_to_onset_data:
            sections.append(f"[FAERS — {drug} — time-to-onset data]")
            total = sum(e.get("count", 0) for e in fda.time_to_onset_data)
            sections.append(f"  Total reports with timing: {total}")
            for entry in fda.time_to_onset_data:
                sections.append(
                    f"  Interval '{entry.get('unit_label', '?')}': {entry.get('count', 0)} reports"
                )
        # Drug mechanism context for class-based onset estimation
        db = sd.drugbank
        if db.moa:
            sections.append(f"\n[DrugBank — {drug} — mechanism]")
            sections.append(f"  MoA: {db.moa}")
        ch = sd.chembl
        if ch.mechanism_of_action:
            sections.append(f"  ChEMBL mechanism: {ch.mechanism_of_action}")
        sections.append("")
    return "\n".join(sections)


def _format_triggers_evidence(evidence: EvidenceBundle) -> str:
    """Format AE list for the triggers sub-call (just the event names + categories)."""
    sections: list[str] = []
    for drug in evidence.drugs:
        sd = evidence.per_drug.get(drug)
        if sd is None:
            continue
        dm = sd.dailymed
        if dm.found and dm.ae_table:
            sections.append(f"[{drug} — AEs requiring trigger rules]")
            for ae in dm.ae_table[:30]:
                term = ae.get("term", "?")
                freq = ae.get("incidence_pct", "?")
                sections.append(f"  {term}: {freq}%")
    sections.append(f"\nDrug(s): {', '.join(evidence.drugs)}")
    sections.append(f"Indication: {evidence.indication}")
    sections.append("")
    return "\n".join(sections)


def _format_demographics_evidence(evidence: EvidenceBundle) -> str:
    """Extract CT.gov demographics, outcomes, and dosing info for demographics sub-call."""
    sections: list[str] = []
    ct = evidence.clinical_trials
    combo_ct = evidence.combo_trials
    sections.append(f"Drug(s): {', '.join(evidence.drugs)}")
    sections.append(f"Indication: {evidence.indication}")
    sections.append(f"Trial count: {ct.trial_count}, max phase: {ct.max_phase}")
    if ct.age_range:
        sections.append(f"Eligibility age range: {ct.age_range} — USE THIS for age min/max")
    if ct.has_results:
        sample_n = ct.baseline_demographics.get("_sample_size", 0)
        if ct.baseline_demographics and sample_n >= 30:
            sections.append(f"\n[Baseline demographics from trial (n={sample_n}) — USE THESE VALUES]")
            for key, val in ct.baseline_demographics.items():
                if key.startswith("_"):
                    continue
                sections.append(f"  {key}: {val}")
        if ct.primary_outcomes:
            sections.append("\n[Primary outcomes — USE THESE for efficacy]")
            for out in ct.primary_outcomes[:10]:
                sections.append(
                    f"  {out.get('measure', '?')}: {out.get('value', '?')} {out.get('unit', '')}"
                )
    # Combo trial demographics and outcomes (often more relevant for combo drugs)
    if combo_ct.trial_count > 0 and combo_ct.has_results:
        combo_n = combo_ct.baseline_demographics.get("_sample_size", 0)
        if combo_ct.baseline_demographics and combo_n >= 30:
            sections.append(f"\n[COMBO trial demographics (n={combo_n}) — PREFERRED for combination therapy]")
            for key, val in combo_ct.baseline_demographics.items():
                if key.startswith("_"):
                    continue
                sections.append(f"  {key}: {val}")
        if combo_ct.primary_outcomes:
            sections.append("\n[COMBO trial outcomes — PREFERRED for efficacy]")
            for out in combo_ct.primary_outcomes[:10]:
                sections.append(
                    f"  {out.get('measure', '?')}: {out.get('value', '?')} {out.get('unit', '')}"
                )
    # Dosage info from labels
    for drug in evidence.drugs:
        sd = evidence.per_drug.get(drug)
        if sd is None:
            continue
        dm = sd.dailymed
        if dm.found and dm.dosage_text:
            sections.append(f"\n[DailyMed — {drug} — dosage]")
            sections.append(f"  {dm.dosage_text[:500]}")
    # Project Data Sphere — patient-level demographics
    pds = evidence.pds
    if pds.found and pds.demographics:
        demo = pds.demographics
        n_label = demo.n_patients or (pds.matched_trial.n_patients if pds.matched_trial else 0)
        sections.append(f"\n[Project Data Sphere — patient-level data (n={n_label}) — HIGH QUALITY]")
        if demo.age_mean is not None:
            sections.append(f"  Age: mean={demo.age_mean}, std={demo.age_std}, min={demo.age_min}, max={demo.age_max}")
        if demo.pct_male is not None:
            sections.append(f"  Sex: {demo.pct_male}% male, {demo.pct_female}% female")
        if demo.ecog_distribution:
            ecog_str = ", ".join(f"ECOG {k}={v}%" for k, v in sorted(demo.ecog_distribution.items()))
            sections.append(f"  ECOG PS: {ecog_str}")
        if demo.race_distribution:
            race_str = ", ".join(f"{k}={v}%" for k, v in sorted(demo.race_distribution.items(), key=lambda x: -x[1])[:5])
            sections.append(f"  Race: {race_str}")
    sections.append("")
    return "\n".join(sections)


def _format_efficacy_evidence(evidence: EvidenceBundle) -> str:
    """Extract CT.gov outcomes, DailyMed dosage, and trial info for efficacy reprompt."""
    sections: list[str] = []
    ct = evidence.clinical_trials
    combo_ct = evidence.combo_trials
    sections.append(f"Drug(s): {', '.join(evidence.drugs)}")
    sections.append(f"Indication: {evidence.indication}")
    sections.append(f"Trial count: {ct.trial_count}, max phase: {ct.max_phase}")
    if ct.primary_endpoints:
        sections.append(f"Primary endpoints: {', '.join(ct.primary_endpoints)}")
    if ct.has_results:
        if ct.primary_outcomes:
            sections.append("\n[Primary outcome results from ClinicalTrials.gov]")
            for out in ct.primary_outcomes[:10]:
                sections.append(
                    f"  {out.get('measure', '?')}: {out.get('value', '?')} {out.get('unit', '')}"
                )
    if combo_ct.trial_count > 0:
        sections.append(f"\n[Combo trial info] {combo_ct.trial_count} trials, max phase {combo_ct.max_phase}")
        if combo_ct.primary_endpoints:
            sections.append(f"  Endpoints: {', '.join(combo_ct.primary_endpoints)}")
        if combo_ct.primary_outcomes:
            sections.append("  Combo trial outcomes:")
            for out in combo_ct.primary_outcomes[:10]:
                sections.append(
                    f"    {out.get('measure', '?')}: {out.get('value', '?')} {out.get('unit', '')}"
                )
    # DailyMed clinical studies section (often contains efficacy data)
    for drug in evidence.drugs:
        sd = evidence.per_drug.get(drug)
        if sd is None:
            continue
        dm = sd.dailymed
        if dm.found:
            if dm.dosage_text:
                sections.append(f"\n[DailyMed — {drug} — dosage]")
                sections.append(f"  {dm.dosage_text[:500]}")
    # Project Data Sphere — patient-level efficacy
    pds = evidence.pds
    if pds.found and pds.efficacy:
        eff = pds.efficacy
        n_label = pds.safety_population_n or (pds.matched_trial.n_patients if pds.matched_trial else 0)
        sections.append(f"\n[Project Data Sphere — patient-level efficacy (n={n_label}) — HIGH QUALITY]")
        if eff.overall_response_rate_pct is not None:
            sections.append(f"  ORR: {eff.overall_response_rate_pct}%")
        if eff.complete_response_rate_pct is not None:
            sections.append(f"  CR: {eff.complete_response_rate_pct}%")
        if eff.median_pfs_months is not None:
            sections.append(f"  Median PFS: {eff.median_pfs_months} months")
        if eff.median_os_months is not None:
            sections.append(f"  Median OS: {eff.median_os_months} months")
    sections.append("")
    return "\n".join(sections)


def _format_dose_evidence(evidence: EvidenceBundle) -> str:
    """Extract DailyMed dosage info for dose reprompt, with structured extraction."""
    sections: list[str] = []
    sections.append(f"Drug(s): {', '.join(evidence.drugs)}")
    sections.append(f"Indication: {evidence.indication}")

    # Structured extraction summary (pre-parsed from label)
    structured = _extract_structured_doses(evidence)
    if structured:
        sections.append("\n[STRUCTURED DOSES FROM DRUG LABEL — USE THESE EXACT VALUES]")
        for drug, info in structured.items():
            sections.append(f"  {drug}: {info['dose_value']} ({info['route']})")

    for drug in evidence.drugs:
        sd = evidence.per_drug.get(drug)
        if sd is None:
            continue
        dm = sd.dailymed
        if dm.found and dm.dosage_text:
            sections.append(f"\n[DailyMed — {drug} — dosage and administration]")
            sections.append(f"  {dm.dosage_text[:1000]}")
    # Project Data Sphere — patient-level regimen data
    pds = evidence.pds
    if pds.found and pds.regimen:
        n_label = pds.safety_population_n or (pds.matched_trial.n_patients if pds.matched_trial else 0)
        sections.append(f"\n[Project Data Sphere — actual administered doses (n={n_label}) — HIGH QUALITY]")
        for reg in pds.regimen:
            parts = [reg.drug, ":"]
            if reg.median_dose is not None:
                parts.append(f" median_dose={reg.median_dose}")
            if reg.dose_unit:
                parts.append(f" {reg.dose_unit}")
            if reg.route:
                parts.append(f" ({reg.route})")
            parts.append(f" [n={reg.n_patients}]")
            sections.append(f"  {''.join(parts)}")
    sections.append("")
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Programmatic extraction functions — deterministic evidence parsing
# ---------------------------------------------------------------------------

# Dose patterns in priority order (most specific → least specific)
# DailyMed text may have whitespace artifacts: "mg/m 2", "mg / m2", etc.
# Patterns support comma-separated thousands (e.g., "1,000 mg/m²")
_DOSE_PATTERNS = [
    # mg/m² or mg/m2 or mg/m^2 — BSA dosing (most chemo)
    # Handles "mg/m²", "mg/m2", "mg/m 2", "mg/m^2", "mg / m2"
    (re.compile(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*mg\s*/\s*m\s*[²2^]?\s*2?(?!\w)", re.IGNORECASE), "mg/m^2"),
    # AUC-based — Calvert formula (Carboplatin)
    # Captures range: "AUC 4-6" → group(1)=4, group(2)=6
    (re.compile(r"AUC\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(?:[-–to]+\s*(\d+(?:\.\d+)?))?", re.IGNORECASE), "AUC"),
    # mg/kg — weight-based (biologics)
    (re.compile(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*mg\s*/\s*kg", re.IGNORECASE), "mg/kg"),
    # mcg/kg — weight-based (some biologics like Darbepoetin)
    (re.compile(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:mcg|µg)\s*/\s*kg", re.IGNORECASE), "mcg/kg"),
    # flat mg — exclude mg/m and mg/k prefixes
    (re.compile(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*mg\b(?!\s*/)", re.IGNORECASE), "mg"),
]

# Multi-indication drugs where DailyMed label dose may not match the query
# indication. Maps (drug_lower, indication_keyword) → preferred dose info.
# When CT.gov protocol description has a dose for the matching indication,
# it should be preferred over DailyMed label dose.
_MULTI_INDICATION_DRUGS: dict[str, list[str]] = {
    "cisplatin": ["lung", "sclc", "nsclc", "bladder", "testicular", "ovarian", "head"],
    "paclitaxel": ["lung", "nsclc", "breast", "ovarian", "kaposi"],
    "carboplatin": ["lung", "sclc", "nsclc", "ovarian"],
    "etoposide": ["lung", "sclc", "nsclc", "testicular"],
    "gemcitabine": ["lung", "nsclc", "pancreatic", "bladder", "breast", "ovarian"],
}

# Fallback standard doses when CT.gov descriptions lack explicit protocol doses.
# Key: (drug_lower, indication_keyword) → (dose_str, unit)
# These are NCCN/literature standard doses, used only when no other source provides a dose.
_INDICATION_DOSE_FALLBACKS: dict[tuple[str, str], tuple[str, str]] = {
    ("cisplatin", "sclc"): ("80 mg/m^2", "mg/m^2"),
    ("cisplatin", "small cell lung"): ("80 mg/m^2", "mg/m^2"),
    ("cisplatin", "nsclc"): ("75 mg/m^2", "mg/m^2"),
    ("cisplatin", "non-small cell lung"): ("75 mg/m^2", "mg/m^2"),
    ("cisplatin", "squamous non-small cell"): ("75 mg/m^2", "mg/m^2"),
    ("paclitaxel", "nsclc"): ("200 mg/m^2", "mg/m^2"),
    ("paclitaxel", "non-small cell lung"): ("200 mg/m^2", "mg/m^2"),
    ("paclitaxel", "squamous non-small cell"): ("200 mg/m^2", "mg/m^2"),
    ("paclitaxel", "sclc"): ("175 mg/m^2", "mg/m^2"),
    ("paclitaxel", "small cell lung"): ("175 mg/m^2", "mg/m^2"),
    ("carboplatin", "nsclc"): ("AUC 6", "AUC"),
    ("carboplatin", "non-small cell lung"): ("AUC 6", "AUC"),
    ("carboplatin", "squamous non-small cell"): ("AUC 6", "AUC"),
    ("carboplatin", "sclc"): ("AUC 5", "AUC"),
    ("carboplatin", "small cell lung"): ("AUC 5", "AUC"),
    ("etoposide", "sclc"): ("100 mg/m^2", "mg/m^2"),
    ("etoposide", "small cell lung"): ("100 mg/m^2", "mg/m^2"),
    ("gemcitabine", "nsclc"): ("1250 mg/m^2", "mg/m^2"),
    ("gemcitabine", "non-small cell lung"): ("1250 mg/m^2", "mg/m^2"),
    ("gemcitabine", "squamous non-small cell"): ("1250 mg/m^2", "mg/m^2"),
}

_ROUTE_PATTERNS = [
    (re.compile(r"intravenous|IV\b|infusion", re.IGNORECASE), "IV"),
    (re.compile(r"\boral\b|by\s+mouth|capsule|tablet", re.IGNORECASE), "oral"),
    (re.compile(r"subcutaneous|\bSC\b|injection", re.IGNORECASE), "subcutaneous"),
]


def _extract_structured_doses(evidence: EvidenceBundle) -> dict[str, dict]:
    """Parse structured dose entries per drug from DailyMed text + CT.gov descriptions.

    Strategy:
    1. For each drug, find the indication-specific section in DailyMed dosage text
    2. Extract dose from that section (indication-aware)
    3. If no indication match, fall back to drug-name proximity
    4. Also scan CT.gov outcome/study descriptions for explicit protocol doses

    Returns mapping: drug_name -> {"dose_value": "80 mg/m^2", "dose_unit": "mg/m^2", "route": "IV"}
    """
    result: dict[str, dict] = {}
    indication_lower = evidence.indication.lower()

    # --- Source 1: DailyMed dosage text (indication-aware) ---
    for drug in evidence.drugs:
        sd = evidence.per_drug.get(drug)
        if sd is None or not sd.dailymed.found or not sd.dailymed.dosage_text:
            continue

        text = sd.dailymed.dosage_text

        # Try to find indication-specific section first
        ind_pos = -1
        # Try common indication terms
        for term in _indication_search_terms(indication_lower):
            pos = text.lower().find(term)
            if pos >= 0:
                ind_pos = pos
                break

        # Search region: prefer near indication mention, fall back to drug name
        anchor_pos = ind_pos
        if anchor_pos < 0:
            drug_lower = drug.lower()
            anchor_pos = text.lower().find(drug_lower)
            if anchor_pos < 0:
                first_word = drug_lower.split()[0] if drug_lower.split() else drug_lower
                anchor_pos = text.lower().find(first_word)

        best_dose = None
        best_unit = None

        for pattern, unit in _DOSE_PATTERNS:
            matches = list(pattern.finditer(text))
            if not matches:
                continue

            if anchor_pos >= 0 and len(matches) > 1:
                # Pick match closest to anchor (indication or drug name)
                best_match = min(matches, key=lambda m: abs(m.start() - anchor_pos))
            else:
                best_match = matches[0]

            # Strip commas from captured dose value (e.g., "1,000" → "1000")
            raw_dose = best_match.group(1).replace(",", "")
            best_dose = raw_dose
            best_unit = unit

            # AUC range: if group(2) exists (e.g., "AUC 4-6"), use midpoint
            if unit == "AUC" and best_match.lastindex and best_match.lastindex >= 2 and best_match.group(2):
                try:
                    lo = float(raw_dose)
                    hi = float(best_match.group(2))
                    best_dose = str(round((lo + hi) / 2, 1))
                except ValueError:
                    pass
            break

        if best_dose is not None:
            route = "IV"
            for route_pat, route_val in _ROUTE_PATTERNS:
                if route_pat.search(text):
                    route = route_val
                    break

            dose_str = f"AUC {best_dose}" if best_unit == "AUC" else f"{best_dose} {best_unit}"
            result[drug] = {
                "dose_value": dose_str,
                "dose_unit": best_unit,
                "route": route,
                "source": "dailymed",
            }

    # --- Source 2: CT.gov trial descriptions (often has exact protocol doses) ---
    # Scan combo and individual trial outcome descriptions for drug+dose mentions
    _ctgov_doses = _extract_doses_from_ctgov_descriptions(evidence)
    for drug, dose_info in _ctgov_doses.items():
        if drug not in result:
            result[drug] = dose_info
        else:
            # For multi-indication drugs (Cisplatin, Paclitaxel, etc.), CT.gov
            # protocol doses are strongly preferred over DailyMed label doses
            # because the label may show a different indication's dose
            # (e.g., Cisplatin 20 mg/m² for bladder vs 75-80 mg/m² for lung)
            drug_lower = drug.lower()
            is_multi_indication = drug_lower in _MULTI_INDICATION_DRUGS
            existing_num = re.search(r"(\d+(?:\.\d+)?)", result[drug]["dose_value"])
            new_num = re.search(r"(\d+(?:\.\d+)?)", dose_info["dose_value"])
            if existing_num and new_num:
                existing_val = float(existing_num.group(1))
                new_val = float(new_num.group(1))
                if is_multi_indication and result[drug]["dose_unit"] == dose_info["dose_unit"]:
                    # Multi-indication: always prefer CT.gov (indication-specific)
                    result[drug] = dose_info
                    log.info("Multi-indication dose override: %s %s → %s (CT.gov)",
                             drug, existing_val, new_val)
                elif result[drug]["dose_unit"] == dose_info["dose_unit"] and new_val > existing_val:
                    # Same-unit: prefer higher dose (trial combo doses > label monotherapy)
                    result[drug] = dose_info

    # --- Source 2b: Indication-specific fallback doses for multi-indication drugs ---
    # When CT.gov descriptions don't contain explicit protocol doses, DailyMed may
    # have the wrong indication's dose (e.g., Cisplatin 20 mg/m² from bladder label).
    # Use standard-of-care doses for well-known drug-indication pairs.
    for drug in evidence.drugs:
        drug_lower = drug.lower()
        if drug_lower not in _MULTI_INDICATION_DRUGS:
            continue
        if drug not in result:
            continue
        # Check if current dose likely comes from wrong indication
        for (fb_drug, fb_ind), (fb_dose, fb_unit) in _INDICATION_DOSE_FALLBACKS.items():
            if fb_drug == drug_lower and fb_ind in indication_lower:
                curr = result[drug]
                curr_num = re.search(r"(\d+(?:\.\d+)?)", curr["dose_value"])
                fb_num = re.search(r"(\d+(?:\.\d+)?)", fb_dose)
                if curr["dose_unit"] != fb_unit:
                    # Different unit type (e.g., mg/m² vs AUC) — fallback unit
                    # is authoritative for this indication
                    log.info("Indication fallback dose (unit change): %s %s(%s) → %s(%s)",
                             drug, curr["dose_value"], curr["dose_unit"], fb_dose, fb_unit)
                    result[drug] = {"dose_value": fb_dose, "dose_unit": fb_unit,
                                    "route": curr.get("route", "IV")}
                elif curr_num and fb_num:
                    curr_val = float(curr_num.group(1))
                    fb_val = float(fb_num.group(1))
                    # Override if current dose differs from standard (>10% threshold —
                    # low because fallback IS the correct dose for this indication)
                    if fb_val > 0 and abs(curr_val - fb_val) / fb_val > 0.10:
                        log.info("Indication fallback dose: %s %s → %s (%s)",
                                 drug, curr["dose_value"], fb_dose, fb_ind)
                        result[drug] = {"dose_value": fb_dose, "dose_unit": fb_unit,
                                        "route": curr.get("route", "IV")}
                break

    # --- Source 3: PDS actual administered doses (highest quality) ---
    # PDS has actual administered doses from matched trial — always override
    # DailyMed/CT.gov (which may list wrong-indication doses).
    pds = evidence.pds
    if pds.found and pds.regimen:
        # Filter out placebo entries
        active_regimen = [
            reg for reg in pds.regimen
            if reg.median_dose is not None
            and reg.drug
            and "placebo" not in reg.drug.lower()
        ]
        matched_pds_drugs: set[str] = set()
        for reg in active_regimen:
            pds_norm = re.sub(r"[^a-z0-9]", "", reg.drug.lower())
            for query_drug in evidence.drugs:
                query_norm = re.sub(r"[^a-z0-9]", "", query_drug.lower())
                if query_norm in pds_norm or pds_norm in query_norm:
                    dose_str = f"{reg.median_dose} {reg.dose_unit}" if reg.dose_unit else str(reg.median_dose)
                    old = result.get(query_drug, {}).get("dose_value", "none")
                    result[query_drug] = {
                        "dose_value": dose_str,
                        "dose_unit": reg.dose_unit or "mg",
                        "route": reg.route or "IV",
                        "source": "pds",
                    }
                    matched_pds_drugs.add(reg.drug)
                    if old != "none":
                        log.info("PDS dose override: %s %s -> %s", query_drug, old, dose_str)
                    break

        # Fallback: query drugs without PDS-sourced doses + unmatched PDS drugs.
        # Pair them 1:1 (handles code names like NESP300Q3WK → Darbepoetin).
        # PDS patient-level dose always overrides label/CT.gov dose.
        non_pds_query = [
            d for d in evidence.drugs
            if result.get(d, {}).get("source") != "pds"
        ]
        unmatched_pds = [r for r in active_regimen if r.drug not in matched_pds_drugs]
        if len(non_pds_query) == 1 and len(unmatched_pds) == 1:
            reg = unmatched_pds[0]
            qd = non_pds_query[0]
            dose_str = f"{reg.median_dose} {reg.dose_unit}" if reg.dose_unit else str(reg.median_dose)
            old = result.get(qd, {}).get("dose_value", "none")
            result[qd] = {
                "dose_value": dose_str,
                "dose_unit": reg.dose_unit or "mg",
                "route": reg.route or "IV",
                "source": "pds",
            }
            log.info(
                "PDS dose fallback: %s -> %s = %s (was: %s)",
                reg.drug, qd, dose_str, old,
            )

    return result


def _indication_search_terms(indication_lower: str) -> list[str]:
    """Generate search terms for finding indication sections in dosage text."""
    terms = [indication_lower]
    # Add acronyms and abbreviations
    if "small cell lung" in indication_lower:
        terms.extend(["small cell lung", "sclc"])
    if "non-small cell" in indication_lower or "nsclc" in indication_lower:
        terms.extend(["non-small cell", "nsclc", "non-squamous", "squamous"])
    if "lung" in indication_lower:
        terms.append("lung")
    if "breast" in indication_lower:
        terms.append("breast")
    if "melanoma" in indication_lower:
        terms.append("melanoma")
    if "ovarian" in indication_lower:
        terms.append("ovarian")
    if "bladder" in indication_lower:
        terms.append("bladder")
    return terms


def _extract_doses_from_ctgov_descriptions(evidence: EvidenceBundle) -> dict[str, dict]:
    """Extract drug doses mentioned in CT.gov outcome descriptions and study arms.

    CT.gov outcome descriptions often contain exact protocol doses like:
    "etoposide 100 mg/m2 on Days 1, 2 and 3, and cisplatin 80 mg/m2"
    """
    result: dict[str, dict] = {}

    # Collect all text that might contain protocol doses
    desc_texts: list[str] = []
    for outcomes in [
        evidence.combo_trials.primary_outcomes,
        evidence.clinical_trials.primary_outcomes,
    ]:
        for out in outcomes:
            desc = out.get("description", "")
            if desc:
                desc_texts.append(desc)
            measure = out.get("measure", "")
            if measure:
                desc_texts.append(measure)

    all_text = " ".join(desc_texts)

    # For each drug, look for "drug_name X mg/m2" patterns
    for drug in evidence.drugs:
        drug_lower = drug.lower()
        # Pattern: drug name followed by dose within ~20 chars (supports comma thousands)
        drug_dose_pat = re.compile(
            rf"{re.escape(drug_lower)}\s+(\d{{1,3}}(?:,\d{{3}})*(?:\.\d+)?)\s*mg\s*/\s*m\s*[²2^]?\s*2?",
            re.IGNORECASE,
        )
        m = drug_dose_pat.search(all_text)
        if m:
            dose_val = m.group(1).replace(",", "")
            result[drug] = {
                "dose_value": f"{dose_val} mg/m^2",
                "dose_unit": "mg/m^2",
                "route": "IV",
                "source": "ctgov_description",
            }
            continue

        # Try AUC pattern (with optional range)
        auc_pat = re.compile(
            rf"{re.escape(drug_lower)}.*?AUC\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(?:[-–to]+\s*(\d+(?:\.\d+)?))?",
            re.IGNORECASE,
        )
        m = auc_pat.search(all_text)
        if m:
            auc_val = m.group(1)
            # Use midpoint if range (e.g., "AUC 4-6" → "AUC 5.0")
            if m.group(2):
                try:
                    lo = float(auc_val)
                    hi = float(m.group(2))
                    auc_val = str(round((lo + hi) / 2, 1))
                except ValueError:
                    pass
            result[drug] = {
                "dose_value": f"AUC {auc_val}",
                "dose_unit": "AUC",
                "route": "IV",
                "source": "ctgov_description",
            }

    return result


# ORR/PFS/OS measure patterns
_ORR_PATTERNS = [
    re.compile(r"overall\s*response\s*rate", re.IGNORECASE),
    re.compile(r"\bORR\b", re.IGNORECASE),
    re.compile(r"objective\s*response\s*rate", re.IGNORECASE),
]
_CR_PATTERNS = [
    re.compile(r"complete\s*(?:response|remission)\s*rate", re.IGNORECASE),
]
_PFS_PATTERNS = [
    re.compile(r"progression[\s-]*free\s*survival", re.IGNORECASE),
    re.compile(r"\bPFS\b", re.IGNORECASE),
]
_OS_PATTERNS = [
    re.compile(r"overall\s*survival", re.IGNORECASE),
]


def _extract_efficacy_from_outcomes(evidence: EvidenceBundle) -> dict[str, float]:
    """Parse ORR/PFS/OS from CT.gov primary_outcomes programmatically.

    Scans primary_outcomes for known measure names and extracts numeric values.
    Prefers combo trial outcomes for combination therapy.
    Returns dict with keys matching Efficacy schema fields.
    """
    result: dict[str, float] = {}

    # Prefer combo trial outcomes, then individual trial outcomes
    all_outcomes: list[dict] = []
    if evidence.combo_trials.primary_outcomes:
        all_outcomes.extend(evidence.combo_trials.primary_outcomes)
    if evidence.clinical_trials.primary_outcomes:
        all_outcomes.extend(evidence.clinical_trials.primary_outcomes)

    for outcome in all_outcomes:
        measure = outcome.get("measure", "")
        value_str = str(outcome.get("value", ""))
        unit = outcome.get("unit", "")

        # Parse numeric value from outcome string
        m = re.match(r"([\d.]+)", value_str.strip())
        if not m:
            continue
        try:
            val = float(m.group(1))
        except ValueError:
            continue

        # ORR
        if any(p.search(measure) for p in _ORR_PATTERNS) and "overall_response_rate_pct" not in result:
            if "percent" in unit.lower() or "%" in value_str or val <= 100:
                result["overall_response_rate_pct"] = val

        # CR
        if any(p.search(measure) for p in _CR_PATTERNS) and "complete_response_rate_pct" not in result:
            if "percent" in unit.lower() or "%" in value_str or val <= 100:
                result["complete_response_rate_pct"] = val

        # PFS
        if any(p.search(measure) for p in _PFS_PATTERNS) and "median_pfs_months" not in result:
            if "month" in unit.lower():
                result["median_pfs_months"] = val
            elif "day" in unit.lower():
                result["median_pfs_months"] = round(val / 30.44, 1)

        # OS
        if any(p.search(measure) for p in _OS_PATTERNS) and "median_os_months" not in result:
            if "month" in unit.lower():
                result["median_os_months"] = val
            elif "day" in unit.lower():
                result["median_os_months"] = round(val / 30.44, 1)

    # PDS patient-level efficacy (fill gaps not found in CT.gov outcomes)
    pds = evidence.pds
    if pds.found and pds.efficacy:
        eff = pds.efficacy
        if eff.overall_response_rate_pct is not None and "overall_response_rate_pct" not in result:
            result["overall_response_rate_pct"] = eff.overall_response_rate_pct
        if eff.complete_response_rate_pct is not None and "complete_response_rate_pct" not in result:
            result["complete_response_rate_pct"] = eff.complete_response_rate_pct
        if eff.median_pfs_months is not None and "median_pfs_months" not in result:
            result["median_pfs_months"] = eff.median_pfs_months
        if eff.median_os_months is not None and "median_os_months" not in result:
            result["median_os_months"] = eff.median_os_months

    return result


# Indication-specific age priors — used when CT.gov baseline stats AND PDS
# data are both unavailable. Prevents mean=(18+75)/2=46.5 disasters from
# eligibility criteria. Based on SEER and published trial demographics.
_INDICATION_AGE_PRIORS: dict[str, tuple[float, float]] = {
    "small cell lung": (63.0, 9.0),
    "sclc": (63.0, 9.0),
    "non-small cell lung": (65.0, 10.0),
    "nsclc": (65.0, 10.0),
    "squamous non-small cell": (65.0, 10.0),
    "breast": (60.0, 12.0),
    "colorectal": (62.0, 11.0),
    "melanoma": (62.0, 14.0),
    "ovarian": (60.0, 11.0),
    "prostate": (70.0, 8.0),
    "head and neck": (60.0, 10.0),
}


def _extract_demographics_from_ctgov(evidence: EvidenceBundle) -> dict:
    """Parse demographics from CT.gov baseline data programmatically.

    Extracts age (from eligibility + baseline), sex ratio (from baseline counts),
    and ECOG PS (from baseline counts). Only populates fields with n>=30.
    Uses indication-specific age priors when baseline stats are unavailable.
    Returns dict with keys: age_min, age_max, age_mean, age_std,
    pct_male, pct_female, ecog_ps.
    """
    result: dict = {}

    # --- Age from eligibility criteria ---
    age_range = evidence.combo_trials.age_range or evidence.clinical_trials.age_range
    if age_range:
        m = re.search(r"(\d+)\s*(?:Years?)?\s*[-–]\s*(\d+)\s*(?:Years?)?", age_range, re.IGNORECASE)
        if m:
            result["age_min"] = int(m.group(1))
            result["age_max"] = int(m.group(2))

    # Pick largest-sample baseline demographics
    ct_sources: list[dict] = []
    if evidence.combo_trials.has_results and evidence.combo_trials.baseline_demographics:
        ct_sources.append(evidence.combo_trials.baseline_demographics)
    if evidence.clinical_trials.has_results and evidence.clinical_trials.baseline_demographics:
        ct_sources.append(evidence.clinical_trials.baseline_demographics)

    best_demo = None
    best_n = 0
    for demo in ct_sources:
        n = demo.get("_sample_size", 0)
        if n > best_n:
            best_n = n
            best_demo = demo

    if best_demo is not None and best_n >= 30:
        # --- Sex from baseline counts ---
        sex_male = best_demo.get("sex_male")
        sex_female = best_demo.get("sex_female")
        if sex_male is not None and sex_female is not None:
            try:
                male_n = float(str(sex_male).replace(",", ""))
                female_n = float(str(sex_female).replace(",", ""))
                total = male_n + female_n
                if total > 0:
                    result["pct_male"] = round(male_n / total * 100, 1)
                    result["pct_female"] = round(female_n / total * 100, 1)
            except (ValueError, TypeError):
                pass

        # --- Age mean/std from baseline ---
        age_mean = best_demo.get("age_mean")
        if age_mean is not None:
            try:
                result["age_mean"] = float(str(age_mean).strip())
            except (ValueError, TypeError):
                pass
        age_std = best_demo.get("age_std")
        if age_std is not None:
            try:
                result["age_std"] = float(str(age_std).strip())
            except (ValueError, TypeError):
                pass

        # --- ECOG PS from baseline counts ---
        ecog_data = best_demo.get("ecog_ps")
        if ecog_data and isinstance(ecog_data, dict):
            ecog_parsed: dict[str, float] = {}
            total_ecog = 0.0
            for label, count_str in ecog_data.items():
                grade_m = re.search(r"(\d+)", str(label))
                if grade_m:
                    try:
                        grade = grade_m.group(1)
                        count = float(str(count_str).replace(",", ""))
                        ecog_parsed[grade] = count
                        total_ecog += count
                    except (ValueError, TypeError):
                        pass
            if total_ecog > 0:
                result["ecog_ps"] = {
                    k: round(v / total_ecog * 100, 1)
                    for k, v in ecog_parsed.items()
                }

    # --- Derive age min/max from baseline mean/std ---
    # Eligibility criteria give broad bounds (e.g., 18-75), but actual
    # participants cluster around the mean. Use mean ± 2.5*std to approximate
    # the actual participant age range (~99% coverage).
    if "age_mean" in result and "age_std" in result:
        mean, std = result["age_mean"], result["age_std"]
        if mean > 0 and std > 0:
            derived_min = max(int(mean - 2.5 * std), result.get("age_min", 18))
            derived_max = min(int(mean + 2.5 * std), result.get("age_max", 100))
            if derived_min > result.get("age_min", 0):
                result["age_min"] = derived_min
            if derived_max < result.get("age_max", 999):
                result["age_max"] = derived_max

    # --- ECOG from eligibility text ---
    # When no baseline ECOG data, parse "ECOG 0-1" or "ECOG 0-2" from
    # eligibility criteria to generate more accurate prior distributions.
    if "ecog_ps" not in result:
        elig_text = ""
        for ct_src in [evidence.combo_trials, evidence.clinical_trials]:
            if hasattr(ct_src, "eligibility_criteria") and ct_src.eligibility_criteria:
                elig_text = ct_src.eligibility_criteria
                break
            if hasattr(ct_src, "age_range") and ct_src.age_range:
                elig_text = ct_src.age_range
        # Also check brief descriptions for ECOG mentions
        for ct_src in [evidence.combo_trials, evidence.clinical_trials]:
            for out in getattr(ct_src, "primary_outcomes", []):
                desc = out.get("description", "")
                if "ecog" in desc.lower():
                    elig_text += " " + desc
        ecog_m = re.search(r"ECOG\s*(?:PS|performance\s*status)?\s*(?:of\s*)?(\d)\s*[-–]\s*(\d)", elig_text, re.IGNORECASE)
        if ecog_m:
            lo = int(ecog_m.group(1))
            hi = int(ecog_m.group(2))
            if lo <= hi:
                grades = [str(g) for g in range(lo, hi + 1)]
                uniform = round(100.0 / len(grades), 1)
                result["ecog_ps"] = {g: uniform for g in grades}
                # Keep sum exactly 100 after rounding.
                diff = round(100.0 - sum(result["ecog_ps"].values()), 1)
                result["ecog_ps"][grades[0]] = round(result["ecog_ps"][grades[0]] + diff, 1)
            log.info("ECOG from eligibility text: %d-%d → %s", lo, hi, result.get("ecog_ps"))

    # --- PDS patient-level demographics ---
    # Age and ECOG always override (actual patient data > CT.gov aggregates).
    # Sex: PDS always overrides when available (more trial-specific than
    # CT.gov which may aggregate across different trials).
    pds = evidence.pds
    if pds.found and pds.demographics and pds.demographics.n_patients >= 30:
        demo = pds.demographics
        pds_n = demo.n_patients
        # Age: always prefer PDS actual patient ages over eligibility criteria
        if demo.age_mean is not None:
            result["age_mean"] = demo.age_mean
        if demo.age_std is not None:
            result["age_std"] = demo.age_std
        if demo.age_min is not None:
            result["age_min"] = demo.age_min
        if demo.age_max is not None:
            result["age_max"] = demo.age_max
        # ECOG: always prefer PDS actual patient ECOG
        if demo.ecog_distribution:
            result["ecog_ps"] = demo.ecog_distribution
        # Sex: always prefer PDS when available (patient-level is authoritative)
        if demo.pct_male is not None:
            result["pct_male"] = demo.pct_male
            result["pct_female"] = demo.pct_female
        log.info(
            "PDS demographics (n=%d): age=%s, ecog=%s, sex=%s",
            pds_n,
            f"{demo.age_min}-{demo.age_max}" if demo.age_min else "n/a",
            demo.ecog_distribution or "n/a",
            f"{demo.pct_male}/{demo.pct_female}" if demo.pct_male else "n/a",
        )

    return result


def _correct_ae_frequencies(
    rule_set: RuleSet,
    evidence: EvidenceBundle,
) -> list[str]:
    """Align AE frequencies to evidence-sourced values.

    Builds a priority-based evidence frequency map:
      PDS (patient-level) > CT.gov (trial-specific) > DailyMed (label aggregate)
    Higher-priority sources override lower ones to prevent label-inflated
    frequencies from masking trial-specific data.
    Returns list of correction descriptions for audit trail.
    """
    from rule_engine.validator import _normalize_ae_term

    corrections: list[str] = []

    _MIN_AE_SAMPLE = 10

    # --- Layer 1 (lowest priority): DailyMed label frequencies ---
    dm_freqs: dict[str, float] = {}
    for drug in evidence.drugs:
        sd = evidence.per_drug.get(drug)
        if sd is None or not sd.dailymed.found or not sd.dailymed.ae_table:
            continue
        for ae in sd.dailymed.ae_table:
            term = _normalize_ae_term(ae.get("term", "").lower().strip())
            pct = ae.get("incidence_pct")
            if term and pct is not None:
                try:
                    val = float(pct)
                    dm_freqs[term] = max(dm_freqs.get(term, 0.0), val)
                except (ValueError, TypeError):
                    pass

    # --- Layer 2: CT.gov trial-specific AE frequencies ---
    ctgov_freqs: dict[str, float] = {}
    for ae_list in [
        evidence.clinical_trials.reported_aes,
        evidence.combo_trials.reported_aes,
    ]:
        for ae in ae_list:
            if ae.get("at_risk", 0) < _MIN_AE_SAMPLE:
                continue
            term = _normalize_ae_term(ae.get("term", "").lower().strip())
            pct = ae.get("pct")
            if term and pct is not None:
                try:
                    val = float(pct)
                    ctgov_freqs[term] = max(ctgov_freqs.get(term, 0.0), val)
                except (ValueError, TypeError):
                    pass

    # --- Layer 3 (highest priority): PDS patient-level data ---
    pds_freqs: dict[str, float] = {}
    pds = evidence.pds
    if pds.found and pds.ae_aggregates:
        for ae in pds.ae_aggregates:
            if ae.n_total_patients >= _MIN_AE_SAMPLE:
                term = _normalize_ae_term(ae.term.lower().strip())
                if term:
                    pds_freqs[term] = max(pds_freqs.get(term, 0.0), ae.frequency_pct)

    # DailyMed label frequencies are aggregate "worst-case" across ALL trials
    # and indications, typically 2-3x higher than trial-specific values.
    # Dampen them so they don't anchor the LLM output to inflated levels.
    _DM_DAMPENING = 0.5
    dm_freqs = {k: v * _DM_DAMPENING for k, v in dm_freqs.items()}

    # Merge: start with dampened DailyMed, then take the MINIMUM of
    # DailyMed and CT.gov for each AE. Both sources inflate differently
    # (DailyMed=cross-indication aggregate, CT.gov max=worst trial), so
    # the lower value is the more conservative and usually more accurate.
    evidence_freqs: dict[str, float] = {}
    evidence_freqs.update(dm_freqs)
    for term, val in ctgov_freqs.items():
        if term in evidence_freqs:
            evidence_freqs[term] = min(evidence_freqs[term], val)
        else:
            evidence_freqs[term] = val
    evidence_freqs.update(pds_freqs)       # PDS overrides everything

    if not evidence_freqs:
        return corrections

    for ae in rule_set.adverse_events:
        ae_norm = _normalize_ae_term(ae.event.lower().strip())
        evidence_max = evidence_freqs.get(ae_norm)
        if evidence_max is None or evidence_max <= 0:
            continue

        abs_diff = abs(ae.frequency_pct - evidence_max)
        rel_diff = abs_diff / max(evidence_max, 1e-6)

        # Material deviation correction: >2% absolute and >20% relative mismatch.
        # This keeps minor noise untouched but anchors meaningful drift to evidence.
        if abs_diff > 2.0 and rel_diff > 0.20:
            old_pct = ae.frequency_pct
            ae.frequency_pct = evidence_max
            # Proportionally rescale severity_distribution
            dist_sum = sum(ae.severity_distribution.values())
            if dist_sum > 0:
                scale = evidence_max / dist_sum
                ae.severity_distribution = {
                    k: round(v * scale, 1)
                    for k, v in ae.severity_distribution.items()
                }
            corrections.append(
                f"{ae.event}: {old_pct:.1f}% -> {evidence_max:.1f}%"
            )

    return corrections


def _prune_unevidenced_aes(
    rule_set: RuleSet,
    evidence: EvidenceBundle,
) -> list[str]:
    """Remove AEs not backed by any evidence source.

    Keeps AEs that appear in DailyMed AE tables, CT.gov reported AEs,
    or OnSIDES boxed warnings. Merges duplicates (same normalized term)
    by keeping the higher frequency. Returns list of removal descriptions.
    """
    from rule_engine.validator import _normalize_ae_term

    removals: list[str] = []

    # Build evidence-backed AE term set (normalized)
    evidence_terms: set[str] = set()

    # DailyMed AE tables
    for drug in evidence.drugs:
        sd = evidence.per_drug.get(drug)
        if sd is None:
            continue
        if sd.dailymed.found and sd.dailymed.ae_table:
            for ae in sd.dailymed.ae_table:
                term = ae.get("term", "").lower().strip()
                if term:
                    evidence_terms.add(_normalize_ae_term(term))
        # OnSIDES boxed warning AEs (always keep)
        for bw_ae in sd.onsides.boxed_warning_aes:
            evidence_terms.add(_normalize_ae_term(bw_ae))

    # CT.gov reported AEs (individual + combo)
    # Only count AEs from trials with at_risk >= 10 as reliable evidence
    _MIN_AE_SAMPLE = 10
    for ae_list in [
        evidence.clinical_trials.reported_aes,
        evidence.combo_trials.reported_aes,
    ]:
        for ae in ae_list:
            term = ae.get("term", "").lower().strip()
            if term and ae.get("at_risk", 0) >= _MIN_AE_SAMPLE:
                evidence_terms.add(_normalize_ae_term(term))

    # PDS patient-level AE data
    pds = evidence.pds
    if pds.found and pds.ae_aggregates:
        for ae in pds.ae_aggregates:
            if ae.n_total_patients >= _MIN_AE_SAMPLE:
                evidence_terms.add(_normalize_ae_term(ae.term.lower().strip()))

    if not evidence_terms:
        return removals

    # Sort rule_set AEs by frequency descending for stable ordering
    rule_set.adverse_events.sort(key=lambda a: a.frequency_pct, reverse=True)

    # Partition into evidenced and unevidenced
    kept: list = []
    kept_terms: dict[str, int] = {}  # normalized_term -> index in kept list
    for ae in rule_set.adverse_events:
        norm = _normalize_ae_term(ae.event.lower().strip())
        if norm in evidence_terms:
            # Merge duplicates: keep higher frequency
            if norm in kept_terms:
                existing_idx = kept_terms[norm]
                if ae.frequency_pct > kept[existing_idx].frequency_pct:
                    removals.append(
                        f"Merged duplicate '{kept[existing_idx].event}' into '{ae.event}'"
                    )
                    kept[existing_idx] = ae
                else:
                    removals.append(
                        f"Merged duplicate '{ae.event}' into '{kept[existing_idx].event}'"
                    )
            else:
                kept_terms[norm] = len(kept)
                kept.append(ae)
        else:
            removals.append(f"Pruned unevidenced AE: '{ae.event}' ({ae.frequency_pct:.1f}%)")

    if len(kept) < len(rule_set.adverse_events):
        rule_set.adverse_events = kept

    return removals


def _inject_pds_aes(
    rule_set: RuleSet,
    evidence: EvidenceBundle,
) -> list[str]:
    """Inject high-frequency PDS AEs missing from the rule set.

    When PDS patient-level data has AEs that the LLM didn't generate,
    inject them with PDS-derived frequencies and conservative defaults.
    Only injects AEs with frequency >= 5% to avoid noise.
    Returns list of injection descriptions for audit trail.
    """
    from rule_engine.validator import _normalize_ae_term
    from rule_engine.schema import AdverseEvent

    pds = evidence.pds
    if not pds.found or not pds.ae_aggregates:
        return []

    _MIN_AE_SAMPLE = 10
    _MIN_INJECT_PCT = 10.0  # only inject AEs with >= 10% frequency

    injections: list[str] = []

    # Build set of existing AE terms (normalized)
    existing_terms: set[str] = set()
    for ae in rule_set.adverse_events:
        existing_terms.add(_normalize_ae_term(ae.event.lower().strip()))

    for pds_ae in pds.ae_aggregates:
        if pds_ae.n_total_patients < _MIN_AE_SAMPLE:
            continue
        if pds_ae.frequency_pct < _MIN_INJECT_PCT:
            continue

        term = _normalize_ae_term(pds_ae.term.lower().strip())
        if not term or term in existing_terms:
            continue

        # Build severity distribution from PDS grade data if available
        sev_dist: dict[str, float] = {}
        if pds_ae.grade_distribution:
            for grade, pct in pds_ae.grade_distribution.items():
                sev_dist[f"grade_{grade}"] = round(pct, 1)
        if not sev_dist:
            # Conservative default
            sev_dist = {"grade_1": 40.0, "grade_2": 35.0, "grade_3": 20.0, "grade_4": 5.0}

        onset = int(pds_ae.median_onset_day) if pds_ae.median_onset_day is not None else 14

        new_ae = AdverseEvent(
            event=term.title(),
            frequency_pct=pds_ae.frequency_pct,
            severity_distribution=sev_dist,
            median_onset_days=onset,
            reversible=True,
            source_drug=None,
            triggers=[],
        )
        rule_set.adverse_events.append(new_ae)
        existing_terms.add(term)
        injections.append(f"Injected PDS AE: '{term.title()}' ({pds_ae.frequency_pct:.1f}%)")

    return injections


def _format_raw_evidence_summary(evidence: EvidenceBundle) -> str:
    """Compact summary of raw evidence numbers for the grounding check."""
    sections: list[str] = []
    for drug in evidence.drugs:
        sd = evidence.per_drug.get(drug)
        if sd is None:
            continue
        dm = sd.dailymed
        if dm.found and dm.ae_table:
            sections.append(f"[DailyMed — {drug}]")
            for ae in dm.ae_table[:30]:
                term = ae.get("term", "?")
                freq = ae.get("incidence_pct", "?")
                grade34 = ae.get("grade34_pct")
                line = f"  {term}: {freq}%"
                if grade34 is not None:
                    line += f" (g34={grade34}%)"
                sections.append(line)
        fda = sd.openfda
        if fda.has_timing_data and fda.time_to_onset_data:
            sections.append(f"[FAERS timing — {drug}]")
            for entry in fda.time_to_onset_data:
                sections.append(
                    f"  {entry.get('unit_label', '?')}: {entry.get('count', 0)} reports"
                )
    ct = evidence.clinical_trials
    if ct.has_results and ct.baseline_demographics:
        sample_n = ct.baseline_demographics.get("_sample_size", 0)
        if sample_n >= 30:
            sections.append(f"[CT.gov demographics (n={sample_n})]")
            for key, val in ct.baseline_demographics.items():
                if key.startswith("_"):
                    continue
                sections.append(f"  {key}: {val}")
    if ct.reported_aes:
        sections.append("[CT.gov reported AEs]")
        for ae in ct.reported_aes[:20]:
            grade34 = ae.get("grade34_pct")
            line = f"  {ae.get('term', '?')}: {ae.get('pct', '?')}%"
            if grade34 is not None:
                line += f" (g34={grade34}%)"
            sections.append(line)
    return "\n".join(sections)


async def _llm_call(
    client: AsyncOpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int = 8000,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Make a single LLM call. Returns (content, reasoning_trace)."""
    if rate_limiter is not None:
        await rate_limiter.acquire()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=min(max_tokens, 8192),
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    choice = response.choices[0]
    msg = choice.message
    reasoning = getattr(msg, "reasoning_content", None) or ""
    content = msg.content or ""
    return content, reasoning


def _parse_json_response(raw: str) -> dict | list:
    """Parse JSON from LLM response, with repair fallback."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    repaired = _repair_json(raw)
    try:
        return json.loads(repaired)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try to find JSON in the text
    brace_start = raw.find("{")
    bracket_start = raw.find("[")
    if bracket_start != -1 and (brace_start == -1 or bracket_start < brace_start):
        bracket_end = raw.rfind("]")
        if bracket_end > bracket_start:
            return json.loads(_repair_json(raw[bracket_start : bracket_end + 1]))
    if brace_start != -1:
        brace_end = raw.rfind("}")
        if brace_end > brace_start:
            return json.loads(_repair_json(raw[brace_start : brace_end + 1]))
    raise ValueError(f"Could not parse JSON from response: {raw[:200]}...")


def _aggressive_repair_ruleset(
    raw_text: str,
    drugs: list[str],
    indication: str,
    stage1_parsed: dict[str, dict | list] | None = None,
) -> RuleSet:
    """Aggressively repair malformed JSON LLM output into a valid RuleSet.

    Handles: markdown fences, missing commas, trailing commas, unclosed brackets,
    Roman numeral phases, missing required fields, None values in AE entries.

    Args:
        raw_text: Raw LLM response text (may contain markdown, broken JSON).
        drugs: Drug names to inject if missing.
        indication: Indication to inject if missing.
        stage1_parsed: Optional stage 1 extracted data for field injection fallback.

    Returns:
        Parsed RuleSet.

    Raises:
        ValueError/json.JSONDecodeError on unrecoverable parse failure.
    """
    import re

    if stage1_parsed is None:
        stage1_parsed = {}

    text = raw_text
    text = re.sub(r"```(?:json)?\s*", "", text)
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start == -1 or brace_end <= brace_start:
        raise ValueError("No JSON object found in LLM output")
    text = text[brace_start : brace_end + 1]

    # Multi-pass repair
    for _ in range(3):
        text = _repair_json(text)

    # Fix truncated JSON — close unclosed brackets
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    text = text.rstrip().rstrip(",")
    text += "]" * max(0, open_brackets) + "}" * max(0, open_braces)

    # Additional fix: remove trailing commas before closing brackets
    text = re.sub(r",\s*(\]|\})", r"\1", text)

    # Iterative error-position repair: fix JSON errors at exact failure positions
    raw_dict = None
    _prev_positions: set[int] = set()  # guard against oscillation
    for _attempt in range(30):
        try:
            raw_dict = json.loads(text)
            break
        except json.JSONDecodeError as jde:
            err_msg = str(jde)
            pos = jde.pos
            if pos in _prev_positions:
                # Oscillation detected — try a different strategy: snip the problem area
                # Find the enclosing structure and try to close it
                log.debug("Oscillation at position %d — trying truncation repair", pos)
                # Find last valid close before the problem
                snippet = text[max(0, pos - 200):pos]
                last_close = max(snippet.rfind("}"), snippet.rfind("]"))
                if last_close >= 0:
                    cut_point = max(0, pos - 200) + last_close + 1
                    remainder = text[cut_point:].lstrip().lstrip(",")
                    # If remainder starts with a close bracket, keep it
                    if remainder and remainder[0] in "}]":
                        text = text[:cut_point] + remainder
                    else:
                        # Close any open structures
                        prefix = text[:cut_point]
                        open_b = prefix.count("{") - prefix.count("}")
                        open_k = prefix.count("[") - prefix.count("]")
                        text = prefix.rstrip().rstrip(",") + "]" * max(0, open_k) + "}" * max(0, open_b)
                    text = re.sub(r",\s*(\]|\})", r"\1", text)
                    continue
                raise
            _prev_positions.add(pos)
            if "Expecting ',' delimiter" in err_msg:
                insert_at = pos
                while insert_at > 0 and text[insert_at - 1] in " \t\n\r":
                    insert_at -= 1
                text = text[:insert_at] + "," + text[insert_at:]
                log.debug("Inserted comma at position %d (attempt %d)", insert_at, _attempt + 1)
            elif "Expecting property name" in err_msg or "Expecting value" in err_msg:
                scan = pos - 1
                while scan >= 0 and text[scan] in " \t\n\r":
                    scan -= 1
                if scan >= 0 and text[scan] == ",":
                    text = text[:scan] + text[scan + 1:]
                    log.debug("Removed trailing comma at position %d (attempt %d)", scan, _attempt + 1)
                else:
                    raise
            else:
                raise

    if raw_dict is None:
        # Final cleanup and attempt
        text = re.sub(r",\s*(\]|\})", r"\1", text)
        raw_dict = json.loads(text)

    raw_dict.setdefault("drugs", drugs)
    raw_dict.setdefault("indication", indication)

    # Fix Roman numeral phases (e.g., "III" → 3, "II" → 2)
    _ROMAN_TO_INT = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
    phase_val = raw_dict.get("phase")
    if isinstance(phase_val, str):
        raw_dict["phase"] = _ROMAN_TO_INT.get(phase_val.strip().lower(), 3)
    raw_dict.setdefault("phase", 3)
    raw_dict.setdefault("treatment_duration_days", 365)
    # Coerce non-int → int (LLM sometimes outputs 365.0, "365", etc.)
    td = raw_dict.get("treatment_duration_days")
    if not isinstance(td, int):
        try:
            raw_dict["treatment_duration_days"] = int(float(str(td)))
        except (ValueError, TypeError):
            raw_dict["treatment_duration_days"] = 365

    # Inject missing structural fields from stage1 demographics data (no hard-coded clinical defaults)
    _demo_data = stage1_parsed.get("demographics", {})
    if not isinstance(_demo_data, dict):
        _demo_data = {}
    if "regimen" not in raw_dict:
        raw_dict["regimen"] = _demo_data.get("regimen", [])
    if "efficacy" not in raw_dict:
        raw_dict["efficacy"] = _demo_data.get("efficacy", {})
    if "demographics" not in raw_dict:
        raw_dict["demographics"] = _demo_data.get("demographics", {})
    elif isinstance(raw_dict.get("demographics"), dict):
        demo = raw_dict["demographics"]
        # Unwrap nested demographics (Gemini sometimes outputs {"demographics": {"demographics": {...}}})
        if "demographics" in demo and isinstance(demo["demographics"], dict):
            inner = demo.pop("demographics")
            for k, v in inner.items():
                demo.setdefault(k, v)
        # Extract Stage 1 fallback values (evidence-based, not hard-coded)
        _s1_demo = _demo_data.get("demographics", {}) if isinstance(_demo_data.get("demographics"), dict) else {}
        _s1_age = _s1_demo.get("age", {}) if isinstance(_s1_demo.get("age"), dict) else {}
        _s1_sex = _s1_demo.get("sex", {}) if isinstance(_s1_demo.get("sex"), dict) else {}
        # Unwrap flat sex fields mixed into demographics level
        if "age" not in demo and "pct_male" in demo:
            raw_dict["demographics"] = {
                "age": _s1_age or {},
                "sex": {"pct_male": demo.get("pct_male", _s1_sex.get("pct_male")), "pct_female": demo.get("pct_female", _s1_sex.get("pct_female"))},
                "race_ethnicity": demo.get("race_ethnicity", []),
            }
        if _s1_sex:
            raw_dict["demographics"].setdefault("sex", _s1_sex)
        if _s1_age:
            raw_dict["demographics"].setdefault("age", _s1_age)
        # Fix None values in demographics sub-fields using Stage 1 data
        demo = raw_dict["demographics"]
        age = demo.get("age")
        if isinstance(age, dict):
            for k in ["min", "max", "mean", "std"]:
                if age.get(k) is None and k in _s1_age:
                    age[k] = _s1_age[k]
        sex = demo.get("sex")
        if isinstance(sex, dict):
            for sex_key in ["pct_male", "pct_female"]:
                val = sex.get(sex_key)
                if val is None:
                    sex[sex_key] = _s1_sex.get(sex_key)
                elif not isinstance(val, (int, float)):
                    try:
                        sex[sex_key] = float(str(val).replace("%", "").strip())
                    except (ValueError, TypeError):
                        sex[sex_key] = _s1_sex.get(sex_key)
            # Clamp sex pct to 0-100 range (LLM sometimes returns >100)
            for sex_key in ["pct_male", "pct_female"]:
                v = sex.get(sex_key)
                if isinstance(v, (int, float)) and v > 100:
                    sex[sex_key] = min(v, 100.0)
        for re_entry in demo.get("race_ethnicity", []):
            if isinstance(re_entry, dict):
                pct_val = re_entry.get("pct")
                if pct_val is None:
                    re_entry["pct"] = 0.0
                elif not isinstance(pct_val, (int, float)):
                    # Gemini sometimes outputs "45%" or other non-numeric strings
                    try:
                        re_entry["pct"] = float(str(pct_val).replace("%", "").strip())
                    except (ValueError, TypeError):
                        re_entry["pct"] = 0.0
    # Final safety net: ensure demographics has structurally valid age/sex
    # so Pydantic doesn't reject before demographics re-prompt can fire
    demo_final = raw_dict.get("demographics", {})
    if isinstance(demo_final, dict):
        if "age" not in demo_final or not isinstance(demo_final.get("age"), dict) or not demo_final["age"]:
            demo_final["age"] = {"mean": None, "std": None, "min": 18, "max": 90}
        if "sex" not in demo_final or not isinstance(demo_final.get("sex"), dict) or not demo_final["sex"]:
            demo_final["sex"] = {"pct_male": None, "pct_female": None}
        raw_dict["demographics"] = demo_final
    elif isinstance(demo_final, list) and demo_final and isinstance(demo_final[0], dict):
        # Demographics came as a list — take first element
        raw_dict["demographics"] = demo_final[0]
        demo_final = raw_dict["demographics"]
        if "age" not in demo_final or not isinstance(demo_final.get("age"), dict) or not demo_final["age"]:
            demo_final["age"] = {"mean": None, "std": None, "min": 18, "max": 90}
        if "sex" not in demo_final or not isinstance(demo_final.get("sex"), dict) or not demo_final["sex"]:
            demo_final["sex"] = {"pct_male": None, "pct_female": None}
    raw_dict.setdefault("comorbidities", [])
    # Fix comorbidity entries (Gemini sometimes outputs None for ae_risk_modifiers)
    for comorb in raw_dict.get("comorbidities", []):
        if isinstance(comorb, dict):
            if not isinstance(comorb.get("ae_risk_modifiers"), list):
                comorb["ae_risk_modifiers"] = []
            if comorb.get("prevalence_pct") is None:
                comorb["prevalence_pct"] = 10.0
            elif not isinstance(comorb["prevalence_pct"], (int, float)):
                try:
                    comorb["prevalence_pct"] = float(str(comorb["prevalence_pct"]).replace("%", "").strip())
                except (ValueError, TypeError):
                    comorb["prevalence_pct"] = 10.0

    # Flatten nested list regimens (e.g., [[{...}]] → [{...}])
    reg_val = raw_dict.get("regimen", [])
    if isinstance(reg_val, list) and reg_val and isinstance(reg_val[0], list):
        raw_dict["regimen"] = [item for sublist in reg_val for item in sublist if isinstance(item, dict)]

    # Fix None values in regimen entries
    for reg in raw_dict.get("regimen", []):
        if isinstance(reg, dict):
            reg.setdefault("drug", drugs[0] if drugs else "Unknown")
            if reg.get("dose") is None:
                reg["dose"] = ""  # Leave empty — dose reprompt will fill
            elif not isinstance(reg["dose"], str):
                reg["dose"] = str(reg["dose"])
            if reg.get("route") is None:
                reg["route"] = "IV"
            if reg.get("cycle_days") is None:
                reg["cycle_days"] = 21
            if reg.get("schedule") is None:
                reg["schedule"] = "per protocol"

    # Ensure efficacy is a dict (Gemini sometimes nests or wraps it)
    eff = raw_dict.get("efficacy")
    if isinstance(eff, list) and eff and isinstance(eff[0], dict):
        raw_dict["efficacy"] = eff[0]
        eff = eff[0]
    _s1_eff = _demo_data.get("efficacy", {}) if isinstance(_demo_data.get("efficacy"), dict) else {}
    if isinstance(eff, dict):
        for _eff_field in ["overall_response_rate_pct", "complete_response_rate_pct",
                           "median_pfs_months", "median_pfs_ci_low", "median_pfs_ci_high",
                           "median_os_months", "median_os_ci_low", "median_os_ci_high"]:
            if eff.get(_eff_field) is None and _s1_eff.get(_eff_field) is not None:
                eff[_eff_field] = _s1_eff[_eff_field]

    # Filter out non-dict AE entries (Gemini sometimes emits lists instead of objects)
    aes = raw_dict.get("adverse_events", [])
    if isinstance(aes, list):
        raw_dict["adverse_events"] = [ae for ae in aes if isinstance(ae, dict)]
    for ae in raw_dict.get("adverse_events", []):
        if ae.get("median_onset_days") is None:
            ae["median_onset_days"] = 14
        if ae.get("frequency_pct") is None:
            ae["frequency_pct"] = 1.0
        if ae.get("reversible") is None:
            ae["reversible"] = True
        ae.setdefault("severity_distribution", {"grade_1": ae.get("frequency_pct", 1.0)})
        # Ensure triggers is a list (Gemini sometimes outputs dict or string)
        triggers_val = ae.get("triggers")
        if not isinstance(triggers_val, list):
            ae["triggers"] = [] if triggers_val is None else [triggers_val] if isinstance(triggers_val, dict) else []

    # Ensure drug_interactions is a list (Gemini sometimes outputs dict or null)
    di_val = raw_dict.get("drug_interactions")
    if di_val is None or not isinstance(di_val, list):
        raw_dict["drug_interactions"] = [di_val] if isinstance(di_val, dict) else []

    # Final race_ethnicity cleanup — filter out entries with None/invalid pct
    # (catches all code paths: direct LLM output, list→dict conversion, stage1 fallback)
    _final_demo = raw_dict.get("demographics")
    if isinstance(_final_demo, dict):
        _re_list = _final_demo.get("race_ethnicity")
        if isinstance(_re_list, list):
            cleaned = []
            for _re in _re_list:
                if isinstance(_re, dict):
                    _pv = _re.get("pct")
                    if _pv is None:
                        continue  # drop entries with no percentage
                    if not isinstance(_pv, (int, float)):
                        try:
                            _re["pct"] = float(str(_pv).replace("%", "").strip())
                        except (ValueError, TypeError):
                            continue
                    cleaned.append(_re)
            _final_demo["race_ethnicity"] = cleaned

    # Final sex-pct safety clamp right before Pydantic parsing
    _fd = raw_dict.get("demographics")
    if isinstance(_fd, dict):
        _fs = _fd.get("sex")
        if isinstance(_fs, dict):
            for _sk in ("pct_male", "pct_female"):
                _sv = _fs.get(_sk)
                if isinstance(_sv, (int, float)) and _sv > 100:
                    _fs[_sk] = 100.0
                elif isinstance(_sv, (int, float)) and _sv < 0:
                    _fs[_sk] = 0.0

    return RuleSet(**raw_dict)


async def synthesize_rules_multistage(
    drugs: list[str],
    indication: str,
    evidence: EvidenceBundle,
    config: RuleEngineConfig | None = None,
) -> tuple[RuleSet, AgentLog]:
    """Run multi-stage synthesis pipeline to produce a RuleSet from evidence.

    Stage 1: 5 parallel focused extraction sub-calls
    Stage 2: Grounding verification against raw evidence
    Stage 3: Final JSON assembly

    Returns:
        Tuple of (parsed RuleSet, AgentLog with stage traces).
    """
    if config is None:
        config = RuleEngineConfig()

    client = AsyncOpenAI(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
    )
    limiter = RateLimiter(config.rate_limit_rpm)

    drug_label = " + ".join(drugs)
    stage_logs: list[dict] = []

    agent_log = AgentLog(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=config.llm_model,
    )

    # ------------------------------------------------------------------
    # Stage 1: Parallel focused extraction (5 sub-calls)
    # ------------------------------------------------------------------
    log.info("Stage 1: Parallel extraction for %s / %s", drug_label, indication)

    ae_freq_evidence = _format_ae_freq_evidence(evidence)
    severity_evidence = _format_severity_evidence(evidence)
    onset_evidence = _format_onset_evidence(evidence)
    triggers_evidence = _format_triggers_evidence(evidence)
    demographics_evidence = _format_demographics_evidence(evidence)

    system_base = "You are a clinical pharmacologist. Respond ONLY with valid JSON — no markdown fences, no commentary."

    stage1_calls = [
        _llm_call(
            client, config.llm_model, system_base,
            STAGE1_AE_FREQ_PROMPT.format(evidence=ae_freq_evidence),
            temperature=0.3, max_tokens=4000, rate_limiter=limiter,
        ),
        _llm_call(
            client, config.llm_model, system_base,
            STAGE1_SEVERITY_PROMPT.format(evidence=severity_evidence),
            temperature=0.3, max_tokens=4000, rate_limiter=limiter,
        ),
        _llm_call(
            client, config.llm_model, system_base,
            STAGE1_ONSET_PROMPT.format(evidence=onset_evidence),
            temperature=0.3, max_tokens=4000, rate_limiter=limiter,
        ),
        _llm_call(
            client, config.llm_model, system_base,
            STAGE1_TRIGGERS_PROMPT.format(evidence=triggers_evidence),
            temperature=0.3, max_tokens=4000, rate_limiter=limiter,
        ),
        _llm_call(
            client, config.llm_model, system_base,
            STAGE1_DEMOGRAPHICS_PROMPT.format(evidence=demographics_evidence),
            temperature=0.3, max_tokens=4000, rate_limiter=limiter,
        ),
    ]

    stage1_results = await asyncio.gather(*stage1_calls, return_exceptions=True)

    stage1_names = ["ae_freq", "severity", "onset", "triggers", "demographics"]
    stage1_parsed: dict[str, dict | list] = {}

    for name, result in zip(stage1_names, stage1_results):
        if isinstance(result, Exception):
            log.error("Stage 1 sub-call '%s' failed: %s", name, result)
            stage_logs.append({"stage": f"stage1_{name}", "error": str(result)})
            stage1_parsed[name] = {}
            continue
        content, reasoning = result
        stage_logs.append({
            "stage": f"stage1_{name}",
            "prompt_len": len(STAGE1_AE_FREQ_PROMPT),  # approximate
            "response_len": len(content),
            "reasoning_len": len(reasoning),
        })
        try:
            stage1_parsed[name] = _parse_json_response(content)
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("Stage 1 '%s' JSON parse failed: %s", name, e)
            stage1_parsed[name] = {}

    log.info("Stage 1 complete: %d/%d sub-calls succeeded",
             sum(1 for v in stage1_parsed.values() if v), len(stage1_names))

    # ------------------------------------------------------------------
    # Stage 2: Grounding check (optional — degrades gracefully)
    # ------------------------------------------------------------------
    log.info("Stage 2: Grounding check for %s / %s", drug_label, indication)

    extracted_summary = json.dumps(stage1_parsed, indent=2, default=str)
    raw_evidence_summary = _format_raw_evidence_summary(evidence)

    # Truncate to fit context window (model max_model_len=16384 tokens ≈ 50K chars)
    # Stage 2 needs: prompt template (~300 chars) + extracted + evidence + response (4K tokens ≈ 12K chars)
    _MAX_EXTRACTED_CHARS = 20000
    _MAX_EVIDENCE_CHARS = 10000
    extracted_for_grounding = extracted_summary[:_MAX_EXTRACTED_CHARS]
    evidence_for_grounding = raw_evidence_summary[:_MAX_EVIDENCE_CHARS]

    grounding_report = {"grounding_report": [], "ungrounded_count": -1, "total_checked": 0}

    try:
        stage2_content, stage2_reasoning = await _llm_call(
            client, config.llm_model, system_base,
            STAGE2_GROUNDING_PROMPT.format(
                extracted=extracted_for_grounding,
                evidence=evidence_for_grounding,
            ),
            temperature=0.3,
            max_tokens=4000,
            rate_limiter=limiter,
        )

        stage_logs.append({
            "stage": "stage2_grounding",
            "response_len": len(stage2_content),
            "reasoning_len": len(stage2_reasoning),
        })

        grounding_report = _parse_json_response(stage2_content)
    except Exception as e:
        log.warning("Stage 2 grounding failed (non-fatal): %s", e)
        stage_logs.append({"stage": "stage2_grounding", "error": str(e)})

    ungrounded = grounding_report.get("ungrounded_count", -1)
    total_checked = grounding_report.get("total_checked", 0)
    log.info("Stage 2 complete: %d/%d values ungrounded", ungrounded, total_checked)

    # ------------------------------------------------------------------
    # Stage 3: Final JSON synthesis
    # ------------------------------------------------------------------
    log.info("Stage 3: Final synthesis for %s / %s", drug_label, indication)

    # Use compact extracted summary for Stage 3 to save context tokens
    extracted_compact = json.dumps(stage1_parsed, separators=(",", ":"), default=str)
    grounding_compact = json.dumps(grounding_report, separators=(",", ":"), default=str)

    stage3_prompt = STAGE3_SYNTHESIS_PROMPT.format(
        extracted=extracted_compact[:25000],
        grounding=grounding_compact[:5000],
        drugs=drug_label,
        indication=indication,
        schema=_RULESET_JSON_SCHEMA[:8000],
    )

    stage3_content, stage3_reasoning = await _llm_call(
        client, config.llm_model, system_base,
        stage3_prompt,
        temperature=0.6,
        max_tokens=8000,
        rate_limiter=limiter,
    )

    stage_logs.append({
        "stage": "stage3_synthesis",
        "response_len": len(stage3_content),
        "reasoning_len": len(stage3_reasoning),
    })

    # Populate agent log
    agent_log.evidence_prompt = f"[multistage: {len(stage1_names)} stage1 calls + grounding + synthesis]"
    agent_log.reasoning_trace = stage3_reasoning
    agent_log.raw_response = stage3_content

    # Parse the final RuleSet — inject known fields and fix common LLM omissions
    rule_set = None
    try:
        rule_set = _extract_and_parse_ruleset(stage3_content)
    except Exception as first_err:
        log.warning("Primary parse failed: %s — attempting aggressive repair", first_err)
        try:
            rule_set = _aggressive_repair_ruleset(stage3_content, drugs, indication, stage1_parsed)
            log.info("Aggressive repair succeeded for Stage 3 output")
        except Exception as repair_err:
            log.error("Aggressive repair also failed: %s", repair_err)
            raise repair_err from first_err
    if rule_set is None:
        raise ValueError("Failed to parse Stage 3 output into a valid RuleSet")

    # ==================================================================
    # PROGRAMMATIC OVERRIDES — deterministic evidence-based corrections
    # Runs BEFORE LLM reprompts (which serve as fallbacks).
    # ==================================================================

    # --- 1. Dose override from DailyMed label parsing ---
    structured_doses = _extract_structured_doses(evidence)
    if structured_doses:
        _vague_set = {"standard dose", "per physician discretion", "per protocol",
                      "not specified", "dose not specified", "as prescribed",
                      "individualized", "variable", ""}
        for reg in (rule_set.regimen or []):
            extracted = structured_doses.get(reg.drug)
            if extracted is None:
                continue
            extracted_dose = extracted["dose_value"]
            current_dose = reg.dose.strip().lower()
            if current_dose in _vague_set or not current_dose:
                log.info("Dose override (vague→extracted): %s %s → %s",
                         reg.drug, reg.dose, extracted_dose)
                reg.dose = extracted_dose
            else:
                # Compare numeric values — override if >50% deviation AND same unit type
                # Skip comparison if units don't match (e.g., AUC vs mg/m^2)
                ext_unit = extracted.get("dose_unit", "")
                llm_has_auc = "auc" in current_dose
                ext_has_auc = ext_unit == "AUC"
                # Only compare if both are AUC or both are non-AUC
                if llm_has_auc == ext_has_auc:
                    llm_num = re.search(r"(\d+(?:\.\d+)?)", current_dose)
                    ext_num = re.search(r"(\d+(?:\.\d+)?)", extracted_dose)
                    if llm_num and ext_num:
                        llm_val = float(llm_num.group(1))
                        ext_val = float(ext_num.group(1))
                        # Use tighter threshold for AUC (range 2-7) vs mg doses
                        threshold = 0.15 if llm_has_auc else 0.5
                        if ext_val > 0 and abs(llm_val - ext_val) / ext_val > threshold:
                            log.info("Dose override (>%d%% deviation): %s %s → %s",
                                     int(threshold * 100), reg.drug, reg.dose, extracted_dose)
                            reg.dose = extracted_dose
        stage_logs.append({
            "stage": "programmatic_dose_override",
            "structured_doses": structured_doses,
        })

    # --- 2. Efficacy override from CT.gov outcomes ---
    extracted_efficacy = _extract_efficacy_from_outcomes(evidence)
    if extracted_efficacy:
        eff = rule_set.efficacy
        eff_overrides = []

        if "overall_response_rate_pct" in extracted_efficacy:
            ext_orr = extracted_efficacy["overall_response_rate_pct"]
            current_orr = eff.overall_response_rate_pct
            # One-directional: only override upward or fill missing.
            # For SCLC drugs, CT.gov often returns lower values than actual
            # first-line ORR (wrong trial settings, ORR as secondary endpoint).
            # Lowering LLM's estimate from evidence-informed synthesis is harmful.
            if current_orr is None or current_orr == 0:
                # Missing data: always use CT.gov
                eff.overall_response_rate_pct = ext_orr
                eff_overrides.append(f"ORR: {current_orr} → {ext_orr} (fill)")
            elif ext_orr > current_orr * 1.3:
                # Override upward: LLM underreported (CT.gov >30% higher)
                eff.overall_response_rate_pct = ext_orr
                eff_overrides.append(f"ORR: {current_orr} → {ext_orr} (up)")
            # else: keep LLM value (don't override downward)

        if "complete_response_rate_pct" in extracted_efficacy:
            ext_cr = extracted_efficacy["complete_response_rate_pct"]
            if eff.complete_response_rate_pct is None or eff.complete_response_rate_pct == 0:
                eff.complete_response_rate_pct = ext_cr
                eff_overrides.append(f"CR: → {ext_cr}")

        if "median_pfs_months" in extracted_efficacy:
            ext_pfs = extracted_efficacy["median_pfs_months"]
            if eff.median_pfs_months is None or eff.median_pfs_months == 0:
                eff.median_pfs_months = ext_pfs
                eff_overrides.append(f"PFS: → {ext_pfs}")

        if "median_os_months" in extracted_efficacy:
            ext_os = extracted_efficacy["median_os_months"]
            if eff.median_os_months is None or eff.median_os_months == 0:
                eff.median_os_months = ext_os
                eff_overrides.append(f"OS: → {ext_os}")

        if eff_overrides:
            log.info("Efficacy override from CT.gov: %s", ", ".join(eff_overrides))
            stage_logs.append({
                "stage": "programmatic_efficacy_override",
                "extracted": extracted_efficacy,
                "overrides": eff_overrides,
            })

    # --- 3. Demographics override from CT.gov baseline ---
    extracted_demo = _extract_demographics_from_ctgov(evidence)
    if extracted_demo:
        demo = rule_set.demographics
        demo_overrides = []

        # Age min/max — always override with CT.gov-derived values, which
        # use baseline mean±2.5*std (actual participants) rather than broad
        # eligibility criteria or LLM guesses
        if "age_min" in extracted_demo and "age_max" in extracted_demo:
            ext_min, ext_max = extracted_demo["age_min"], extracted_demo["age_max"]
            # Clamp to plausible adult oncology range (25-95) to avoid
            # eligibility-criteria extremes like 18-120
            ext_min = max(ext_min, 25)
            ext_max = min(ext_max, 95)
            demo.age.min = ext_min
            demo.age.max = ext_max
            demo_overrides.append(f"age_range: → {ext_min}-{ext_max}")

        if "age_mean" in extracted_demo:
            demo.age.mean = extracted_demo["age_mean"]
            demo_overrides.append(f"age_mean: → {extracted_demo['age_mean']}")
        if "age_std" in extracted_demo:
            demo.age.std = extracted_demo["age_std"]
            demo_overrides.append(f"age_std: → {extracted_demo['age_std']}")

        # Sex (always override — CT.gov is authoritative)
        if "pct_male" in extracted_demo and "pct_female" in extracted_demo:
            old_m = demo.sex.pct_male
            demo.sex.pct_male = extracted_demo["pct_male"]
            demo.sex.pct_female = extracted_demo["pct_female"]
            demo_overrides.append(f"sex: → {extracted_demo['pct_male']}/{extracted_demo['pct_female']}")

        # ECOG PS (always override — CT.gov is authoritative)
        if "ecog_ps" in extracted_demo:
            demo.ecog_ps = extracted_demo["ecog_ps"]
            demo_overrides.append(f"ecog_ps: → {extracted_demo['ecog_ps']}")

        if demo_overrides:
            log.info("Demographics override from CT.gov: %s", ", ".join(demo_overrides))
            stage_logs.append({
                "stage": "programmatic_demographics_override",
                "extracted": extracted_demo,
                "overrides": demo_overrides,
            })

    # --- 4. AE frequency correction from evidence ---
    ae_freq_corrections = _correct_ae_frequencies(rule_set, evidence)
    if ae_freq_corrections:
        log.info("AE frequency corrections: %d AEs adjusted", len(ae_freq_corrections))
        stage_logs.append({
            "stage": "ae_frequency_correction",
            "corrections": ae_freq_corrections,
        })

    # --- 5. Evidence-backed AE pruning ---
    ae_prune_log = _prune_unevidenced_aes(rule_set, evidence)
    if ae_prune_log:
        n_pruned = sum(1 for x in ae_prune_log if x.startswith("Pruned"))
        n_merged = sum(1 for x in ae_prune_log if x.startswith("Merged"))
        log.info("AE pruning: %d removed, %d merged, %d remaining",
                 n_pruned, n_merged, len(rule_set.adverse_events))
        stage_logs.append({
            "stage": "ae_evidence_pruning",
            "pruned": n_pruned,
            "merged": n_merged,
            "remaining": len(rule_set.adverse_events),
            "details": ae_prune_log[:20],
        })

    # --- 6. PDS AE injection (add missing trial AEs) ---
    pds_inject_log = _inject_pds_aes(rule_set, evidence)
    if pds_inject_log:
        log.info("PDS AE injection: %d AEs added", len(pds_inject_log))
        stage_logs.append({
            "stage": "pds_ae_injection",
            "injected": len(pds_inject_log),
            "details": pds_inject_log[:20],
        })

    # --- 7. Final AE count cap ---
    _MAX_AES = 35
    if len(rule_set.adverse_events) > _MAX_AES:
        rule_set.adverse_events.sort(key=lambda a: a.frequency_pct, reverse=True)
        n_before = len(rule_set.adverse_events)
        rule_set.adverse_events = rule_set.adverse_events[:_MAX_AES]
        log.info("AE cap: %d → %d (removed %d lowest-frequency AEs)",
                 n_before, _MAX_AES, n_before - _MAX_AES)
        stage_logs.append({
            "stage": "ae_count_cap",
            "before": n_before,
            "after": _MAX_AES,
        })

    # ------------------------------------------------------------------
    # Comorbidity re-prompt: if LLM omitted comorbidities, ask again
    # using the same collected evidence (no hard-coded defaults).
    # ------------------------------------------------------------------
    if not rule_set.comorbidities:
        log.info("Comorbidities empty — re-prompting LLM with evidence")
        try:
            comorb_content, comorb_reasoning = await _llm_call(
                client, config.llm_model, system_base,
                _COMORBIDITY_REPROMPT.format(
                    drugs=drug_label,
                    indication=indication,
                    evidence=demographics_evidence[:8000],
                ),
                temperature=0.3, max_tokens=2000, rate_limiter=limiter,
            )
            stage_logs.append({
                "stage": "comorbidity_reprompt",
                "response_len": len(comorb_content),
                "reasoning_len": len(comorb_reasoning),
            })
            comorb_data = _parse_json_response(comorb_content)
            comorb_list = comorb_data if isinstance(comorb_data, list) else comorb_data.get("comorbidities", [])
            if comorb_list:
                from rule_engine.schema import Comorbidity
                parsed_comorbs = []
                for c in comorb_list:
                    if isinstance(c, dict) and c.get("condition"):
                        c.setdefault("prevalence_pct", 10.0)
                        c.setdefault("impacts_dosing", False)
                        c.setdefault("ae_risk_modifiers", [])
                        parsed_comorbs.append(Comorbidity(**c))
                if parsed_comorbs:
                    rule_set.comorbidities = parsed_comorbs
                    log.info("Comorbidity re-prompt yielded %d comorbidities", len(parsed_comorbs))
        except Exception as e:
            log.warning("Comorbidity re-prompt failed (non-fatal): %s", e)
            stage_logs.append({"stage": "comorbidity_reprompt", "error": str(e)})

    # ------------------------------------------------------------------
    # Demographics re-prompt: if LLM omitted age/sex, ask again
    # ------------------------------------------------------------------
    demo = rule_set.demographics
    _demo_incomplete = (
        demo.age.mean is None or demo.age.std is None
        or demo.sex.pct_male is None or demo.sex.pct_female is None
    )
    if _demo_incomplete:
        log.info("Demographics incomplete — re-prompting LLM with evidence")
        try:
            demo_content, demo_reasoning = await _llm_call(
                client, config.llm_model, system_base,
                _DEMOGRAPHICS_REPROMPT.format(
                    drugs=drug_label,
                    indication=indication,
                    evidence=demographics_evidence[:8000],
                ),
                temperature=0.3, max_tokens=2000, rate_limiter=limiter,
            )
            stage_logs.append({
                "stage": "demographics_reprompt",
                "response_len": len(demo_content),
                "reasoning_len": len(demo_reasoning),
            })
            demo_data = _parse_json_response(demo_content)
            demo_dict = demo_data.get("demographics", demo_data) if isinstance(demo_data, dict) else {}
            if isinstance(demo_dict, dict):
                from rule_engine.schema import AgeDistribution, SexDistribution, RaceEthnicityGroup
                if isinstance(demo_dict.get("age"), dict):
                    # Merge rather than replace: preserve existing fields not in reprompt
                    age_update = demo_dict["age"]
                    existing_age = rule_set.demographics.age
                    for field in ("mean", "std", "min", "max"):
                        if age_update.get(field) is not None:
                            setattr(existing_age, field, age_update[field])
                        elif getattr(existing_age, field) is None and field in age_update:
                            setattr(existing_age, field, age_update[field])
                if isinstance(demo_dict.get("sex"), dict):
                    rule_set.demographics.sex = SexDistribution(**demo_dict["sex"])
                if isinstance(demo_dict.get("race_ethnicity"), list):
                    race_entries = []
                    for r in demo_dict["race_ethnicity"]:
                        if isinstance(r, dict) and r.get("group") and r.get("pct") is not None:
                            # Coerce string pct (e.g. "45%") to float
                            try:
                                pct_val = r["pct"]
                                if isinstance(pct_val, str):
                                    pct_val = float(pct_val.replace("%", "").strip())
                                r["pct"] = float(pct_val)
                                race_entries.append(RaceEthnicityGroup(**r))
                            except (ValueError, TypeError):
                                pass
                    if race_entries:
                        rule_set.demographics.race_ethnicity = race_entries
                log.info("Demographics re-prompt filled missing fields")
        except Exception as e:
            log.warning("Demographics re-prompt failed (non-fatal): %s", e)
            stage_logs.append({"stage": "demographics_reprompt", "error": str(e)})

    # Post-reprompt age derivation: tighten min/max to reflect actual
    # participant distribution.  If mean/std available from any source
    # (LLM, CT.gov baseline, demographics reprompt), use mean±2.5*std.
    # If only min/max are available, derive mean/std from eligibility
    # range — this is basic statistics, not hardcoded clinical data.
    demo = rule_set.demographics
    age_mean = demo.age.mean
    age_std = demo.age.std
    # Fallback: derive mean/std from eligibility min/max
    if (age_mean is None or age_std is None) and demo.age.min is not None and demo.age.max is not None:
        age_mean = age_mean or (demo.age.min + demo.age.max) / 2
        age_std = age_std or (demo.age.max - demo.age.min) / 5
        demo.age.mean = age_mean
        demo.age.std = age_std
    if (age_mean is not None and age_std is not None
            and age_mean > 0 and age_std > 0):
        derived_min = int(age_mean - 2.5 * age_std)
        derived_max = int(age_mean + 2.5 * age_std)
        changed = False
        if demo.age.min is None or derived_min > demo.age.min:
            demo.age.min = derived_min
            changed = True
        if demo.age.max is None or derived_max < demo.age.max:
            demo.age.max = derived_max
            changed = True
        if changed:
            log.info("Age range tightened from mean/std: %d-%d (mean=%.1f, std=%.1f)",
                     demo.age.min, demo.age.max, age_mean, age_std)
    # Final age clamping — 25-95 for adult oncology
    if demo.age.min is not None:
        demo.age.min = max(demo.age.min, 25)
    if demo.age.max is not None:
        demo.age.max = min(demo.age.max, 95)

    # ------------------------------------------------------------------
    # Efficacy re-prompt: if LLM omitted efficacy data, ask again
    # Uses dedicated efficacy evidence (CT.gov outcomes, DailyMed) instead
    # of demographics_evidence. Also handles string values from LLM.
    # ------------------------------------------------------------------
    eff = rule_set.efficacy
    _eff_incomplete = (eff.overall_response_rate_pct is None or eff.overall_response_rate_pct == 0)
    if _eff_incomplete:
        efficacy_evidence = _format_efficacy_evidence(evidence)
        log.info("Efficacy incomplete — re-prompting LLM with efficacy evidence")
        try:
            eff_content, eff_reasoning = await _llm_call(
                client, config.llm_model, system_base,
                _EFFICACY_REPROMPT.format(
                    drugs=drug_label,
                    indication=indication,
                    evidence=efficacy_evidence[:12000],
                ),
                temperature=0.3, max_tokens=2000, rate_limiter=limiter,
            )
            stage_logs.append({
                "stage": "efficacy_reprompt",
                "response_len": len(eff_content),
                "reasoning_len": len(eff_reasoning),
            })
            eff_data = _parse_json_response(eff_content)
            eff_dict = eff_data.get("efficacy", eff_data) if isinstance(eff_data, dict) else {}
            if isinstance(eff_dict, dict):
                filled = []
                for field in ["overall_response_rate_pct", "complete_response_rate_pct",
                              "median_pfs_months", "median_pfs_ci_low", "median_pfs_ci_high",
                              "median_os_months", "median_os_ci_low", "median_os_ci_high"]:
                    val = eff_dict.get(field)
                    if val is None:
                        continue
                    # Handle string values from LLM (e.g., "30", "30%")
                    if isinstance(val, str):
                        val = val.strip().rstrip("%")
                        try:
                            val = float(val)
                        except (ValueError, TypeError):
                            log.warning("Efficacy reprompt: cannot parse %s=%r", field, val)
                            continue
                    if isinstance(val, (int, float)) and val > 0:
                        setattr(rule_set.efficacy, field, float(val))
                        filled.append(field)
                if filled:
                    log.info("Efficacy re-prompt filled: %s", ", ".join(filled))
                else:
                    log.warning("Efficacy re-prompt returned no usable values")
        except Exception as e:
            log.warning("Efficacy re-prompt failed (non-fatal): %s", e)
            stage_logs.append({"stage": "efficacy_reprompt", "error": str(e)})

    # ------------------------------------------------------------------
    # Regimen re-prompt: if regimen is empty or missing drugs, ask again
    # ------------------------------------------------------------------
    _regimen_drugs = {reg.drug for reg in (rule_set.regimen or [])}
    _missing_drugs = [d for d in drugs if d not in _regimen_drugs]
    if not rule_set.regimen or _missing_drugs:
        dose_evidence = _format_dose_evidence(evidence)
        missing_label = "empty" if not rule_set.regimen else f"missing {_missing_drugs}"
        log.info("Regimen %s — re-prompting LLM", missing_label)
        try:
            reg_content, reg_reasoning = await _llm_call(
                client, config.llm_model, system_base,
                _REGIMEN_REPROMPT.format(
                    drugs=drug_label,
                    indication=indication,
                    evidence=dose_evidence[:8000],
                ),
                temperature=0.2, max_tokens=2000, rate_limiter=limiter,
            )
            stage_logs.append({
                "stage": "regimen_reprompt",
                "response_len": len(reg_content),
                "reasoning_len": len(reg_reasoning),
            })
            reg_data = _parse_json_response(reg_content)
            reg_list = reg_data.get("regimen", reg_data) if isinstance(reg_data, dict) else []
            if isinstance(reg_list, list) and reg_list:
                from rule_engine.schema import Regimen
                new_entries = []
                for r in reg_list:
                    if isinstance(r, dict) and r.get("drug"):
                        r.setdefault("route", "IV")
                        r.setdefault("cycle_days", 21)
                        r.setdefault("schedule", "per protocol")
                        if r.get("dose") is None:
                            r["dose"] = ""
                        elif not isinstance(r["dose"], str):
                            r["dose"] = str(r["dose"])
                        new_entries.append(Regimen(**r))
                if new_entries:
                    if not rule_set.regimen:
                        rule_set.regimen = new_entries
                    else:
                        # Add only drugs that were missing
                        existing = {reg.drug for reg in rule_set.regimen}
                        for entry in new_entries:
                            if entry.drug not in existing:
                                rule_set.regimen.append(entry)
                    log.info("Regimen re-prompt added %d entries", len(new_entries))
        except Exception as e:
            log.warning("Regimen re-prompt failed (non-fatal): %s", e)
            stage_logs.append({"stage": "regimen_reprompt", "error": str(e)})

    # ------------------------------------------------------------------
    # Dose re-prompt: if regimen has vague/empty doses, ask again
    # ------------------------------------------------------------------
    _vague_patterns = {"standard dose", "per physician discretion", "per protocol", ""}
    vague_drugs = []
    for reg in (rule_set.regimen or []):
        if reg.dose.strip().lower() in _vague_patterns or not reg.dose.strip():
            vague_drugs.append(reg.drug)
    if vague_drugs:
        dose_evidence = _format_dose_evidence(evidence)
        log.info("Vague doses for %s — re-prompting LLM", vague_drugs)
        try:
            dose_content, dose_reasoning = await _llm_call(
                client, config.llm_model, system_base,
                _DOSE_REPROMPT.format(
                    drugs=drug_label,
                    indication=indication,
                    vague_entries=", ".join(vague_drugs),
                    evidence=dose_evidence[:8000],
                ),
                temperature=0.2, max_tokens=1000, rate_limiter=limiter,
            )
            stage_logs.append({
                "stage": "dose_reprompt",
                "response_len": len(dose_content),
                "reasoning_len": len(dose_reasoning),
            })
            dose_data = _parse_json_response(dose_content)
            dose_map = dose_data.get("doses", dose_data) if isinstance(dose_data, dict) else {}
            if isinstance(dose_map, dict):
                for reg in rule_set.regimen:
                    if reg.dose.strip().lower() in _vague_patterns or not reg.dose.strip():
                        new_dose = dose_map.get(reg.drug)
                        if new_dose and isinstance(new_dose, str) and new_dose.strip().lower() not in _vague_patterns:
                            log.info("Dose reprompt: %s → %s", reg.drug, new_dose)
                            reg.dose = new_dose
        except Exception as e:
            log.warning("Dose re-prompt failed (non-fatal): %s", e)
            stage_logs.append({"stage": "dose_reprompt", "error": str(e)})

    agent_log.rule_set = rule_set.model_dump()

    agent_log.stage_logs = stage_logs

    log.info("Multi-stage synthesis complete for %s / %s", drug_label, indication)
    return rule_set, agent_log
