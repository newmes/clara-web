"""Prompt templates for the Rule Discovery synthesis agent.

The system prompt establishes a clinical pharmacologist persona.
The user prompt is dynamically formatted with the evidence bundle per drug.
"""

from __future__ import annotations

import logging

from rule_engine.schema import EvidenceBundle

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intercurrent illness filter — non-drug-related AEs from pandemic-era trials
# ---------------------------------------------------------------------------

# Substring patterns — safe to match anywhere in the AE term
_INTERCURRENT_SUBSTRING_PATTERNS: list[str] = [
    "covid-19", "covid", "sars-cov-2", "coronavirus",
    "accidental injury", "road traffic accident",
]

# Exact-match terms — only match when the full AE term equals one of these
# (avoids "influenza" matching "influenza-like illness", "fall" matching
# "hair fall", "death" matching "sudden cardiac death")
_INTERCURRENT_EXACT_TERMS: set[str] = {
    "influenza", "fall", "death", "pregnancy",
    "abortion spontaneous",
}


def _is_intercurrent_illness(term: str) -> bool:
    """Return True if an AE term matches a known intercurrent illness pattern."""
    term_lower = term.lower().strip()
    if term_lower in _INTERCURRENT_EXACT_TERMS:
        return True
    return any(pattern in term_lower for pattern in _INTERCURRENT_SUBSTRING_PATTERNS)


# ---------------------------------------------------------------------------
# Checkpoint inhibitor detection for irAE prompt enrichment
# ---------------------------------------------------------------------------

_CHECKPOINT_TARGETS = {"PDCD1", "CD274", "CTLA4", "CTLA-4"}
_CHECKPOINT_MOA_KEYWORDS = [
    "pd-1", "pd-l1", "ctla-4", "checkpoint", "anti-pd", "programmed death",
]


def _is_checkpoint_inhibitor(bundle: EvidenceBundle) -> bool:
    """Detect if any drug in the bundle is a checkpoint inhibitor."""
    for drug in bundle.drugs:
        sd = bundle.per_drug.get(drug)
        if sd is None:
            continue
        for target in sd.drugbank.targets:
            name = target.get("uniprot_name", "").upper()
            if name in _CHECKPOINT_TARGETS:
                return True
        if sd.drugbank.moa:
            moa_lower = sd.drugbank.moa.lower()
            if any(kw in moa_lower for kw in _CHECKPOINT_MOA_KEYWORDS):
                return True
        if sd.chembl.mechanism_of_action:
            mech_lower = sd.chembl.mechanism_of_action.lower()
            if any(kw in mech_lower for kw in _CHECKPOINT_MOA_KEYWORDS):
                return True
    return False


SYSTEM_PROMPT = """\
You are a clinical pharmacologist specializing in trial design and drug safety.

Your task: synthesize real-world evidence into a structured clinical trial simulation rule set.
The evidence comes from DailyMed drug labels, ClinicalTrials.gov, OpenFDA FAERS, ChEMBL,
DrugBank, PubChem, PubMed, MeSH, and PrimeKG. For combination therapies, evidence is
provided per-drug with shared indication-level data.

EVIDENCE HIERARCHY:
1. DailyMed drug label data is the PRIMARY source for AE frequencies and dosing.
   Use the exact percentages from the label. Label data is FDA-reviewed ground truth.
2. ClinicalTrials.gov results (demographics, AEs, outcomes) are the PRIMARY source
   for demographics and efficacy. Prefer large trials (n >= 100).
3. FAERS report counts are reference-only — they are raw spontaneous report numbers
   WITHOUT denominators. Do NOT convert FAERS counts to percentages.
4. ChEMBL, DrugBank, PubChem, PrimeKG provide mechanism and molecular context.

PERCENTAGE SCALE — THIS IS MANDATORY:
- ALL percentage fields MUST use the 0-100 scale. For example: 55.0 means 55%, NOT 0.55.
- pct_male: 55.0 means 55% male. NEVER output 0.55.
- pct_female: 45.0 means 45% female. NEVER output 0.45.
- Race/ethnicity pct: 75.0 means 75%. NEVER output 0.75.
- frequency_pct: 26.0 means 26%. NEVER output 0.26.
- prevalence_pct: 30.0 means 30%. NEVER output 0.30.
- overall_response_rate_pct: 40.0 means 40%. NEVER output 0.40.

DEMOGRAPHICS REQUIREMENTS:
- Race/ethnicity groups must sum to approximately 100%.
- You MUST include at minimum these 5 groups: White, Black or African American, Asian,
  Hispanic or Latino, Other. Use your knowledge of typical US clinical trial demographics
  if no large-trial data is available.
- Mean age should reflect the typical age of the indication population.
- When ClinicalTrials.gov results provide baseline demographics from large trials (n >= 30),
  use those distributions VERBATIM. IGNORE demographics from small trials (n < 30).
- Sex ratio: USE the trial baseline value if available. Different indications have very different
  sex ratios (e.g., lung cancer ~70-80% male, breast cancer ~99% female).
  Do NOT default to 60/40 when trial data exists.
- ECOG PS: When baseline ECOG data is provided, use those exact proportions.
  Do NOT use generic distributions (40/50/10) when trial-specific data exists.
- Age min/max: When eligibility criteria specify age bounds (e.g., "18 Years-75 Years"),
  use those bounds. Do NOT default to 18-90.

COMORBIDITY REQUIREMENTS:
- Include 5-8 comorbidities relevant to the indication population.
- Comorbidities are pre-existing medical conditions: hypertension, diabetes, obesity,
  cardiovascular disease, prior autoimmune conditions, renal impairment, hepatic impairment,
  COPD, hyperlipidemia, depression.
- BIOMARKERS and molecular subtypes (BRAF V600E mutation, PD-L1 expression, MSI-H status,
  HER2 status, EGFR mutation, ALK rearrangement) are NOT comorbidities.
- ae_risk_modifiers: for each comorbidity, specify which AEs have increased risk and
  by what multiplier (1.0 = no change). Only include modifiers where the comorbidity
  genuinely increases the AE risk (e.g., hepatic impairment → hepatotoxicity risk × 1.5).

ADVERSE EVENT REQUIREMENTS:
- Use DailyMed label AE frequencies as primary source. Do NOT inflate them.
- Include at least 10-15 AEs covering the drug's safety profile.
- severity_distribution: for each AE, provide grade breakdown (grade_1, grade_2, grade_3,
  grade_4, grade_5). The sum of all grade percentages must equal frequency_pct.
  Example: frequency_pct=30.0 → {"grade_1": 15.0, "grade_2": 10.0, "grade_3": 4.0, "grade_4": 1.0}
- triggers: specify downstream consequences when an AE occurs at certain severity.
  Each AE type has DIFFERENT management thresholds. Use these guidelines:

  HEMATOLOGIC (neutropenia, anemia, thrombocytopenia):
  - grade >= 3 → Dose reduction (50-70%)
  - grade >= 4 → Treatment discontinuation (15-30%)

  DERMATOLOGIC (rash, pruritus, skin reaction):
  - grade >= 2 → Dose reduction (20-40%)
  - grade >= 3 → Treatment discontinuation (30-60%)

  HEPATIC (ALT increased, AST increased, bilirubin):
  - grade >= 3 → Dose reduction (50-80%)
  - grade >= 4 → Treatment discontinuation (40-70%)

  GASTROINTESTINAL (diarrhea, nausea, vomiting):
  - grade >= 3 → Dose reduction (30-50%)
  - grade >= 4 → Treatment discontinuation (10-20%)

  ENDOCRINE/irAE (hypothyroidism, hyperthyroidism, adrenal insufficiency):
  - grade >= 3 → Treatment discontinuation (10-20%)
  (endocrinopathies are managed with hormone replacement, NOT dose reduction)

  NEUROLOGIC (peripheral neuropathy):
  - grade >= 2 → Dose reduction (30-50%)
  - grade >= 3 → Treatment discontinuation (40-70%)

  LOW-GRADE COMMON AEs (fatigue, decreased appetite, weight loss):
  - Typically NO triggers (grade >= 3 is rare)
  - Only add triggers if grade 3-4 rate > 3%

  VARY the probability_pct based on clinical severity — do NOT use the same values for all AEs.
  target_ae can be another AE name, 'Dose reduction', or 'Treatment discontinuation'.
- EXCLUDE intercurrent illnesses: COVID-19, influenza, accidental injuries, pregnancy.
- Each AE should appear only ONCE in the list — no duplicates.
- For combination therapy: set source_drug on every AE to indicate which drug is primarily
  responsible. source_drug must be one of the drugs in the drugs list.

FREQUENCY_PCT SOURCE — THIS IS MANDATORY:
- frequency_pct must reflect ALL-GRADES incidence (any grade), NOT grade 3-4 only.
  DailyMed tables may show both "All Grades" and "Grade 3-4" columns.
  Always use the "All Grades" (or "any grade") column for frequency_pct.
  The grade 3-4 percentage is informational for severity_distribution weighting.
- If a DailyMed AE is tagged [lab], it came from a laboratory abnormalities table.
  Lab AEs (e.g., ALT increased, neutrophil count decreased) are valid AEs — include them
  but recognize their incidence reflects lab monitoring, not clinical symptoms.

REGIMEN REQUIREMENTS:
- Provide one regimen entry per drug in the drugs list.
- Include dose, route, cycle_days, and schedule from label data.

EFFICACY:
- Complete response rate must be <= overall response rate.
- AE onset days must be <= treatment duration.
"""


# fmt: off
def format_evidence_prompt(bundle: EvidenceBundle) -> str:
    """Format the evidence bundle into the user prompt for the synthesis agent."""
    ct = bundle.clinical_trials
    combo_ct = bundle.combo_trials
    kg = bundle.primekg
    lit = bundle.literature
    msh = bundle.mesh

    drug_label = " + ".join(bundle.drugs)
    sections = [
        f"Generate clinical trial simulation rules for:\n"
        f"Drug(s): {drug_label}\n"
        f"Indication: {bundle.indication}\n",
        "=== EVIDENCE FROM DATABASES ===\n",
    ]

    # ---------------------------------------------------------------
    # Per-drug sections
    # ---------------------------------------------------------------
    for drug in bundle.drugs:
        sd = bundle.per_drug.get(drug)
        if sd is None:
            sections.append(f"[No evidence collected for {drug}]\n")
            continue

        dm = sd.dailymed
        fda = sd.openfda
        ch = sd.chembl
        db = sd.drugbank
        pc = sd.pubchem

        # DailyMed — presented FIRST as ground-truth AE frequencies
        sections.append(f"[DailyMed — {drug} — FDA-approved AE frequencies (USE THESE)]")
        if dm.found:
            sections.append(f"- Drug label: {dm.drug_name_label}")
            if dm.ae_table:
                sections.append("- ADVERSE REACTION INCIDENCE TABLE (from FDA label — use these exact values):")
                for ae in dm.ae_table[:30]:
                    term = ae.get("term", "?")
                    if _is_intercurrent_illness(term):
                        continue
                    table_type = ae.get("table_type", "")
                    type_tag = f" [{table_type}]" if table_type and table_type != "clinical" else ""
                    line = f"  {term}: {ae.get('incidence_pct', '?')}%{type_tag}"
                    if ae.get("grade34_pct") is not None:
                        line += f" (grade 3-4: {ae['grade34_pct']}%)"
                    sections.append(line)
            if dm.boxed_warning:
                sections.append(f"- BOXED WARNING: {dm.boxed_warning[:500]}")
            if dm.contraindications:
                sections.append(f"- Contraindications: {dm.contraindications[:500]}")
            if dm.dosage_text:
                sections.append(f"- Dosage: {dm.dosage_text[:500]}")
            if dm.special_populations:
                sections.append(f"- Special populations: {dm.special_populations[:500]}")
            if dm.adverse_reactions_text and not dm.ae_table:
                sections.append(f"- Adverse reactions text: {dm.adverse_reactions_text[:1000]}")
        else:
            sections.append("- No DailyMed label found")
        sections.append("")

        # DrugBank
        sections.append(f"[DrugBank — {drug}]")
        if db.found:
            if db.moa:
                sections.append(f"- MoA: {db.moa}")
            if db.targets:
                target_names = [t.get("uniprot_name", t.get("name", "?")) for t in db.targets[:10]]
                sections.append(f"- Targets: {', '.join(target_names)}")
            sections.append(f"- DDI count: {db.ddi_count}")
        else:
            sections.append("- Not found in DrugBank")
        sections.append("")

        # OpenFDA FAERS
        sections.append(f"[OpenFDA FAERS — {drug} — spontaneous reports (reference only, NOT incidence rates)]")
        if fda.top_adverse_events:
            ae_lines = []
            for ae in fda.top_adverse_events[:15]:
                ae_lines.append(f"  {ae.get('term', 'Unknown')}: {ae.get('count', 0)} reports")
            sections.append(f"- Total AE reports: {fda.total_ae_reports}")
            sections.append("- Top adverse events (raw report counts, not percentages):")
            sections.extend(ae_lines)
            # Time-to-onset data from FAERS
            if fda.has_timing_data and fda.time_to_onset_data:
                total_onset = sum(e.get("count", 0) for e in fda.time_to_onset_data)
                sections.append(f"- Time-to-onset data available ({total_onset} reports with timing):")
                for entry in fda.time_to_onset_data:
                    sections.append(
                        f"  Interval unit '{entry.get('unit_label', '?')}': "
                        f"{entry.get('count', 0)} reports"
                    )
        else:
            sections.append("- No FAERS data found")
        sections.append("")

        # ChEMBL
        sections.append(f"[ChEMBL — {drug}]")
        if ch.has_data:
            if ch.mechanism_of_action:
                sections.append(f"- Mechanism: {ch.mechanism_of_action}")
            sections.append(f"- Max phase: {ch.max_phase}")
            sections.append(f"- Activity records: {ch.activity_count}, Target count: {ch.target_count}")
            if ch.molecule_type:
                sections.append(f"- Molecule type: {ch.molecule_type}")
        else:
            sections.append("- No ChEMBL data found")
        sections.append("")

        # PubChem
        sections.append(f"[PubChem — {drug}]")
        if pc.found:
            sections.append(f"- CID: {pc.pubchem_cid}")
            if pc.molecular_weight is not None:
                sections.append(f"- Molecular weight: {pc.molecular_weight:.1f}")
            if pc.logp is not None:
                sections.append(f"- LogP: {pc.logp:.2f}")
            if pc.tpsa is not None:
                sections.append(f"- TPSA: {pc.tpsa:.1f}")
            sections.append(f"- H-bond donors: {pc.hbd_count}, acceptors: {pc.hba_count}")
            sections.append(f"- Rotatable bonds: {pc.rotatable_bonds}")
            sections.append(f"- Lipinski violations: {pc.lipinski_violations}")
            if pc.bioassay_total_count > 0:
                sections.append(f"- Bioassays: {pc.bioassay_active_count} active / {pc.bioassay_total_count} total")
            if pc.pharmacological_class:
                sections.append(f"- Pharmacological class: {pc.pharmacological_class}")
        else:
            sections.append("- Not found in PubChem")
        sections.append("")

        # OnSIDES validated ADE pairs
        onsides = sd.onsides
        sections.append(f"[OnSIDES — {drug} — validated drug-ADE pairs from FDA labels (PubMedBERT F1=0.90)]")
        if onsides.found:
            if onsides.boxed_warning_aes:
                sections.append(f"- BOXED WARNING AEs: {', '.join(onsides.boxed_warning_aes)}")
            sections.append(f"- Total validated ADE pairs: {onsides.total_pairs}")
            sections.append("- Top validated adverse events (by label count):")
            for ae in onsides.ae_pairs[:20]:
                bw_tag = " [BOXED WARNING]" if ae.get("is_boxed_warning") else ""
                pred = ae.get("mean_pred_score")
                pred_str = f", pred={pred:.2f}" if pred is not None else ""
                sections.append(
                    f"  {ae.get('pt_meddra_term', '?')}: labels={ae.get('label_count', '?')}"
                    f"{pred_str}{bw_tag}"
                )
        else:
            sections.append("- No OnSIDES data found")
        sections.append("")

    # ---------------------------------------------------------------
    # Shared sections
    # ---------------------------------------------------------------

    # ClinicalTrials.gov
    sections.append("[ClinicalTrials.gov]")
    if ct.trial_count > 0:
        sections.append(f"- {ct.trial_count} trials found, max phase: {ct.max_phase}")
        if ct.age_range:
            sections.append(f"- Age range: {ct.age_range}")
        if ct.sex_eligibility:
            sections.append(f"- Sex eligibility: {ct.sex_eligibility}")
        if ct.primary_endpoints:
            sections.append(f"- Primary endpoints: {', '.join(ct.primary_endpoints)}")
        if ct.sample_sizes:
            sections.append(f"- Sample sizes: {ct.sample_sizes}")
    else:
        sections.append("- No trials found")
    sections.append("")

    # ClinicalTrials.gov results (if available)
    if ct.has_results:
        sample_n = ct.baseline_demographics.get("_sample_size", 0)
        if ct.baseline_demographics and sample_n >= 30:
            source_nct = ct.baseline_demographics.get("_source_nct_id", "unknown")
            source_phase = ct.baseline_demographics.get("_source_phase", "")
            source_title = ct.baseline_demographics.get("_source_title", "")
            phase_tag = f", {source_phase}" if source_phase else ""
            sections.append(
                f"[ClinicalTrials.gov Results — demographics from {source_nct} "
                f"(n={sample_n}{phase_tag})]"
            )
            if source_title:
                sections.append(f"- Source trial: {source_title}")
            sections.append("- Baseline demographics:")
            for key, val in ct.baseline_demographics.items():
                if key.startswith("_"):
                    continue
                sections.append(f"  {key}: {val}")
        elif ct.baseline_demographics:
            sections.append(f"[ClinicalTrials.gov Results — WARNING: small sample (n={sample_n}), "
                          "do NOT use for demographics — use domain knowledge instead]")
        if ct.reported_aes:
            sections.append("- Reported adverse events from trial results:")
            for ae in ct.reported_aes[:20]:
                term = ae.get("term", "?")
                if _is_intercurrent_illness(term):
                    continue
                serious_tag = " [SERIOUS]" if ae.get("serious") else ""
                grade34 = ae.get("grade34_pct")
                grade_info = f" (grade 3-4: {grade34}%)" if grade34 is not None else ""
                sections.append(
                    f"  {term}: {ae.get('pct', '?')}% "
                    f"({ae.get('affected', '?')}/{ae.get('at_risk', '?')}){serious_tag}{grade_info}"
                )
        if ct.primary_outcomes:
            sections.append("- Primary outcome results:")
            for out in ct.primary_outcomes[:5]:
                sections.append(
                    f"  {out.get('measure', '?')}: {out.get('value', '?')} {out.get('unit', '')}"
                )
        sections.append("")

    # Combo trials (if applicable)
    if combo_ct.trial_count > 0:
        sections.append("[ClinicalTrials.gov — Combo Trial]")
        sections.append(f"- {combo_ct.trial_count} combo trials found, max phase: {combo_ct.max_phase}")
        if combo_ct.primary_endpoints:
            sections.append(f"- Primary endpoints: {', '.join(combo_ct.primary_endpoints)}")
        if combo_ct.sample_sizes:
            sections.append(f"- Sample sizes: {combo_ct.sample_sizes}")
        if combo_ct.reported_aes:
            sections.append("- Combo trial AEs:")
            for ae in combo_ct.reported_aes[:15]:
                term = ae.get("term", "?")
                if _is_intercurrent_illness(term):
                    continue
                sections.append(
                    f"  {term}: {ae.get('pct', '?')}% "
                    f"({ae.get('affected', '?')}/{ae.get('at_risk', '?')})"
                )
        if combo_ct.primary_outcomes:
            sections.append("- Combo trial outcomes:")
            for out in combo_ct.primary_outcomes[:5]:
                sections.append(
                    f"  {out.get('measure', '?')}: {out.get('value', '?')} {out.get('unit', '')}"
                )
        sections.append("")

    # PubMed
    sections.append("[PubMed]")
    sections.append(f"- Co-occurrence score: {lit.cooccurrence_score:.3f}")
    sections.append(f"- Article count: {lit.article_count}")
    sections.append("")

    # MeSH
    sections.append("[MeSH Disease Hierarchy]")
    if msh.found:
        sections.append(f"- MeSH ID: {msh.disease_mesh_id} ({msh.disease_mesh_name})")
        if msh.tree_numbers:
            sections.append(f"- Tree numbers: {', '.join(msh.tree_numbers[:5])}")
        if msh.parent_terms:
            sections.append(f"- Broader categories: {', '.join(msh.parent_terms[:5])}")
        if msh.child_terms:
            sections.append(f"- Subtypes: {', '.join(msh.child_terms[:10])}")
        if msh.related_terms:
            sections.append(f"- Related diseases: {', '.join(msh.related_terms[:5])}")
        if msh.qualifiers:
            sections.append(f"- Applicable subheadings: {', '.join(msh.qualifiers[:10])}")
    else:
        sections.append("- Disease not found in MeSH")
    sections.append("")

    # PrimeKG
    sections.append("[PrimeKG]")
    if kg.found:
        if kg.neighbor_summary:
            sections.append(f"- KG neighbors: {kg.neighbor_summary}")
        if kg.disease_associations:
            disease_names = [d.get("name", "?") for d in kg.disease_associations[:10]]
            sections.append(f"- Disease associations: {', '.join(disease_names)}")
        if kg.gene_targets:
            gene_names = [g.get("name", "?") for g in kg.gene_targets[:10]]
            sections.append(f"- Gene targets: {', '.join(gene_names)}")
    else:
        sections.append("- Not found in PrimeKG")
    sections.append("")

    # Project Data Sphere (patient-level data)
    pds = bundle.pds
    if pds.found and pds.matched_trial:
        n_label = pds.safety_population_n or pds.matched_trial.n_patients
        sections.append(f"[Project Data Sphere — patient-level data (n={n_label}) — HIGH QUALITY]")
        sections.append(f"- Matched trial: {pds.matched_trial.trial_id} (score={pds.matched_trial.match_score})")
        if pds.demographics:
            demo = pds.demographics
            if demo.age_mean is not None:
                sections.append(f"- Age: mean={demo.age_mean}, std={demo.age_std}, min={demo.age_min}, max={demo.age_max}")
            if demo.pct_male is not None:
                sections.append(f"- Sex: {demo.pct_male}% male, {demo.pct_female}% female")
            if demo.ecog_distribution:
                ecog_str = ", ".join(f"ECOG {k}={v}%" for k, v in sorted(demo.ecog_distribution.items()))
                sections.append(f"- ECOG PS: {ecog_str}")
        if pds.efficacy:
            eff = pds.efficacy
            if eff.overall_response_rate_pct is not None:
                sections.append(f"- ORR: {eff.overall_response_rate_pct}%")
            if eff.median_pfs_months is not None:
                sections.append(f"- Median PFS: {eff.median_pfs_months} months")
            if eff.median_os_months is not None:
                sections.append(f"- Median OS: {eff.median_os_months} months")
        if pds.ae_aggregates:
            sections.append(f"- Top AEs ({len(pds.ae_aggregates)} total):")
            for ae in pds.ae_aggregates[:15]:
                line = f"  {ae.term}: {ae.frequency_pct}% ({ae.n_patients_with_event}/{ae.n_total_patients})"
                sections.append(line)
        if pds.regimen:
            sections.append("- Regimen (actual administered doses):")
            for reg in pds.regimen:
                dose_str = f"{reg.median_dose} {reg.dose_unit}" if reg.median_dose else "?"
                route_str = reg.route or "?"
                sections.append(f"  {reg.drug}: {dose_str} ({route_str}) [n={reg.n_patients}]")
        sections.append("")

    # ---------------------------------------------------------------
    # Checkpoint inhibitor irAE enrichment (conditional)
    # ---------------------------------------------------------------
    if _is_checkpoint_inhibitor(bundle):
        sections.append("=== CHECKPOINT INHIBITOR — IMMUNE-RELATED AEs (irAEs) ===")
        sections.append(
            "This drug is a checkpoint inhibitor (PD-1/PD-L1/CTLA-4 pathway).\n"
            "You MUST include the following immune-related adverse events (irAEs), even if low frequency.\n"
            "These are clinically critical and define the drug's safety profile:\n\n"
            "REQUIRED irAEs (include ALL of these):\n"
            "- Pneumonitis (2-5%, but can be fatal — grade 3-4: <2%)\n"
            "- Colitis/Diarrhea (immune-mediated, 1-20% depending on agent)\n"
            "- Hepatitis / Increased ALT / Increased AST (immune-mediated, 5-15%)\n"
            "- Hypothyroidism (5-15%)\n"
            "- Hyperthyroidism (1-5%)\n"
            "- Nephritis / Increased creatinine (1-3%)\n"
            "- Skin reactions / Rash (10-40%)\n\n"
            "irAE characteristics:\n"
            "- Delayed onset: most irAEs appear weeks to months after treatment start\n"
            "- Reversible with immunosuppression (corticosteroids), except thyroid disorders\n"
            "- Triggers: grade ≥ 2 irAEs → hold treatment + start steroids;\n"
            "  grade ≥ 3 → permanent discontinuation (except endocrinopathies)\n"
            "- For combination checkpoint inhibitors (e.g., nivo+ipi), irAE rates are HIGHER"
        )
        sections.append("")

    # ---------------------------------------------------------------
    # Output requirements
    # ---------------------------------------------------------------
    n_drugs = len(bundle.drugs)
    combo_note = ""
    if n_drugs > 1:
        combo_note = (
            f"\n- This is a COMBINATION THERAPY with {n_drugs} drugs. "
            "Every AE MUST have source_drug set to one of the drugs in the drugs list.\n"
            f"- The regimen list must have exactly {n_drugs} entries (one per drug)."
        )

    sections.append("=== OUTPUT REQUIREMENTS ===")
    sections.append(
        "New fields you MUST populate correctly:\n"
        "- severity_distribution: dict mapping grade keys to percentages.\n"
        "  Keys: grade_1, grade_2, grade_3, grade_4, grade_5 (omit grades with 0%).\n"
        "  The sum of all grade values MUST equal frequency_pct (±0.5%).\n"
        "- triggers: list of {target_ae, condition, probability_pct} for AE cascades.\n"
        "  target_ae can be another AE name, 'Dose reduction', or 'Treatment discontinuation'.\n"
        "- ae_risk_modifiers on comorbidities: list of {ae, risk_multiplier}.\n"
        "  ae must reference an AE in adverse_events. risk_multiplier >= 1.0.\n"
        "- regimen: list of {drug, dose, route, cycle_days, schedule} — one entry per drug."
        f"{combo_note}"
    )
    sections.append("")

    # Instructions / mandatory checklist
    sections.append("=== INSTRUCTIONS ===")
    sections.append(
        "Synthesize the above evidence into a complete clinical trial rule set.\n"
        "MANDATORY CHECKLIST — verify before responding:\n"
        "1. ALL percentage fields use 0-100 scale (55.0 = 55%, NOT 0.55)\n"
        "2. AE frequencies: use DailyMed label percentages when available. Do NOT inflate them.\n"
        "3. severity_distribution values sum to frequency_pct for each AE.\n"
        "4. Demographics: use large-trial data or domain knowledge. Ignore n < 30 samples.\n"
        "5. Race/ethnicity: 5+ groups summing to ~100% (White, Black, Asian, Hispanic, Other).\n"
        "6. Comorbidities: 5-8 pre-existing conditions. NO biomarkers/mutations.\n"
        "7. Each AE appears only once — no duplicates.\n"
        "8. Mean age reflects typical indication population (melanoma ~62, lung ~67, breast ~57).\n"
        "9. Regimen has one entry per drug with dose, route, cycle_days, schedule from label.\n"
        "10. EXCLUDE intercurrent illnesses (COVID-19, influenza, accidental injuries, pregnancy)."
    )

    return "\n".join(sections)
# fmt: on


# ---------------------------------------------------------------------------
# Multi-stage pipeline prompt templates
# ---------------------------------------------------------------------------

STAGE1_AE_FREQ_PROMPT = """\
You are a clinical pharmacologist extracting adverse event frequencies from drug label data.

Given the DailyMed AE incidence table, OnSIDES validated drug-ADE pairs, and ClinicalTrials.gov
reported AEs below, extract a JSON list of adverse events with their frequencies.

CRITICAL — COMPLETENESS:
- Include ALL treatment-emergent adverse events (TEAEs) from ClinicalTrials.gov with frequency >= 5%.
  TEAEs include both drug-related side effects AND disease symptoms observed during the trial
  (e.g., cough, dyspnea, pain, weight loss). In oncology trials, disease-related symptoms are
  routinely reported as TEAEs — include them.
- Include ALL AEs from DailyMed AE incidence tables.
- For combination therapy, include AEs from BOTH individual drug labels AND combo trial AEs.
- Common AEs like neutropenia, fatigue, nausea MUST appear if they are in any evidence source.
- If Project Data Sphere patient-level data is provided, include high-frequency AEs from that source.

Rules:
- Use the "All Grades" / "any grade" incidence column for frequency_pct (0-100 scale).
- When both DailyMed and CT.gov have the same AE, use the HIGHER value (max of the two).
  CT.gov values come from actual patient data and are often more accurate.
- If OnSIDES lists an AE not in DailyMed or CT.gov, include it with frequency_pct: null.
- Exclude intercurrent illnesses (COVID-19, influenza, accidental injuries, pregnancy).
- Each AE appears only once. For duplicates, use the highest frequency across sources.
- Aim for at least 20-30 AEs for a complete safety profile.

Respond with a JSON array:
[{{"event": "...", "frequency_pct": <number|null>, "source": "dailymed"|"ctgov"|"onsides"|"combo"|"pds", "is_boxed_warning": <bool>}}]

EVIDENCE:
{evidence}
"""

STAGE1_SEVERITY_PROMPT = """\
You are a clinical pharmacologist determining adverse event severity grade distributions.

Given AE frequency data, DailyMed grade 3-4 percentages, and ClinicalTrials.gov grade 3-4 data below,
produce a severity grade distribution for each AE.

Rules:
- severity_distribution keys: grade_1, grade_2, grade_3, grade_4, grade_5 (omit grades with 0%).
- The sum of all grade values MUST equal the AE's frequency_pct.
- Use grade34_pct from DailyMed/CT.gov to anchor grades 3-4. Distribute the remainder across grades 1-2.
- If no grade data exists, mark the AE as "no_grade_data": true and estimate conservatively.
- grade_5 (death) should be 0 for most AEs unless evidence supports it.

CRITICAL — VARY the grade distribution by AE type. Do NOT use the same ratio for all AEs:
  - GI AEs (nausea, diarrhea): mostly grade 1-2 (grade_1 = 55-65% of total)
  - Hematologic (neutropenia, anemia): more grade 3-4 (grade_1 = 25-40% of total)
  - Dermatologic (rash, pruritus): mostly grade 1 (grade_1 = 60-75% of total)
  - Hepatic (ALT/AST increased): bimodal (grade_1 = 40-50% of total)
  - Fatigue/constitutional: overwhelmingly mild (grade_1 = 65-80% of total)
  - Endocrine (hypothyroidism): mostly grade 1-2, rarely grade 3+ (grade_1 = 50-70%)
  Each AE should have a DIFFERENT grade_1/frequency_pct ratio.

Respond with a JSON object:
{{"<event_name>": {{"grade_1": X, "grade_2": Y, "grade_3": Z, "grade_4": W}}, ...}}

EVIDENCE:
{evidence}
"""

STAGE1_ONSET_PROMPT = """\
You are a clinical pharmacologist determining adverse event onset timing from FAERS data and clinical knowledge.

Given the FAERS time-to-onset data and drug mechanism information below, estimate median onset days
for EACH AE individually. Different AE types have DIFFERENT onset timing — you MUST vary onset across AEs.

CRITICAL: Each AE must have its OWN clinically appropriate onset timing. Do NOT assign the same onset
to all AEs. If you lack specific data, use these evidence-based category defaults as a STARTING POINT
and adjust ±3-5 days for individual AEs within each category:

  GASTROINTESTINAL (nausea, vomiting, diarrhea, constipation): 2-5 days
  DERMATOLOGIC (rash, pruritus, dry skin, alopecia): 14-28 days
  HEMATOLOGIC (neutropenia, anemia, thrombocytopenia, lymphopenia): 10-18 days
  HEPATIC (ALT/AST increased, bilirubin increased): 21-42 days
  ENDOCRINE (hypothyroidism, hyperthyroidism): 42-90 days
  NEUROLOGIC (peripheral neuropathy, headache, dizziness): 14-30 days
  MUSCULOSKELETAL (arthralgia, myalgia, back pain): 14-28 days
  FATIGUE/CONSTITUTIONAL (fatigue, asthenia, decreased appetite): 7-14 days
  RESPIRATORY (cough, dyspnea, pneumonitis): 30-60 days
  CARDIAC (QT prolongation, hypertension): 14-30 days
  METABOLIC (hyperglycemia, hyponatremia): 7-21 days
  INFECTION-RELATED (upper respiratory infection, UTI): 21-42 days
  IMMUNE-RELATED/irAE (colitis, hepatitis, pneumonitis): 42-90 days

Rules:
- Use FAERS interval distributions when available (day-level reports suggest early onset, week/month later).
- For AEs without FAERS timing, use the category defaults above BUT add ±3-5 day variation per AE.
- EVERY AE must have a DIFFERENT median_onset_days value — no two AEs should share the same onset.
- Output median_onset_days must be > 0 and <= treatment duration.
- Annotate each with "source": "faers" or "estimated".

Respond with a JSON object:
{{"<event_name>": {{"median_onset_days": <int>, "source": "faers"|"estimated"}}, ...}}

EVIDENCE:
{evidence}
"""

STAGE1_TRIGGERS_PROMPT = """\
You are a clinical pharmacologist defining AE management triggers and cascade rules.

For each AE category below, define trigger rules based on clinical management guidelines.

Guidelines by AE category:
- HEMATOLOGIC (neutropenia, anemia, thrombocytopenia): grade >= 3 → Dose reduction (50-70%); grade >= 4 → Discontinuation (15-30%)
- DERMATOLOGIC (rash, pruritus): grade >= 2 → Dose reduction (20-40%); grade >= 3 → Discontinuation (30-60%)
- HEPATIC (ALT/AST increased): grade >= 3 → Dose reduction (50-80%); grade >= 4 → Discontinuation (40-70%)
- GI (diarrhea, nausea, vomiting): grade >= 3 → Dose reduction (30-50%); grade >= 4 → Discontinuation (10-20%)
- ENDOCRINE (hypothyroidism, hyperthyroidism): grade >= 3 → Discontinuation (10-20%) — managed with hormone replacement
- NEUROLOGIC (neuropathy): grade >= 2 → Dose reduction (30-50%); grade >= 3 → Discontinuation (40-70%)
- LOW-GRADE COMMON (fatigue, appetite loss): triggers only if grade 3-4 rate > 3%

VARY probability_pct based on clinical severity. target_ae can be another AE name, "Dose reduction", or "Treatment discontinuation".

Respond with a JSON object:
{{"<event_name>": [{{"target_ae": "...", "condition": "grade >= N", "probability_pct": X}}], ...}}

ADVERSE EVENTS:
{evidence}
"""

STAGE1_DEMOGRAPHICS_PROMPT = """\
You are a clinical trial statistician extracting demographics and efficacy data from trial results.

Given ClinicalTrials.gov baseline demographics, primary outcomes, and indication context below,
extract structured demographics and efficacy data.

CRITICAL — USE THE EVIDENCE DATA, DO NOT GUESS:
- When baseline demographics are provided from a trial (n >= 30), use those EXACT values for
  sex ratios, ECOG distribution, and age statistics. Do NOT substitute generic estimates.
- When eligibility age range is provided (e.g., "18 Years-75 Years"), use those bounds for
  age min/max. Do NOT default to generic 18-90 ranges.
- When primary outcomes provide ORR, PFS, OS values, use those EXACT numbers.
- When ECOG PS data appears in baseline demographics (e.g., "ecog_ps": {{"0": "45", "1": "154", "2": "1"}}),
  convert patient counts to percentages and use EXACTLY.
  Example: {{"0": "45", "1": "154", "2": "1"}} with 200 total → ecog_ps: {{"0": 22.5, "1": 77.0, "2": 0.5}}
  Do NOT output generic ECOG distributions (40/50/10) when trial data shows different values.
  ECOG is critical for simulation accuracy.

Rules:
- ALL percentages on 0-100 scale (55.0 = 55%).
- Race/ethnicity: include at least White, Black or African American, Asian, Hispanic or Latino, Other — sum ~100%.
- Use large-trial demographics (n >= 30). Ignore small samples.
- Sex ratio: if baseline shows 76% Male / 24% Female, output pct_male=76, pct_female=24.
  Do NOT use generic 60/40 splits when trial data exists.
- ECOG: if baseline shows ECOG 0=22.5%, 1=77%, 2=0.5%, include ecog_ps with those proportions.
- Efficacy: complete_response_rate <= overall_response_rate.
- PFS/OS: provide the MEDIAN and 95% CONFIDENCE INTERVAL from the pivotal trial. Each drug has different PFS/OS.
- Include regimen details (dose, route, cycle_days, schedule) from label data.
- Include 5-8 comorbidities relevant to the indication (pre-existing conditions only, NOT biomarkers).
  Use trial-population prevalences (typically 5-15% each), NOT general population prevalences.

Respond with a JSON object:
{{
  "demographics": {{"age": {{"min": .., "max": .., "mean": .., "std": ..}}, "sex": {{"pct_male": .., "pct_female": ..}}, "ecog_ps": {{"0": .., "1": .., "2": ..}}, "race_ethnicity": [{{"group": "..", "pct": ..}}]}},
  "efficacy": {{"overall_response_rate_pct": .., "complete_response_rate_pct": .., "median_pfs_months": .., "median_pfs_ci_low": .., "median_pfs_ci_high": .., "median_os_months": .., "median_os_ci_low": .., "median_os_ci_high": ..}},
  "regimen": [{{"drug": "..", "dose": "..", "route": "..", "cycle_days": .., "schedule": ".."}}],
  "comorbidities": [{{"condition": "..", "prevalence_pct": .., "impacts_dosing": <bool>}}],
  "phase": <int>,
  "treatment_duration_days": <int>
}}

EVIDENCE:
{evidence}
"""

STAGE2_GROUNDING_PROMPT = """\
You are a clinical data auditor verifying extracted values against raw evidence.

Below you will see:
1. EXTRACTED DATA — structured outputs from a previous extraction stage
2. RAW EVIDENCE — the original evidence data these values were extracted from

Your task: For each extracted value, check whether the raw evidence supports it.

Rules:
- Mark each field as "GROUNDED" (value matches evidence) or "UNGROUNDED" (no supporting data).
- For UNGROUNDED fields, explain what evidence is missing.
- Do NOT change grounded values — only flag unsupported ones.
- Frequency percentages within ±2% of evidence values are GROUNDED.
- Severity distributions that sum to frequency_pct (±0.5%) are GROUNDED.

Respond with a JSON object:
{{
  "grounding_report": [
    {{"field": "...", "value": "...", "status": "GROUNDED"|"UNGROUNDED", "evidence_ref": "...", "note": "..."}}
  ],
  "ungrounded_count": <int>,
  "total_checked": <int>
}}

EXTRACTED DATA:
{extracted}

RAW EVIDENCE:
{evidence}
"""

STAGE3_SYNTHESIS_PROMPT = """\
You are a clinical pharmacologist assembling a final clinical trial simulation rule set.

Below you will see:
1. EXTRACTED DATA — AE frequencies, severity distributions, onset timing, triggers, demographics/efficacy
2. GROUNDING REPORT — which values are verified vs ungrounded
3. SCHEMA — the exact JSON schema to produce

CRITICAL RULES:
- Use GROUNDED values as-is. For UNGROUNDED values, use conservative clinical estimates.
- severity_distribution values MUST sum to frequency_pct for each AE (±0.5%).
- ALL percentages on 0-100 scale.
- Race/ethnicity sums to ~100%.
- Each AE appears only once.
- For combination therapy, set source_drug on every AE.
- Comorbidities: include ae_risk_modifiers referencing AEs in the list.

ACCURACY REQUIREMENTS:
- CRITICAL: When extracted data includes specific numeric values (ORR, PFS, OS, doses, sex ratios),
  copy them EXACTLY into the output. Do NOT round, adjust, or estimate "more typical" values.
  The extracted data comes from real clinical trials and FDA-approved labels.
- For ORR: if extracted efficacy shows overall_response_rate_pct=87, output 87. NOT 60, NOT 45.
- For doses: if extracted regimen shows "100 mg/m^2", output "100 mg/m^2". NOT "35 mg/m^2".
- Use the EXACT sex ratio from extracted demographics (e.g., if extracted shows 76/24 male/female,
  output pct_male=76.0, pct_female=24.0). Do NOT substitute generic 60/40 splits.
- Use the EXACT age min/max from eligibility criteria if provided.
- Use the EXACT ORR, PFS, OS values from extracted efficacy data.
- If extracted data includes ECOG PS distribution, use those EXACT proportions for ecog_ps.
  (e.g., {{"0": 22.5, "1": 77.0, "2": 0.5}})
- AE FILTERING: Include ONLY adverse events from the extracted AE frequency data with specific
  numeric frequencies. Do NOT add AEs beyond what the evidence provides. Quality over quantity:
  25 well-sourced AEs are better than 60 with guessed frequencies.
- Include ALL AEs from the extracted AE frequency list. Do not drop AEs with high frequencies.
- AEs should include all TEAEs (treatment-emergent adverse events), including disease symptoms
  that were observed and reported during the trial (e.g., cough, dyspnea, pain). These are
  standard clinical trial safety reporting and belong in the AE profile.
- Comorbidity prevalences should reflect oncology trial populations (typically 5-15% each),
  NOT general population rates (which are 2-6x higher).

Respond with a single JSON object matching the schema — no markdown fences, no commentary.

EXTRACTED DATA:
{extracted}

GROUNDING REPORT:
{grounding}

DRUG(s): {drugs}
INDICATION: {indication}

JSON SCHEMA:
{schema}
"""
