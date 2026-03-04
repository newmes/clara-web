from __future__ import annotations

import logging

from rule_engine.schema import EvidenceBundle, RuleSet

log = logging.getLogger(__name__)

# Minimum threshold — if all pct fields are at or below this, assume 0-1 scale
_SCALE_THRESHOLD = 1.0

# Reserved trigger targets that don't need to match an AE name
_RESERVED_TRIGGER_TARGETS = {"Dose reduction", "Treatment discontinuation"}

# ---------------------------------------------------------------------------
# AE term synonyms — maps variant terms to a canonical form
# Used for fuzzy matching in DailyMed cross-check and severity cap
# ---------------------------------------------------------------------------
_AE_SYNONYMS: dict[str, str] = {
    # Lab values → clinical terms
    "decreased neutrophils": "neutropenia",
    "neutrophil count decreased": "neutropenia",
    "decreased hemoglobin": "anemia",
    "hemoglobin decreased": "anemia",
    "lower hemoglobin": "anemia",
    "decreased platelets": "thrombocytopenia",
    "platelet count decreased": "thrombocytopenia",
    "decreased lymphocytes": "lymphopenia",
    "lymphocyte count decreased": "lymphopenia",
    "decreased white blood cells": "leukopenia",
    "white blood cell count decreased": "leukopenia",
    "white blood cell decreased": "leukopenia",
    "increased alt": "alanine aminotransferase increased",
    "alt increased": "alanine aminotransferase increased",
    "increased ast": "aspartate aminotransferase increased",
    "ast increased": "aspartate aminotransferase increased",
    "increased creatinine": "blood creatinine increased",
    "creatinine increased": "blood creatinine increased",
    "increased bilirubin": "blood bilirubin increased",
    "bilirubin increased": "blood bilirubin increased",
    "increased alkaline phosphatase": "alkaline phosphatase increased",
    "decreased magnesium": "hypomagnesemia",
    # British → US spelling normalization (MedDRA uses British)
    "anaemia": "anemia",
    "diarrhoea": "diarrhea",
    "oedema": "edema",
    "oedema peripheral": "peripheral edema",
    "generalised oedema": "generalized edema",
    "pulmonary oedema": "pulmonary edema",
    "haemoglobin decreased": "anemia",
    "haemorrhage": "hemorrhage",
    "dyspnoea": "dyspnea",
    "hypokalaemia": "hypokalemia",
    "hyponatraemia": "hyponatremia",
    "hypocalcaemia": "hypocalcemia",
    "hypomagnesaemia": "hypomagnesemia",
    "hypercreatininaemia": "blood creatinine increased",
    "hyperglycaemia": "hyperglycemia",
    "leucopenia": "leukopenia",
    "paraesthesia": "paresthesia",
    # Clinical terms
    "neuropathy peripheral": "peripheral neuropathy",
    "peripheral sensory neuropathy": "peripheral neuropathy",
    "neuropathy-sensory": "peripheral neuropathy",
    "neuropathy sensory": "peripheral neuropathy",
    "rash maculo-papular": "rash",
    "rash maculopapular": "rash",
    "skin reaction": "rash",
    "decreased appetite": "appetite decreased",
    "weight loss": "weight decreased",
    "stomatitis/pharyngitis": "stomatitis",
    # Additional synonyms for GT matching
    "anorexia": "appetite decreased",
    "loss of appetite": "appetite decreased",
    "pyrexia": "fever",
    "febrile neutropenia": "febrile neutropenia",
    "blood alkaline phosphatase increased": "alkaline phosphatase increased",
    "blood lactate dehydrogenase increased": "lactate dehydrogenase increased",
    "mucosal inflammation": "stomatitis",
    "platelet count decreased": "thrombocytopenia",
    "haemoptysis": "hemoptysis",
    "hyperkalaemia": "hyperkalemia",
    "oedema": "edema",
}

# Lab-value patterns that should be replaced with clinical AE names
import re as _re

# Context-aware lab-threshold patterns: "decreased X <N" → clinical term
# Note: DailyMed text may have whitespace artifacts: "cells/mm 3", "g/dL", etc.
_LAB_THRESHOLD_PATTERNS: list[tuple[_re.Pattern, str]] = [
    (_re.compile(r"(?:decreased\s+)?neutrophils?\s*<\s*[\d,]+", _re.IGNORECASE), "neutropenia"),
    (_re.compile(r"(?:decreased\s+)?hemoglobin\s*<\s*[\d,]+", _re.IGNORECASE), "anemia"),
    (_re.compile(r"(?:decreased\s+)?(?:haemoglobin|hgb)\s*<\s*[\d,]+", _re.IGNORECASE), "anemia"),
    (_re.compile(r"(?:decreased\s+)?platelets?\s*<\s*[\d,]+", _re.IGNORECASE), "thrombocytopenia"),
    (_re.compile(r"(?:decreased\s+)?(?:lymphocytes?|wbc)\s*<\s*[\d,]+", _re.IGNORECASE), "leukopenia"),
    # Fallback: "<N cells/mm3" or "<N cells/mm 3" (whitespace variant)
    (_re.compile(r"<\s*[\d,]+\s*cells?/mm\s*3?", _re.IGNORECASE), "leukopenia"),
]

# Junk AE terms that are not actual drug-related adverse events
_JUNK_AE_PATTERNS: list[_re.Pattern] = [
    _re.compile(r"^other[_\s]", _re.IGNORECASE),              # other_miscellaneous_1, other_gi_adverse_reactions
    _re.compile(r"^transfusion", _re.IGNORECASE),              # transfusion: not an AE
    _re.compile(r"^malignant.neoplasm.progression", _re.IGNORECASE),  # disease progression, not drug AE
    _re.compile(r"^cardiovascular$", _re.IGNORECASE),          # too vague — not a specific AE
    _re.compile(r"^central.neurotoxicity$", _re.IGNORECASE),   # vague umbrella term
    _re.compile(r"kaposi.s.sarcoma", _re.IGNORECASE),          # opportunistic disease, not drug AE
    _re.compile(r"^neurologic[_\s]other", _re.IGNORECASE),     # vague catch-all
    _re.compile(r"^myelosuppression$", _re.IGNORECASE),       # umbrella term, not specific AE
    _re.compile(r"^febrile episode$", _re.IGNORECASE),        # vague
    _re.compile(r"^rash/erythema$", _re.IGNORECASE),          # combined — rash already captured
]

# Combined AE terms that should be split (e.g., "nausea_and_vomiting")
_COMBINED_TERM_MAP: dict[str, str] = {
    "nausea and vomiting": "nausea",
    "nausea_and_vomiting": "nausea",
    "nausea/vomiting": "nausea",
}


def _is_junk_ae(term: str) -> bool:
    """Check if an AE term is a junk/non-AE entry."""
    for pattern in _JUNK_AE_PATTERNS:
        if pattern.search(term):
            return True
    return False


def _normalize_ae_term(term: str) -> str:
    """Normalize an AE term using the synonym map."""
    t = term.lower().strip()
    return _AE_SYNONYMS.get(t, t)


def _split_combined_ae(term: str) -> str | None:
    """Split combined AE terms like 'nausea_and_vomiting' → 'nausea'.

    Returns the primary term if the input is a combined term, else None.
    """
    t = term.lower().strip()
    return _COMBINED_TERM_MAP.get(t)


def _normalize_diarrhea_variant(term: str) -> str | None:
    """Handle specific diarrhea variants like 'diarrhea_w/o_prior_colostomy' → 'diarrhea'."""
    t = term.lower().strip()
    if t.startswith("diarrhea") and ("w/o" in t or "without" in t or "colostomy" in t):
        return "diarrhea"
    if t.startswith("diarrhoea") and ("w/o" in t or "without" in t or "colostomy" in t):
        return "diarrhea"
    return None


def _normalize_ae_names(rule_set: RuleSet) -> list[str]:
    """Normalize lab-code AE names to clinical terms in the rule set.

    Renames AE events using _AE_SYNONYMS and fixes lab-value patterns like
    '<2000 cells/mm3' → clinical terms based on context.
    Also handles junk AE removal, combined-term splitting, and
    updates ae_risk_modifier references to match renamed AEs.
    """
    warnings = []
    rename_map: dict[str, str] = {}  # old_name_lower → new_name

    # --- Phase 0: Remove junk AEs ---
    kept = []
    for ae in rule_set.adverse_events:
        if _is_junk_ae(ae.event):
            warnings.append(f"Removed junk AE: '{ae.event}'")
        else:
            kept.append(ae)
    rule_set.adverse_events = kept

    # Build set of existing AE names (for combined-term dedup check)
    existing_terms = {ae.event.lower().strip() for ae in rule_set.adverse_events}

    seen_names: set[str] = set()
    for ae in rule_set.adverse_events:
        old_name = ae.event
        normalized = _normalize_ae_term(old_name)

        # Handle combined terms: "nausea_and_vomiting" → "nausea"
        split_term = _split_combined_ae(old_name)
        if split_term:
            normalized = split_term

        # Handle diarrhea variants
        diarrhea_fix = _normalize_diarrhea_variant(old_name)
        if diarrhea_fix:
            normalized = diarrhea_fix

        # Handle context-aware lab-threshold patterns
        for pattern, clinical_term in _LAB_THRESHOLD_PATTERNS:
            if pattern.search(old_name):
                normalized = clinical_term
                break

        if normalized != old_name.lower().strip():
            new_name = normalized.title() if normalized[0].islower() else normalized
            # Always rename — dedup step will merge duplicates later
            warnings.append(f"AE name normalized: '{old_name}' → '{new_name}'")
            rename_map[old_name.lower()] = new_name
            ae.event = new_name

        seen_names.add(ae.event.lower())

    # Update ae_risk_modifier references to match renamed AEs
    # Use both rename_map and synonym map to catch all references
    ae_names_set = {ae.event.lower() for ae in rule_set.adverse_events}
    for comorb in rule_set.comorbidities:
        for mod in comorb.ae_risk_modifiers:
            # Check rename_map first (from this pass)
            new_name = rename_map.get(mod.ae.lower())
            if new_name:
                mod.ae = new_name
            # Also normalize via synonym map if reference doesn't match any AE
            elif mod.ae.lower() not in ae_names_set:
                normalized = _normalize_ae_term(mod.ae)
                if normalized != mod.ae.lower():
                    candidate = normalized.title() if normalized[0].islower() else normalized
                    if candidate.lower() in ae_names_set:
                        mod.ae = candidate

    return warnings


# Comorbidity → AE category → risk multiplier mapping for auto-populating ae_risk_modifiers
_COMORBIDITY_AE_MODIFIERS: dict[str, list[tuple[str, float]]] = {
    "hepatic": [  # hepatic impairment, cirrhosis, chronic liver disease
        ("hepatic", 1.5),
        ("gi", 1.2),
    ],
    "renal": [  # chronic kidney disease, renal impairment
        ("metabolic", 1.3),
        ("heme", 1.2),
        ("renal", 1.2),
    ],
    "cardiac": [  # cardiovascular disease, heart failure
        ("cardiac", 1.4),
        ("constitutional", 1.1),
    ],
    "pulmonary": [  # chronic pulmonary disease, COPD
        ("respiratory", 1.3),
    ],
    "autoimmune": [  # autoimmune disease
        ("irae", 1.5),
        ("derm", 1.3),
    ],
    "diabetic": [  # diabetes mellitus
        ("metabolic", 1.3),
        ("renal", 1.2),
    ],
    "obesity": [  # obesity
        ("metabolic", 1.2),
        ("cardiac", 1.1),
    ],
}

# Map comorbidity condition names to categories
_COMORBIDITY_CATEGORY: dict[str, str] = {
    "hepatic impairment": "hepatic",
    "chronic liver disease": "hepatic",
    "cirrhosis": "hepatic",
    "hepatic impairment (moderate)": "hepatic",
    "chronic kidney disease": "renal",
    "chronic kidney disease (ckd)": "renal",
    "chronic kidney disease (stage 3 or higher)": "renal",
    "renal impairment": "renal",
    "cardiovascular disease": "cardiac",
    "heart failure": "cardiac",
    "hypertension": "cardiac",
    "chronic pulmonary disease": "pulmonary",
    "copd": "pulmonary",
    "autoimmune disease": "autoimmune",
    "diabetes mellitus": "diabetic",
    "diabetes": "diabetic",
    "obesity": "obesity",
}

# NOTE: _AE_CATEGORY_KEYWORDS and _classify_ae_category are defined below (near line 410)
# in the onset auto-correction section. They are the single canonical definitions used
# by both onset auto-correction and comorbidity ae_risk_modifiers auto-population.


def _auto_populate_ae_risk_modifiers(rule_set: RuleSet) -> list[str]:
    """Auto-populate ae_risk_modifiers for comorbidities with impacts_dosing=True but empty modifiers.

    Uses comorbidity type × AE category mapping to assign clinically plausible risk multipliers.
    """
    from rule_engine.schema import AERiskModifier

    warnings = []
    ae_names = [ae.event for ae in rule_set.adverse_events]

    for comorb in rule_set.comorbidities:
        if not comorb.impacts_dosing or comorb.ae_risk_modifiers:
            continue

        # Determine comorbidity category
        comorb_cat = _COMORBIDITY_CATEGORY.get(comorb.condition.lower().strip())
        if comorb_cat is None:
            # Fuzzy match
            for key, cat in _COMORBIDITY_CATEGORY.items():
                if key in comorb.condition.lower():
                    comorb_cat = cat
                    break
        if comorb_cat is None:
            continue

        modifiers_config = _COMORBIDITY_AE_MODIFIERS.get(comorb_cat, [])
        if not modifiers_config:
            continue

        new_modifiers = []
        for ae_cat, multiplier in modifiers_config:
            for ae_name in ae_names:
                if _classify_ae_category(ae_name) == ae_cat:
                    new_modifiers.append(AERiskModifier(ae=ae_name, risk_multiplier=multiplier))

        if new_modifiers:
            comorb.ae_risk_modifiers = new_modifiers
            warnings.append(
                f"Auto-populated {len(new_modifiers)} ae_risk_modifiers for "
                f"'{comorb.condition}' ({comorb_cat})"
            )

    return warnings


# Default comorbidities by indication keyword — injected when LLM omits them


def _rescale_if_needed(rule_set: RuleSet) -> list[str]:
    """Detect and fix 0-1 scale percentages that should be 0-100.

    Heuristic: if sex pct_male + pct_female <= 2.0 (instead of ~100),
    the model output 0-1 fractions. Multiply all percentage fields by 100.
    """
    fixes = []
    demo = rule_set.demographics

    if demo.sex.pct_male is not None and demo.sex.pct_female is not None:
        sex_total = demo.sex.pct_male + demo.sex.pct_female
        if sex_total <= 2.0 and sex_total > 0:
            factor = 100.0
            demo.sex.pct_male = round(demo.sex.pct_male * factor, 1)
            demo.sex.pct_female = round(demo.sex.pct_female * factor, 1)
            fixes.append(f"Auto-rescaled sex percentages from 0-1 to 0-100 (was {sex_total:.2f})")

    if demo.race_ethnicity:
        race_total = sum(r.pct for r in demo.race_ethnicity)
        if race_total <= 2.0 and race_total > 0:
            for r in demo.race_ethnicity:
                r.pct = round(r.pct * 100.0, 1)
            fixes.append(f"Auto-rescaled race/ethnicity from 0-1 to 0-100 (was {race_total:.2f})")

    for comorb in rule_set.comorbidities:
        if comorb.prevalence_pct is not None and comorb.prevalence_pct <= _SCALE_THRESHOLD and comorb.prevalence_pct > 0:
            comorb.prevalence_pct = round(comorb.prevalence_pct * 100.0, 1)

    eff = rule_set.efficacy
    if eff.overall_response_rate_pct is not None and eff.overall_response_rate_pct <= _SCALE_THRESHOLD and eff.overall_response_rate_pct > 0:
        eff.overall_response_rate_pct = round(eff.overall_response_rate_pct * 100.0, 1)
        if eff.complete_response_rate_pct is not None:
            eff.complete_response_rate_pct = round(eff.complete_response_rate_pct * 100.0, 1)
        fixes.append("Auto-rescaled efficacy ORR/CR from 0-1 to 0-100")

    if fixes:
        log.info("Scale corrections applied: %s", "; ".join(fixes))
    return fixes


def _deduplicate_aes(rule_set: RuleSet) -> list[str]:
    """Remove duplicate AE entries using normalized term matching.

    Uses _normalize_ae_term() so synonym pairs (e.g. "White Blood Cell
    Decreased" and "Leukopenia") are recognised as duplicates even when
    _normalize_ae_names() skipped renaming to avoid a name collision.
    Keeps the first (higher-frequency after sort) occurrence.
    """
    seen: dict[str, int] = {}  # normalized_term -> index in unique
    unique = []
    removed = []
    for ae in rule_set.adverse_events:
        key = _normalize_ae_term(ae.event)
        if key in seen:
            removed.append(ae.event)
        else:
            seen[key] = len(unique)
            unique.append(ae)
    if removed:
        rule_set.adverse_events = unique
        log.info("Removed %d duplicate AEs: %s", len(removed), removed[:5])
    return [f"Removed duplicate AE: {name}" for name in removed]


def _filter_intercurrent_illnesses(rule_set: RuleSet) -> list[str]:
    """Remove intercurrent illnesses that are not drug-related AEs.

    Pandemic-era trials (2020-2023) captured COVID-19 and other intercurrent
    illnesses as AEs. These contaminate the drug safety profile.
    """
    from rule_engine.prompts import _is_intercurrent_illness

    kept = []
    removed_warnings = []
    for ae in rule_set.adverse_events:
        if _is_intercurrent_illness(ae.event):
            removed_warnings.append(f"Removed intercurrent illness AE: '{ae.event}'")
            log.info("Intercurrent illness filter: removed '%s'", ae.event)
        else:
            kept.append(ae)

    if removed_warnings:
        rule_set.adverse_events = kept

    return removed_warnings


def _cap_severity_grade34(
    rule_set: RuleSet, bundle: EvidenceBundle, warnings: list[str]
) -> None:
    """Cap grade 3-4 severity using DailyMed grade34_pct when available."""
    # Build lookup: normalized_ae_term -> grade34_pct (from DailyMed ae_table)
    label_grade34: dict[str, float] = {}
    for drug, sde in bundle.per_drug.items():
        if not sde.dailymed.found or not sde.dailymed.ae_table:
            continue
        for ae_entry in sde.dailymed.ae_table:
            term_raw = ae_entry.get("term", "").lower().strip()
            term = _normalize_ae_term(term_raw)
            g34 = ae_entry.get("grade34_pct")
            if term and g34 is not None:
                try:
                    g34_val = float(g34)
                except (ValueError, TypeError):
                    continue
                # Keep the highest grade34_pct across drugs (combo case)
                existing = label_grade34.get(term)
                if existing is None or g34_val > existing:
                    label_grade34[term] = g34_val

    if not label_grade34:
        return

    for ae in rule_set.adverse_events:
        ae_normalized = _normalize_ae_term(ae.event)
        cap = label_grade34.get(ae_normalized)
        if cap is None:
            continue

        # Sum of grade_3 + grade_4 + grade_5 in the distribution
        high_grades = sum(
            v for k, v in ae.severity_distribution.items()
            if k in ("grade_3", "grade_4", "grade_5")
        )
        low_grades = sum(
            v for k, v in ae.severity_distribution.items()
            if k in ("grade_1", "grade_2")
        )

        # Allow 2% tolerance
        if high_grades > cap + 2.0 and low_grades > 0:
            excess = high_grades - cap
            high_scale = cap / high_grades if high_grades > 0 else 0
            new_dist = {}
            for k, v in ae.severity_distribution.items():
                if k in ("grade_3", "grade_4", "grade_5"):
                    new_dist[k] = round(v * high_scale, 1)
                else:
                    share = v / low_grades if low_grades > 0 else 0.5
                    new_dist[k] = round(v + excess * share, 1)
            ae.severity_distribution = new_dist
            warnings.append(
                f"Severity cap: '{ae.event}' grade≥3 was {high_grades:.1f}%, "
                f"capped to {cap:.1f}% (from DailyMed label)"
            )


def _is_checkpoint_inhibitor_from_bundle(bundle: EvidenceBundle) -> bool:
    """Detect if any drug in the bundle is a checkpoint inhibitor."""
    _TARGETS = {"PDCD1", "CD274", "CTLA4"}
    _MOA_KW = ["pd-1", "pd-l1", "ctla-4", "checkpoint", "anti-pd"]
    for drug in bundle.drugs:
        sd = bundle.per_drug.get(drug)
        if sd is None:
            continue
        for t in sd.drugbank.targets:
            if t.get("uniprot_name", "").upper() in _TARGETS:
                return True
        if sd.drugbank.moa and any(k in sd.drugbank.moa.lower() for k in _MOA_KW):
            return True
        if sd.chembl.mechanism_of_action and any(
            k in sd.chembl.mechanism_of_action.lower() for k in _MOA_KW
        ):
            return True
    return False


_AE_CATEGORY_ONSETS: dict[str, tuple[int, int]] = {
    # (min_days, max_days) for each AE category
    "gi": (2, 8),  # nausea, vomiting, diarrhea, constipation, stomatitis
    "derm": (12, 32),  # rash, pruritus, dry skin, alopecia, skin reaction
    "heme": (8, 21),  # neutropenia, anemia, thrombocytopenia, lymphopenia, leukopenia
    "hepatic": (18, 48),  # ALT/AST increased, bilirubin, hepatitis
    "endocrine": (35, 95),  # hypothyroidism, hyperthyroidism, adrenal insufficiency
    "neuro": (10, 35),  # neuropathy, headache, dizziness, dysgeusia
    "msk": (12, 30),  # arthralgia, myalgia, back pain, pain in extremity
    "constitutional": (5, 16),  # fatigue, asthenia, decreased appetite, weight loss
    "respiratory": (25, 65),  # cough, dyspnea, pneumonitis
    "cardiac": (12, 35),  # QT prolongation, hypertension, edema
    "metabolic": (5, 25),  # hyperglycemia, hyponatremia, hypophosphatemia
    "infection": (18, 48),  # URI, UTI, pneumonia
    "irae": (35, 95),  # immune-related AEs (colitis, hepatitis, pneumonitis from immunotherapy)
}

_AE_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "gi": ["nausea", "vomiting", "diarr", "constipat", "stomatitis", "mucosit", "dyspepsia", "abdominal"],
    "derm": ["rash", "prurit", "dry skin", "alopecia", "skin", "nail", "palmar-plantar", "erythema"],
    "heme": ["neutrop", "anemia", "thrombocytop", "lymphop", "leukop", "pancytop", "febrile neutro"],
    "hepatic": ["alt ", "ast ", "aminotransfer", "bilirubin", "hepat", "liver"],
    "endocrine": ["hypothyroid", "hyperthyroid", "adrenal", "thyroid"],
    "neuro": ["neuropath", "headache", "dizziness", "dysgeusia", "paresthesia", "tremor"],
    "msk": ["arthralg", "myalg", "back pain", "pain in extremit", "musculoskeletal"],
    "constitutional": ["fatigue", "asthenia", "appetite", "weight", "pyrexia", "fever", "malaise"],
    "respiratory": ["cough", "dyspn", "pneumonitis", "interstitial lung"],
    "cardiac": ["qt prolong", "hypertens", "edema", "cardiac", "tachycard"],
    "metabolic": ["hyperglyc", "hyponatr", "hypophos", "hypokale", "hyperkalemia", "hypomag"],
    "infection": ["infection", "pneumonia", "urinary tract", "sepsis", "nasopharyng"],
    "irae": ["colitis", "immune-mediated"],
    "renal": ["creatinine", "renal", "proteinuria", "nephro"],
}


def _classify_ae_category(ae_name: str) -> str:
    """Classify an AE into a category for onset estimation."""
    name_lower = ae_name.lower()
    for cat, keywords in _AE_CATEGORY_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return cat
    return "constitutional"  # default


def _auto_correct_onsets(rule_set: RuleSet) -> list[str]:
    """Auto-correct fabricated onset values when all AEs share identical or near-identical onsets.

    Uses AE category-based default ranges to assign clinically plausible, diverse onset values.
    Triggers when:
      - Onset diversity is catastrophically low (<=3 unique values across 8+ AEs)
      - OR >50% of onsets are round multiples of 10 (LLM round-number fabrication)
      - OR >60% of onsets are multiples of 7 (LLM weekly-cadence fabrication)
    """
    import random

    warnings = []
    aes = rule_set.adverse_events
    if len(aes) < 5:
        return warnings

    onsets = [ae.median_onset_days for ae in aes]
    unique_onsets = set(onsets)

    # Trigger 1: catastrophically low diversity
    low_diversity = len(unique_onsets) <= 3 and len(aes) >= 8
    # Trigger 2: >50% round multiples of 10 (LLM round-number fabrication)
    round_tens = sum(1 for o in onsets if o > 0 and o % 10 == 0)
    high_round_pct = len(onsets) >= 5 and round_tens / len(onsets) > 0.5
    # Trigger 3: >60% multiples of 7 (LLM weekly-cadence fabrication)
    round_sevens = sum(1 for o in onsets if o > 0 and o % 7 == 0)
    high_seven_pct = len(onsets) >= 5 and round_sevens / len(onsets) > 0.6

    if not low_diversity and not high_round_pct and not high_seven_pct:
        return warnings

    random.seed(42)  # deterministic for reproducibility
    used_values: set[int] = set()
    treatment_dur = rule_set.treatment_duration_days or 365

    for ae in aes:
        cat = _classify_ae_category(ae.event)
        lo, hi = _AE_CATEGORY_ONSETS.get(cat, (5, 30))
        hi = min(hi, treatment_dur)
        # Pick a value not yet used
        for _ in range(50):
            val = random.randint(lo, hi)
            if val not in used_values:
                break
        used_values.add(val)
        ae.median_onset_days = val

    warnings.append(
        f"Auto-corrected onset values: {len(unique_onsets)} unique values across {len(aes)} AEs "
        f"detected as fabricated — replaced with category-based clinically plausible onsets"
    )
    log.info("Auto-corrected fabricated onset values for %s", " + ".join(rule_set.drugs))
    return warnings


# Category-specific severity grade ratios: (grade_1_pct, grade_2_pct, grade_3_pct, grade_4_pct)
# as fraction of total frequency_pct. Must sum to 1.0.
_AE_CATEGORY_SEVERITY: dict[str, tuple[float, float, float, float]] = {
    "gi": (0.58, 0.32, 0.08, 0.02),        # mostly mild — nausea, diarrhea
    "derm": (0.65, 0.28, 0.05, 0.02),       # mostly grade 1 — rash, pruritus
    "heme": (0.28, 0.40, 0.22, 0.10),       # significant grade 3-4 — neutropenia, anemia
    "hepatic": (0.35, 0.38, 0.20, 0.07),    # moderate high-grade — ALT/AST elevations
    "endocrine": (0.50, 0.35, 0.12, 0.03),  # mostly mild — hypothyroidism
    "neuro": (0.55, 0.33, 0.09, 0.03),      # mostly mild — neuropathy, headache
    "msk": (0.52, 0.38, 0.08, 0.02),        # mostly mild — arthralgia, myalgia
    "constitutional": (0.60, 0.30, 0.08, 0.02),  # fatigue, appetite loss
    "respiratory": (0.42, 0.35, 0.16, 0.07),     # can be serious — pneumonitis
    "cardiac": (0.30, 0.38, 0.22, 0.10),    # often higher grade — QT, hypertension
    "metabolic": (0.48, 0.35, 0.12, 0.05),  # lab values — hyponatremia, hyperglycemia
    "infection": (0.40, 0.38, 0.16, 0.06),  # infections can escalate
    "irae": (0.35, 0.35, 0.20, 0.10),       # immune-related — colitis, hepatitis
}


def _auto_correct_severity(rule_set: RuleSet) -> list[str]:
    """Auto-correct fabricated severity distributions when >50% of AEs have identical grade ratios.

    Replaces mechanical distributions (e.g., all grade_1=100% or 50/30/15/5 pattern)
    with category-specific grade distributions that reflect clinical reality.
    """
    import random
    from collections import Counter

    warnings = []
    aes = rule_set.adverse_events
    if len(aes) < 5:
        return warnings

    # Compute grade_1 ratio for each AE
    grade1_ratios = []
    for ae in aes:
        if ae.frequency_pct > 0 and "grade_1" in ae.severity_distribution:
            ratio = round(ae.severity_distribution["grade_1"] / ae.frequency_pct, 2)
            grade1_ratios.append(ratio)

    if len(grade1_ratios) < 5:
        return warnings

    ratio_counts = Counter(grade1_ratios)
    most_common_ratio, most_common_count = ratio_counts.most_common(1)[0]

    # Trigger 1: >50% of AEs share the same grade_1 ratio
    ratio_triggered = most_common_count / len(grade1_ratios) > 0.5

    # Trigger 2: ≥3 AEs have the exact mechanical 50/30/15/5 pattern
    n_mechanical = 0
    for ae in aes:
        if ae.frequency_pct <= 0:
            continue
        sd = ae.severity_distribution
        g1 = sd.get("grade_1", 0)
        g2 = sd.get("grade_2", 0)
        g3 = sd.get("grade_3", 0)
        r1 = round(g1 / ae.frequency_pct * 100)
        r2 = round(g2 / ae.frequency_pct * 100)
        r3 = round(g3 / ae.frequency_pct * 100)
        if r1 == 50 and r2 == 30 and r3 == 15:
            n_mechanical += 1
    mechanical_triggered = n_mechanical >= 3

    if not ratio_triggered and not mechanical_triggered:
        return warnings

    random.seed(43)  # different seed from onset correction
    for ae in aes:
        if ae.frequency_pct <= 0:
            continue
        cat = _classify_ae_category(ae.event)
        base_g1, base_g2, base_g3, base_g4 = _AE_CATEGORY_SEVERITY.get(
            cat, (0.55, 0.32, 0.10, 0.03)
        )

        # Add small random jitter (±5%) for diversity
        def _jitter(val: float) -> float:
            return max(0.01, val + random.uniform(-0.05, 0.05))

        g1 = _jitter(base_g1)
        g2 = _jitter(base_g2)
        g3 = _jitter(base_g3)
        g4 = _jitter(base_g4)

        # Normalize to sum to 1.0
        total = g1 + g2 + g3 + g4
        g1, g2, g3, g4 = g1/total, g2/total, g3/total, g4/total

        # Apply to frequency_pct
        freq = ae.frequency_pct
        new_dist = {}
        new_dist["grade_1"] = round(freq * g1, 1)
        new_dist["grade_2"] = round(freq * g2, 1)
        if freq * g3 >= 0.1:
            new_dist["grade_3"] = round(freq * g3, 1)
        if freq * g4 >= 0.1:
            new_dist["grade_4"] = round(freq * g4, 1)

        # Adjust grade_1 so total matches frequency_pct
        assigned = sum(new_dist.values())
        new_dist["grade_1"] = round(new_dist["grade_1"] + (freq - assigned), 1)

        ae.severity_distribution = new_dist

    warnings.append(
        f"Auto-corrected severity distributions: {most_common_count}/{len(grade1_ratios)} AEs had "
        f"identical grade_1 ratio ({most_common_ratio}) — replaced with category-specific distributions"
    )
    log.info("Auto-corrected fabricated severity distributions for %s", " + ".join(rule_set.drugs))
    return warnings


# Category-specific trigger probability ranges: (grade3_hold_pct, grade4_reduce_pct)
# Represents typical clinical dose modification probabilities for each AE category
_AE_CATEGORY_TRIGGERS: dict[str, tuple[float, float]] = {
    "hematologic":    (55.0, 30.0),  # aggressive dose mods for blood toxicity
    "gi":             (30.0, 12.0),  # moderate — often managed supportively
    "hepatic":        (50.0, 25.0),  # liver toxicity requires caution
    "renal":          (45.0, 22.0),
    "dermatologic":   (25.0, 10.0),  # skin toxicity often managed topically
    "neurologic":     (40.0, 20.0),
    "cardiac":        (55.0, 28.0),  # cardiac events require aggressive response
    "pulmonary":      (50.0, 25.0),
    "endocrine":      (20.0, 8.0),   # often managed with replacement therapy
    "musculoskeletal":(25.0, 10.0),
    "ocular":         (30.0, 12.0),
    "infusion":       (35.0, 15.0),
    "constitutional": (20.0, 8.0),   # fatigue etc. — rarely dose-limiting
}


def _auto_correct_triggers(rule_set: RuleSet) -> list[str]:
    """Auto-correct identical trigger patterns when >60% of triggered AEs share the same pattern.

    Uses AE category-based dose modification probabilities with jitter.
    """
    import random

    warnings: list[str] = []
    aes = rule_set.adverse_events

    # Collect trigger signatures (condition, probability) per AE
    triggered_aes = [(ae, tuple(
        (t.condition, t.probability_pct) for t in sorted(ae.triggers, key=lambda x: x.condition)
    )) for ae in aes if ae.triggers]

    if len(triggered_aes) < 3:
        return warnings

    sig_counter: dict[tuple, int] = {}
    for _, sig in triggered_aes:
        sig_counter[sig] = sig_counter.get(sig, 0) + 1

    most_common_sig = max(sig_counter, key=sig_counter.get)
    most_common_count = sig_counter[most_common_sig]

    if most_common_count / len(triggered_aes) <= 0.6:
        return warnings

    # Auto-correct: diversify trigger probabilities by AE category
    random.seed(44)  # deterministic
    for ae in aes:
        if not ae.triggers:
            continue
        cat = _classify_ae_category(ae.event)
        base_g3, base_g4 = _AE_CATEGORY_TRIGGERS.get(cat, (30.0, 15.0))

        for trigger in ae.triggers:
            jitter = random.uniform(-8.0, 8.0)
            if "3" in trigger.condition:
                trigger.probability_pct = round(max(5.0, min(80.0, base_g3 + jitter)), 1)
            elif "4" in trigger.condition:
                trigger.probability_pct = round(max(2.0, min(60.0, base_g4 + jitter)), 1)
            else:
                trigger.probability_pct = round(max(5.0, trigger.probability_pct + jitter), 1)

    warnings.append(
        f"Auto-corrected trigger patterns: {most_common_count}/{len(triggered_aes)} triggered AEs had "
        f"identical pattern — replaced with category-specific probabilities"
    )
    log.info("Auto-corrected fabricated trigger patterns for %s", " + ".join(rule_set.drugs))
    return warnings


def _validate_onset_plausibility(rule_set: RuleSet) -> list[str]:
    """Flag suspicious round-number onset patterns suggesting LLM fabrication."""
    warnings = []
    aes = rule_set.adverse_events
    if len(aes) < 5:
        return warnings

    onsets = [ae.median_onset_days for ae in aes]

    # Check: are >60% of onsets exact multiples of 7?
    multiples_of_7 = sum(1 for o in onsets if o > 0 and o % 7 == 0)
    if multiples_of_7 / len(onsets) > 0.6:
        warnings.append(
            f"Onset plausibility: {multiples_of_7}/{len(onsets)} AE onsets are exact multiples of 7 — "
            "likely fabricated round numbers"
        )

    # Check: are >50% of onsets round tens (10, 20, 30, 60, 90)?
    round_tens = sum(1 for o in onsets if o > 0 and o % 10 == 0)
    if round_tens / len(onsets) > 0.5:
        warnings.append(
            f"Onset plausibility: {round_tens}/{len(onsets)} AE onsets are round multiples of 10 — "
            "likely fabricated"
        )

    # Check: do all onsets come from a tiny set of values?
    unique_onsets = set(onsets)
    if len(unique_onsets) <= 3 and len(onsets) >= 8:
        warnings.append(
            f"Onset plausibility: only {len(unique_onsets)} unique onset values across {len(onsets)} AEs"
        )

    return warnings


def _validate_severity_diversity(rule_set: RuleSet) -> list[str]:
    """Flag identical severity distribution ratios suggesting LLM fabrication."""
    warnings = []
    aes = rule_set.adverse_events
    if len(aes) < 5:
        return warnings

    # Compute grade_1 ratio for each AE
    grade1_ratios = []
    for ae in aes:
        if ae.frequency_pct > 0 and "grade_1" in ae.severity_distribution:
            ratio = round(ae.severity_distribution["grade_1"] / ae.frequency_pct, 2)
            grade1_ratios.append(ratio)

    if len(grade1_ratios) < 5:
        return warnings

    # Check: are >50% of AEs using the same grade_1 ratio?
    from collections import Counter
    ratio_counts = Counter(grade1_ratios)
    most_common_ratio, most_common_count = ratio_counts.most_common(1)[0]
    if most_common_count / len(grade1_ratios) > 0.5:
        warnings.append(
            f"Severity diversity: {most_common_count}/{len(grade1_ratios)} AEs have identical "
            f"grade_1 ratio ({most_common_ratio}) — severity distributions likely fabricated"
        )

    return warnings


def validate_rule_set(rule_set: RuleSet, bundle: EvidenceBundle | None = None) -> list[str]:
    """Run clinical plausibility checks on a RuleSet.

    Performs auto-corrections where possible (scale, dedup), then validates.
    If *bundle* is provided, also runs DailyMed cross-checks.
    Returns list of warnings for issues that couldn't be auto-fixed.
    """
    warnings = []
    drug_label = " + ".join(rule_set.drugs)

    # --- Auto-corrections (modify rule_set in place) ---
    # Normalize AE names (lab-codes → clinical terms) before dedup
    name_fixes = _normalize_ae_names(rule_set)
    warnings.extend(name_fixes)
    scale_fixes = _rescale_if_needed(rule_set)
    dedup_fixes = _deduplicate_aes(rule_set)
    warnings.extend(scale_fixes)
    warnings.extend(dedup_fixes)

    # Note: comorbidity generation is handled by the LLM re-prompt in
    # agent_multistage.py — no hard-coded fallback here.

    # Intercurrent illness filter (pandemic-era trial contamination)
    illness_warnings = _filter_intercurrent_illnesses(rule_set)
    warnings.extend(illness_warnings)

    # Remove zero-frequency AEs
    nonzero_aes = [ae for ae in rule_set.adverse_events if ae.frequency_pct > 0]
    if len(nonzero_aes) < len(rule_set.adverse_events):
        n_removed = len(rule_set.adverse_events) - len(nonzero_aes)
        rule_set.adverse_events = nonzero_aes
        warnings.append(f"Removed {n_removed} AEs with zero frequency")

    # Evidence cross-checks — run before severity_distribution rescaling
    # because correcting frequency_pct changes the target sum for severity_distribution.
    # NOTE: _cross_check_dailymed is DISABLED because _correct_ae_frequencies()
    # in agent_multistage.py already aligns frequencies using priority
    # PDS > CT.gov > DailyMed.  The DailyMed cross-check was overriding
    # CT.gov-based corrections back to inflated DailyMed label values.
    if bundle is not None:
        # _cross_check_dailymed(rule_set, bundle, warnings)  # disabled — see above
        _cross_check_ctgov(rule_set, bundle, warnings)
        _cross_check_onsides(rule_set, bundle, warnings)

    # --- Validation checks ---
    demo = rule_set.demographics

    # Sex percentages should sum to ~100% — auto-correct if not
    if demo.sex.pct_male is None or demo.sex.pct_female is None:
        sex_total = 0
    else:
        sex_total = demo.sex.pct_male + demo.sex.pct_female
    if sex_total > 0 and abs(sex_total - 100.0) > 5.0:
        # Normalize to 100%
        scale = 100.0 / sex_total
        old_male, old_female = demo.sex.pct_male, demo.sex.pct_female
        demo.sex.pct_male = round(demo.sex.pct_male * scale, 1)
        demo.sex.pct_female = round(100.0 - demo.sex.pct_male, 1)  # ensure exact sum
        warnings.append(
            f"Auto-corrected sex percentages: {old_male:.1f}%/{old_female:.1f}% "
            f"(sum={sex_total:.1f}%) → {demo.sex.pct_male}%/{demo.sex.pct_female}%"
        )

    # Race/ethnicity percentages should sum to ~100%
    if demo.race_ethnicity:
        race_total = sum(r.pct for r in demo.race_ethnicity)
        if abs(race_total - 100.0) > 10.0:
            warnings.append(f"Race/ethnicity percentages sum to {race_total:.1f}%, expected ~100%")

    # Mean age within [min, max]
    if demo.age.min is not None and demo.age.mean is not None and demo.age.max is not None:
        if not (demo.age.min <= demo.age.mean <= demo.age.max):
            warnings.append(f"Mean age {demo.age.mean} not within [{demo.age.min}, {demo.age.max}]")

    # Efficacy: CR <= ORR
    eff = rule_set.efficacy
    if (eff.complete_response_rate_pct is not None and eff.overall_response_rate_pct is not None
            and eff.complete_response_rate_pct > eff.overall_response_rate_pct):
        warnings.append(
            f"CR ({eff.complete_response_rate_pct}%) > ORR ({eff.overall_response_rate_pct}%)"
        )

    # AE onset check
    for ae in rule_set.adverse_events:
        if ae.median_onset_days > rule_set.treatment_duration_days:
            warnings.append(
                f"AE '{ae.event}' onset ({ae.median_onset_days}d) > treatment duration ({rule_set.treatment_duration_days}d)"
            )

    # --- New schema checks ---

    # Build AE name set for referential integrity checks
    ae_names = {ae.event for ae in rule_set.adverse_events}
    ae_names_lower = {name.lower() for name in ae_names}

    # Fix negative or zero severity grades before rescaling
    def _get_category_severity(cat: str, freq: float) -> dict[str, float]:
        base = _AE_CATEGORY_SEVERITY.get(cat, (0.55, 0.32, 0.10, 0.03))
        dist = {
            "grade_1": round(freq * base[0], 1),
            "grade_2": round(freq * base[1], 1),
            "grade_3": round(freq * base[2], 1),
            "grade_4": round(freq * base[3], 1),
        }
        assigned = sum(dist.values())
        dist["grade_1"] = round(dist["grade_1"] + (freq - assigned), 1)
        return {k: v for k, v in dist.items() if v >= 0.1}

    for ae in rule_set.adverse_events:
        has_negative = any(v < 0 for v in ae.severity_distribution.values())
        if has_negative:
            category = _classify_ae_category(ae.event)
            ae.severity_distribution = _get_category_severity(category, ae.frequency_pct)
            warnings.append(
                f"Fixed negative severity grades for '{ae.event}' — "
                f"replaced with category-specific ({category}) distribution"
            )

    # Severity distribution sum check — auto-rescale if mismatch
    # Runs AFTER DailyMed auto-correction so frequency_pct is already corrected
    for ae in rule_set.adverse_events:
        dist_sum = sum(ae.severity_distribution.values())
        if dist_sum > 0 and abs(dist_sum - ae.frequency_pct) > 0.5:
            # Proportional rescaling: adjust each grade so they sum to frequency_pct
            scale_factor = ae.frequency_pct / dist_sum
            ae.severity_distribution = {
                k: round(v * scale_factor, 1)
                for k, v in ae.severity_distribution.items()
            }
            warnings.append(
                f"Auto-rescaled severity_distribution for '{ae.event}': "
                f"sum was {dist_sum:.1f}, target {ae.frequency_pct:.1f}"
            )
            log.info(
                "Severity distribution rescaled for '%s': %s → sum=%.1f",
                ae.event, ae.severity_distribution, ae.frequency_pct,
            )

    # Grade 3-4 severity cap using DailyMed grade34_pct
    if bundle is not None:
        _cap_severity_grade34(rule_set, bundle, warnings)

    # Auto-populate empty ae_risk_modifiers for comorbidities with impacts_dosing=True
    comorb_fixes = _auto_populate_ae_risk_modifiers(rule_set)
    warnings.extend(comorb_fixes)

    # ae_risk_modifiers referential integrity
    for comorb in rule_set.comorbidities:
        for mod in comorb.ae_risk_modifiers:
            if mod.ae.lower() not in ae_names_lower:
                warnings.append(
                    f"Comorbidity '{comorb.condition}': ae_risk_modifier references "
                    f"'{mod.ae}' which is not in adverse_events"
                )

    # triggers referential integrity
    for ae in rule_set.adverse_events:
        for trigger in ae.triggers:
            if (trigger.target_ae not in _RESERVED_TRIGGER_TARGETS
                    and trigger.target_ae.lower() not in ae_names_lower):
                warnings.append(
                    f"AE '{ae.event}': trigger target_ae '{trigger.target_ae}' "
                    f"not in adverse_events and not a reserved keyword"
                )

    # Combo source_drug check — auto-assign first drug if None
    if len(rule_set.drugs) > 1:
        drugs_lower = {d.lower() for d in rule_set.drugs}
        for ae in rule_set.adverse_events:
            if ae.source_drug is None:
                ae.source_drug = rule_set.drugs[0]
                warnings.append(
                    f"Combo: AE '{ae.event}' had null source_drug — auto-assigned to '{rule_set.drugs[0]}'"
                )
            elif ae.source_drug.lower() not in drugs_lower:
                # Auto-fix concatenated combo source_drug (e.g. "Gemcitabine+Capecitabine")
                if "+" in ae.source_drug:
                    parts = [p.strip() for p in ae.source_drug.split("+")]
                    matched = [p for p in parts if p.lower() in drugs_lower]
                    if matched:
                        ae.source_drug = matched[0]
                        warnings.append(
                            f"Combo: AE '{ae.event}' source_drug split from "
                            f"'{'+'.join(parts)}' → '{ae.source_drug}'"
                        )
                        continue
                # Fallback: assign to first drug
                old_sd = ae.source_drug
                ae.source_drug = rule_set.drugs[0]
                warnings.append(
                    f"Combo: AE '{ae.event}' source_drug '{old_sd}' "
                    f"not in drugs list {rule_set.drugs} — auto-assigned to '{rule_set.drugs[0]}'"
                )

    # Auto-correct fabricated trigger patterns (must run BEFORE monotonicity check)
    warnings.extend(_auto_correct_triggers(rule_set))

    # Trigger monotonicity check (only AEs that actually have triggers)
    if len(rule_set.adverse_events) >= 5:
        trigger_patterns: list[str] = []
        for ae in rule_set.adverse_events:
            if not ae.triggers:  # skip AEs with no triggers
                continue
            pattern = tuple(
                (t.target_ae, t.condition, t.probability_pct) for t in ae.triggers
            )
            trigger_patterns.append(str(pattern))
        unique_patterns = set(trigger_patterns)
        if len(trigger_patterns) >= 3 and len(unique_patterns) <= 2:
            warnings.append(
                f"Trigger monotonicity: {len(trigger_patterns)} triggered AEs share only "
                f"{len(unique_patterns)} distinct trigger pattern(s) — "
                "triggers should vary by AE type and severity"
            )

    # Auto-correct fabricated severity distributions (must run BEFORE diversity check)
    warnings.extend(_auto_correct_severity(rule_set))

    # Auto-correct fabricated onsets (must run BEFORE plausibility check)
    warnings.extend(_auto_correct_onsets(rule_set))

    # Onset plausibility check (runs after auto-correction)
    warnings.extend(_validate_onset_plausibility(rule_set))

    # Severity diversity check
    warnings.extend(_validate_severity_diversity(rule_set))

    # Regimen check — each drug must have exactly one entry
    regimen_drugs = {r.drug for r in rule_set.regimen}
    for drug in rule_set.drugs:
        if drug not in regimen_drugs:
            warnings.append(f"Drug '{drug}' missing from regimen")
    for rd in regimen_drugs:
        if rd not in set(rule_set.drugs):
            warnings.append(f"Regimen entry for '{rd}' not in drugs list")

    # Checkpoint inhibitor irAE completeness check
    if bundle is not None and _is_checkpoint_inhibitor_from_bundle(bundle):
        _EXPECTED_IRAES = {
            "pneumonitis", "colitis", "hepatitis", "hypothyroidism",
            "hyperthyroidism", "nephritis",
        }
        ae_names_lower_set = {ae.event.lower() for ae in rule_set.adverse_events}
        missing = []
        for expected in sorted(_EXPECTED_IRAES):
            if not any(expected in name for name in ae_names_lower_set):
                missing.append(expected)
        if len(missing) > 2:
            warnings.append(
                f"Checkpoint inhibitor missing irAEs: {', '.join(missing)} "
                f"({len(missing)}/{len(_EXPECTED_IRAES)} expected irAEs absent)"
            )

    if warnings:
        log.warning(f"Validation for {drug_label}: {len(warnings)} items")
        for w in warnings:
            log.warning(f"  - {w}")
    else:
        log.info(f"Validation passed for {drug_label}")

    return warnings


def _cross_check_dailymed(
    rule_set: RuleSet,
    bundle: EvidenceBundle,
    warnings: list[str],
) -> None:
    """Cross-check AE frequencies against DailyMed label data.

    Auto-corrects frequency_pct to label value when deviation > 50%.
    Also rescales severity_distribution proportionally after correction.
    """
    # Build lookup: ae_term_lower -> (label_pct, drug_name, table_type)
    # Prefer clinical table entries over lab/unknown; for same type, keep higher rate
    _TYPE_PRIORITY = {"clinical": 3, "comparison": 2, "lab": 1, "unknown": 0}
    label_aes: dict[str, tuple[float, str]] = {}
    _label_types: dict[str, str] = {}  # track table_type for priority comparison
    for drug, sde in bundle.per_drug.items():
        if not sde.dailymed.found or not sde.dailymed.ae_table:
            continue
        for ae_entry in sde.dailymed.ae_table:
            term_raw = ae_entry.get("term", "").lower().strip()
            term = _normalize_ae_term(term_raw)
            pct = ae_entry.get("incidence_pct")
            if term and pct is not None:
                try:
                    new_pct = float(pct)
                    new_type = ae_entry.get("table_type", "unknown")
                    existing = label_aes.get(term)
                    if existing is None:
                        label_aes[term] = (new_pct, drug)
                        _label_types[term] = new_type
                    else:
                        old_priority = _TYPE_PRIORITY.get(_label_types.get(term, ""), 0)
                        new_priority = _TYPE_PRIORITY.get(new_type, 0)
                        if new_priority > old_priority:
                            label_aes[term] = (new_pct, drug)
                            _label_types[term] = new_type
                        elif new_priority == old_priority and new_pct > existing[0]:
                            # Same table type — keep the higher rate (combo > monotherapy)
                            label_aes[term] = (new_pct, drug)
                except (ValueError, TypeError):
                    pass

    if not label_aes:
        return

    for ae in rule_set.adverse_events:
        ae_normalized = _normalize_ae_term(ae.event)
        if ae_normalized in label_aes:
            label_pct, label_drug = label_aes[ae_normalized]
            if label_pct > 0:
                deviation = abs(ae.frequency_pct - label_pct) / label_pct
                if deviation > 0.5:
                    old_pct = ae.frequency_pct
                    ae.frequency_pct = label_pct
                    # Proportional rescale severity_distribution
                    dist_sum = sum(ae.severity_distribution.values())
                    if dist_sum > 0:
                        scale = label_pct / dist_sum
                        ae.severity_distribution = {
                            k: round(v * scale, 1)
                            for k, v in ae.severity_distribution.items()
                        }
                    warnings.append(
                        f"DailyMed auto-correct: '{ae.event}' frequency_pct "
                        f"{old_pct:.1f}% → {label_pct:.1f}% (from {label_drug} label)"
                    )
        else:
            # AE not found in any label — warn if high frequency
            if ae.frequency_pct > 10.0:
                warnings.append(
                    f"DailyMed cross-check: '{ae.event}' ({ae.frequency_pct:.1f}%) "
                    f"not found in any DailyMed label AE table"
                )


def _cross_check_ctgov(
    rule_set: RuleSet,
    bundle: EvidenceBundle,
    warnings: list[str],
) -> None:
    """Cross-check AE frequencies against ClinicalTrials.gov reported AE data.

    Two modes:
    1. Individual correction: When a rule set AE is significantly underreported
       compared to CT.gov (>2x lower AND >5% absolute difference), correct upward.
    2. Flat-frequency fallback: When ≤3 unique frequency values exist (LLM had
       no frequency data), replace all matching AEs with CT.gov frequencies.
    """
    ct = bundle.clinical_trials
    if not ct.reported_aes:
        return

    # Build lookup: normalized CT.gov term → max pct across all trials
    # Only use AEs from trials with at_risk >= 10 to avoid small-trial
    # inflation (e.g. 3/3 = 100% from a 3-patient trial)
    _MIN_AE_SAMPLE = 10
    ctgov_aes: dict[str, float] = {}
    for ae_entry in ct.reported_aes:
        if ae_entry.get("at_risk", 0) < _MIN_AE_SAMPLE:
            continue
        term = ae_entry.get("term", "").lower().strip()
        pct = ae_entry.get("pct")
        if term and pct is not None:
            try:
                norm = _normalize_ae_term(term)
                val = float(pct)
                ctgov_aes[norm] = max(ctgov_aes.get(norm, 0.0), val)
            except (ValueError, TypeError):
                pass

    if not ctgov_aes:
        return

    # Detect flat-frequency condition: ≤3 unique frequency values
    freq_values = {ae.frequency_pct for ae in rule_set.adverse_events if ae.frequency_pct > 0}
    is_flat = len(freq_values) <= 3

    corrections = 0
    for ae in rule_set.adverse_events:
        ae_norm = _normalize_ae_term(ae.event)
        if ae_norm not in ctgov_aes:
            continue
        ctgov_pct = ctgov_aes[ae_norm]
        if ctgov_pct <= 0:
            continue

        should_correct = False
        if is_flat:
            # Flat-frequency: correct any >1% difference (LLM had no freq data)
            should_correct = abs(ae.frequency_pct - ctgov_pct) > 1.0
        # Individual underreporting correction is DISABLED because
        # _correct_ae_frequencies() in agent_multistage.py already handles
        # evidence-based frequency alignment. The CT.gov max upward correction
        # here was overriding the more nuanced min(DailyMed, CT.gov) logic.

        if should_correct:
            ae.frequency_pct = ctgov_pct
            # Proportional rescale severity_distribution
            dist_sum = sum(ae.severity_distribution.values())
            if dist_sum > 0:
                scale = ctgov_pct / dist_sum
                ae.severity_distribution = {
                    k: round(v * scale, 1)
                    for k, v in ae.severity_distribution.items()
                }
            corrections += 1

    if corrections:
        mode = "flat-frequency fallback" if is_flat else "individual underreporting correction"
        warnings.append(
            f"CT.gov auto-correct: corrected {corrections} AE frequencies "
            f"from ClinicalTrials.gov data ({mode})"
        )


def _inject_missing_boxed_warning_aes(
    rule_set: RuleSet,
    bundle: EvidenceBundle,
    warnings: list[str],
) -> None:
    """Inject missing boxed-warning AEs from OnSIDES into the rule set.

    When a critical boxed-warning AE (e.g. ILD for T-DXd) is absent from
    the rule set, inject it with a conservative default frequency and
    severity distribution based on the AE category.
    """
    from rule_engine.schema import AdverseEvent

    # Collect all boxed-warning AEs across all drugs
    boxed_aes: dict[str, str] = {}  # ae_term → drug_name
    for drug in bundle.drugs:
        sd = bundle.per_drug.get(drug)
        if sd is None:
            continue
        for bw_ae in sd.onsides.boxed_warning_aes:
            boxed_aes[bw_ae] = drug

    if not boxed_aes:
        return

    # Build normalized set of existing AE terms
    rule_ae_terms = {_normalize_ae_term(ae.event) for ae in rule_set.adverse_events}

    # CT.gov data for frequency lookup (keep max across trials)
    ctgov_lookup: dict[str, float] = {}
    for ae_entry in bundle.clinical_trials.reported_aes:
        term = ae_entry.get("term", "").lower().strip()
        pct = ae_entry.get("pct")
        if term and pct is not None:
            try:
                norm = _normalize_ae_term(term)
                val = float(pct)
                ctgov_lookup[norm] = max(ctgov_lookup.get(norm, 0.0), val)
            except (ValueError, TypeError):
                pass

    injected = []
    for bw_ae, drug in sorted(boxed_aes.items()):
        normalized_bw = _normalize_ae_term(bw_ae)
        # Check exact match or substring containment
        if normalized_bw in rule_ae_terms or any(
            normalized_bw in rt or rt in normalized_bw for rt in rule_ae_terms
        ):
            continue

        # Determine frequency: prefer CT.gov, then conservative default
        freq = ctgov_lookup.get(normalized_bw, 5.0)
        category = _classify_ae_category(bw_ae)
        # Category-based severity distribution
        if category == "hematologic":
            sev = {"grade_1": 0.25, "grade_2": 0.30, "grade_3": 0.30, "grade_4": 0.15}
        elif category == "pulmonary":
            sev = {"grade_1": 0.20, "grade_2": 0.50, "grade_3": 0.20, "grade_4": 0.05, "grade_5": 0.05}
        elif category == "cardiac":
            sev = {"grade_1": 0.15, "grade_2": 0.35, "grade_3": 0.30, "grade_4": 0.15, "grade_5": 0.05}
        elif category == "hepatic":
            sev = {"grade_1": 0.40, "grade_2": 0.30, "grade_3": 0.20, "grade_4": 0.10}
        else:
            sev = {"grade_1": 0.50, "grade_2": 0.30, "grade_3": 0.15, "grade_4": 0.05}

        # Scale severity to frequency
        scaled_sev = {k: round(v * freq, 1) for k, v in sev.items() if v * freq >= 0.1}
        # Ensure sum matches frequency
        sev_sum = sum(scaled_sev.values())
        if sev_sum > 0 and abs(sev_sum - freq) > 0.5:
            adj_key = max(scaled_sev, key=scaled_sev.get)
            scaled_sev[adj_key] = round(scaled_sev[adj_key] + (freq - sev_sum), 1)

        # Determine onset and reversibility by category
        onset_map = {
            "hematologic": 14, "gi": 5, "hepatic": 21, "pulmonary": 60,
            "cardiac": 30, "dermatologic": 14, "constitutional": 10,
        }
        reversible_map = {
            "hematologic": True, "gi": True, "hepatic": True, "pulmonary": True,
            "cardiac": False, "dermatologic": True, "constitutional": True,
        }

        new_ae = AdverseEvent(
            event=bw_ae,
            frequency_pct=freq,
            severity_distribution=scaled_sev,
            median_onset_days=onset_map.get(category, 14),
            reversible=reversible_map.get(category, True),
            source_drug=drug,
            triggers=[],
        )
        rule_set.adverse_events.append(new_ae)
        rule_ae_terms.add(normalized_bw)
        injected.append(bw_ae)

    if injected:
        warnings.append(
            f"Injected {len(injected)} missing boxed-warning AE(s): "
            f"{', '.join(injected)}"
        )


def _cross_check_onsides(
    rule_set: RuleSet,
    bundle: EvidenceBundle,
    warnings: list[str],
) -> None:
    """Cross-check rule set AEs against OnSIDES boxed-warning AEs.

    Warns if a boxed-warning AE from OnSIDES is absent from the rule set,
    since boxed warnings represent the most critical safety signals.
    """
    # Collect all boxed-warning AEs across all drugs
    boxed_aes: set[str] = set()
    for drug in bundle.drugs:
        sd = bundle.per_drug.get(drug)
        if sd is None:
            continue
        for bw_ae in sd.onsides.boxed_warning_aes:
            boxed_aes.add(bw_ae)

    if not boxed_aes:
        return

    # Build normalized set of rule set AE terms
    rule_ae_terms = {_normalize_ae_term(ae.event) for ae in rule_set.adverse_events}

    missing = []
    for bw_ae in sorted(boxed_aes):
        normalized_bw = _normalize_ae_term(bw_ae)
        # Check exact match or substring containment (e.g. "hepatotoxicity" in "drug-induced hepatotoxicity")
        if normalized_bw not in rule_ae_terms and not any(
            normalized_bw in rt or rt in normalized_bw for rt in rule_ae_terms
        ):
            missing.append(bw_ae)

    if missing:
        warnings.append(
            f"OnSIDES cross-check: {len(missing)} boxed-warning AE(s) missing from rule set: "
            f"{', '.join(missing[:10])}"
        )


def validate_multi_rule_set(
    rule_set: RuleSet,
    individual_results: list[tuple[RuleSet, list[str]]],
    ddi_evidence: object | None = None,
) -> list[str]:
    """Run multi-indication-specific validation checks on a merged rule set.

    Runs the standard validate_rule_set() first (without evidence bundle),
    then adds multi-drug-specific checks.

    Args:
        rule_set: The merged multi-indication rule set.
        individual_results: List of (individual_rule_set, individual_warnings_list) tuples.
        ddi_evidence: Optional DDIEvidence object from evidence/ddi.py.

    Returns:
        List of warning strings.
    """
    # 1. Run standard single-indication validation (no bundle)
    warnings = validate_rule_set(rule_set)

    # 2. Check all drugs from all individual results are present in unified drugs list
    unified_drugs_lower = {d.lower() for d in rule_set.drugs}
    for individual_rs, _ in individual_results:
        for drug in individual_rs.drugs:
            if drug.lower() not in unified_drugs_lower:
                warnings.append(
                    f"Multi-indication: drug '{drug}' from individual rule set "
                    f"({individual_rs.indication}) missing from unified drugs list"
                )

    # 3. Check all drugs have regimen entries
    regimen_drugs = {r.drug for r in rule_set.regimen}
    for drug in rule_set.drugs:
        if drug not in regimen_drugs:
            warnings.append(
                f"Multi-indication: drug '{drug}' missing from regimen"
            )

    # 4. Check per_indication_efficacy is populated for each indication
    n_indications = len(individual_results)
    if len(rule_set.per_indication_efficacy) < n_indications:
        indications = [rs.indication for rs, _ in individual_results]
        warnings.append(
            f"Multi-indication: per_indication_efficacy has "
            f"{len(rule_set.per_indication_efficacy)} entries but expected "
            f"{n_indications} (indications: {', '.join(indications)})"
        )

    # 5. Overlapping AE frequency sanity — detect naive summation
    # Build dict: ae_name_lower -> list of individual frequencies
    individual_ae_freqs: dict[str, list[float]] = {}
    for individual_rs, _ in individual_results:
        for ae in individual_rs.adverse_events:
            key = ae.event.lower().strip()
            individual_ae_freqs.setdefault(key, []).append(ae.frequency_pct)

    for ae in rule_set.adverse_events:
        key = ae.event.lower().strip()
        freqs = individual_ae_freqs.get(key, [])
        if len(freqs) < 2:
            continue
        naive_sum = sum(freqs)
        merged_freq = ae.frequency_pct
        if abs(naive_sum - merged_freq) <= 2.0:
            warnings.append(
                f"Multi-indication: AE '{ae.event}' merged frequency "
                f"({merged_freq:.1f}%) ≈ naive sum of individual frequencies "
                f"({naive_sum:.1f}%) — should use probabilistic independence model "
                f"P = 1-(1-Pa)(1-Pb) instead of naive addition"
            )

    # 6. Check drug_interactions is populated when DDI evidence exists
    if (
        ddi_evidence is not None
        and hasattr(ddi_evidence, "pairs")
        and len(ddi_evidence.pairs) > 0
        and len(rule_set.drug_interactions) == 0
    ):
        warnings.append(
            f"Multi-indication: DDI evidence contains {len(ddi_evidence.pairs)} "
            f"interaction pair(s) but drug_interactions is empty"
        )

    # 7. Check AE count doesn't exceed sum of individual AE counts (fabrication signal)
    individual_ae_total = sum(
        len(rs.adverse_events) for rs, _ in individual_results
    )
    merged_ae_count = len(rule_set.adverse_events)
    if merged_ae_count > individual_ae_total:
        warnings.append(
            f"Multi-indication: merged AE count ({merged_ae_count}) exceeds sum "
            f"of individual AE counts ({individual_ae_total}) — possible fabrication"
        )

    # 8. Check is_multi_indication is True
    if not rule_set.is_multi_indication:
        warnings.append(
            "Multi-indication: is_multi_indication is False — should be True "
            "for a merged multi-indication rule set"
        )

    if warnings:
        log.warning(
            "Multi-indication validation: %d warnings", len(warnings)
        )
        for w in warnings:
            log.warning("  - %s", w)
    else:
        log.info("Multi-indication validation passed")

    return warnings
