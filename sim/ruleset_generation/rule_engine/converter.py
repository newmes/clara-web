"""Convert validated RuleSet JSON to target simulation schema.

Pure dict-to-dict transform. Takes a loaded JSON dict (current RuleSet format)
and returns a dict matching the target base.json schema — with distribution
wrappers, 0-1 probability scale, and additional template sections.
"""

from __future__ import annotations

import copy
import re
from typing import Any

# Route-specific fields extracted into type overlay
_IV_OVERLAY_FIELDS = {"infusion_duration_minutes"}
_SC_OVERLAY_FIELDS = {"injection_volume_ml"}
_ORAL_OVERLAY_FIELDS = {"daily_dosing_schedule", "continuous_days_per_cycle"}

from rule_engine.reference_data import (
    AE_DURATION_DEFAULTS,
    AE_DURATION_FALLBACK,
    DEFAULT_AE_CASCADE_RULES,
    DEFAULT_DOSE_MOD_RULE,
    DISEASE_BASELINES,
    DISPOSITION_DEFAULTS,
    ECOG_MODEL_DEFAULTS,
    ECOG_PS_DEFAULTS,
    LAB_REFERENCE_RANGES,
    MORTALITY_MODELS,
    ROUTE_MAP,
    SUPPORTIVE_CARE_MAP,
    get_default_baseline,
    get_default_mortality,
)


def _to_snake_case(name: str) -> str:
    """Convert a name to lowercase snake_case for GT format alignment.

    'Neutropenia' -> 'neutropenia'
    'Chronic Kidney Disease' -> 'chronic_kidney_disease'
    'Non-Cardiac Chest Pain' -> 'non-cardiac_chest_pain'
    """
    return name.strip().lower().replace(" ", "_")


def detect_schema_type(data: dict) -> str:
    """Detect which schema type applies based on regimen routes and drug count."""
    regimen = data.get("regimen", [])
    routes: set[str] = set()
    for entry in regimen:
        route = _normalize_route(entry.get("route", "IV"))
        routes.add(route)

    # Use drugs list as ground truth for mono vs combo — regimen entries
    # may have multiple entries per drug (different doses) or omit drugs.
    drugs = data.get("drugs", [])
    is_combo = len(drugs) >= 2 if drugs else len(regimen) >= 2
    has_oral = "ORAL" in routes
    has_iv = "INTRAVENOUS" in routes
    has_sc = "SUBCUTANEOUS" in routes

    if is_combo:
        if has_oral and has_iv:
            return "oral_iv_combination"
        elif has_iv:
            return "iv_combination"
        else:
            return "oral_iv_combination"  # fallback for mixed
    else:
        if has_sc:
            return "subcutaneous_monotherapy"
        elif has_oral:
            return "oral_monotherapy"
        else:
            return "iv_monotherapy"


def convert_ruleset(data: dict) -> tuple[dict, str]:
    """Convert a current-format rule set dict to the target schema dict.

    Returns (converted_dict, schema_type_name).
    """
    drugs = data.get("drugs", [])
    indication = data.get("indication", "")
    regimen = data.get("regimen", [])
    demographics = data.get("demographics", {})
    comorbidities = data.get("comorbidities", [])
    adverse_events = data.get("adverse_events", [])
    efficacy = data.get("efficacy", {})

    schema_type = detect_schema_type(data)
    cycle_length = _extract_cycle_length(regimen)
    ae_profile = _convert_ae_profile(adverse_events)

    result = {
        "drug_name": " + ".join(drugs),
        "indication": indication,
        "trial_design": {"cycle_length_days": cycle_length},
        "demographics": _convert_demographics(demographics),
        "comorbidities": _convert_comorbidities(comorbidities),
        "disease_baseline": _get_disease_baseline(indication, efficacy),
        "ae_profile": ae_profile,
        "efficacy": _convert_efficacy(efficacy),
        "lab_reference_ranges": copy.deepcopy(LAB_REFERENCE_RANGES),
        "administration_schedule": _build_admin_schedule(regimen, schema_type),
        "dose_modification_rules": _build_dose_mod_rules(adverse_events),
        "supportive_care_rules": _get_supportive_care_rules(ae_profile),
        "mortality_model": _get_mortality_model(indication),
        "ecog_model": copy.deepcopy(ECOG_MODEL_DEFAULTS),
        "ae_cascade_rules": _build_ae_cascade_rules(adverse_events),
        "disposition_model": copy.deepcopy(DISPOSITION_DEFAULTS),
        "_schema_type": schema_type,
    }
    return result, schema_type


def split_base_overlay(converted: dict, schema_type: str) -> tuple[dict, dict]:
    """Split a converted rule set into base dict + type-specific overlay dict.

    base.json: everything except route-specific fields in administration_schedule.
    {schema_type}.json: only route-specific fields keyed by drug_name.
    """
    base = copy.deepcopy(converted)
    overlay: dict = {}

    # Determine which fields belong in the overlay based on schema type
    if "iv" in schema_type:
        extract_fields = {"infusion_duration_minutes"}
    elif "subcutaneous" in schema_type:
        extract_fields = {"injection_volume_ml"}
    elif "oral" in schema_type:
        extract_fields = {"daily_dosing_schedule", "continuous_days_per_cycle"}
    else:
        extract_fields = set()

    # Extract route-specific fields from administration_schedule
    if extract_fields:
        overlay_schedule = []
        for item in base.get("administration_schedule", []):
            overlay_item = {"drug_name": item["drug_name"]}
            for field in extract_fields:
                if field in item:
                    overlay_item[field] = item.pop(field)
            overlay_schedule.append(overlay_item)
        overlay["administration_schedule"] = overlay_schedule

    # SC: extract injection_site_specific from ae_profile
    if "subcutaneous" in schema_type:
        overlay_aes = []
        for ae in base.get("ae_profile", []):
            is_injection = "injection_site" in ae.get("ae_term", "")
            overlay_aes.append({
                "ae_term": ae["ae_term"],
                "injection_site_specific": is_injection,
            })
        overlay["ae_profile"] = overlay_aes

    # Remove metadata from base (GT doesn't have it)
    base.pop("_schema_type", None)

    return base, overlay


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def _make_numeric_dist(
    mean: float,
    std: float,
    min_val: float,
    max_val: float,
    dist: str = "normal",
) -> dict:
    return {
        "type": "numeric",
        "distribution": dist,
        "params": {"mean": mean, "std": std, "min": min_val, "max": max_val},
    }


def _make_categorical_dist(options: dict[str, float]) -> dict:
    return {"type": "categorical", "options": options}


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------

def _convert_demographics(demo: dict) -> dict:
    age = demo.get("age") or {}
    sex = demo.get("sex") or {}
    race = demo.get("race_ethnicity") or []

    result: dict[str, Any] = {}

    # Age — guard against None sub-fields
    result["age"] = _make_numeric_dist(
        mean=age.get("mean") or 60,
        std=age.get("std") or 12,
        min_val=age.get("min") or 18,
        max_val=age.get("max") or 90,
    )

    # Sex — scale /100, guard against None
    pct_male = sex.get("pct_male") or 50
    pct_female = sex.get("pct_female") or 50
    result["sex"] = _make_categorical_dist({
        "Male": round(pct_male / 100, 4),
        "Female": round(pct_female / 100, 4),
    })

    # Race — scale /100
    if race:
        race_opts = {}
        for entry in race:
            group = entry.get("group", "Other")
            pct = entry.get("pct", 0)
            race_opts[group] = round(pct / 100, 4)
        result["race"] = _make_categorical_dist(race_opts)
    else:
        result["race"] = _make_categorical_dist({"Unknown": 1.0})

    # ECOG PS — use LLM-extracted data if available, else template default
    ecog_data = demo.get("ecog_ps")
    if ecog_data and isinstance(ecog_data, dict):
        ecog_opts = {}
        total = sum(float(v) for v in ecog_data.values() if v is not None)
        if total > 0:
            for k, v in ecog_data.items():
                if v is not None:
                    ecog_opts[str(k)] = round(float(v) / total, 4) if total > 1.5 else round(float(v), 4)
            result["ecog_ps"] = _make_categorical_dist(ecog_opts)
        else:
            result["ecog_ps"] = _make_categorical_dist(copy.deepcopy(ECOG_PS_DEFAULTS))
    else:
        result["ecog_ps"] = _make_categorical_dist(copy.deepcopy(ECOG_PS_DEFAULTS))

    return result


# ---------------------------------------------------------------------------
# Adverse events
# ---------------------------------------------------------------------------

def _get_ae_duration(event_name: str) -> dict:
    """Look up duration distribution for an AE by keyword match."""
    name_lower = event_name.lower()
    for keyword, params in AE_DURATION_DEFAULTS.items():
        if keyword in name_lower:
            return _make_numeric_dist(
                mean=params["mean"],
                std=params["std"],
                min_val=params["min"],
                max_val=params["max"],
                dist="lognormal",
            )
    return _make_numeric_dist(
        mean=AE_DURATION_FALLBACK["mean"],
        std=AE_DURATION_FALLBACK["std"],
        min_val=AE_DURATION_FALLBACK["min"],
        max_val=AE_DURATION_FALLBACK["max"],
        dist="lognormal",
    )


def _convert_single_ae(ae: dict) -> dict:
    """Convert one AE entry from current to target format."""
    event = ae.get("event") or "Unknown"
    freq_pct = ae.get("frequency_pct") or 1.0
    sev = ae.get("severity_distribution") or {}
    onset = ae.get("median_onset_days") or 14
    reversible = ae.get("reversible") if ae.get("reversible") is not None else True

    # Incidence: /100
    incidence = round(freq_pct / 100, 4)

    # Grade distribution: normalize to proportions summing to ~1.0
    grade_dist = _normalize_grade_distribution(sev, freq_pct)

    # Onset distribution
    onset_std = max(onset * 0.5, 2)
    onset_min = max(1, int(onset * 0.2))
    onset_max = max(int(onset * 3), onset + 7)
    onset_dist = _make_numeric_dist(
        mean=onset,
        std=round(onset_std, 1),
        min_val=onset_min,
        max_val=onset_max,
        dist="lognormal",
    )

    ae_term = _to_snake_case(event)

    # Cumulative: AEs that worsen with repeated exposure
    _CUMULATIVE_KEYWORDS = (
        "alopecia", "neuropath", "paraesthesia", "ototoxic", "tinnitus",
        "deafness", "hearing", "nephrotox", "creatinine", "cardiomyop",
        "pulmonary_fibrosis",
    )
    cumulative = any(kw in ae_term for kw in _CUMULATIVE_KEYWORDS)

    return {
        "ae_term": ae_term,
        "incidence_all_grade": incidence,
        "grade_distribution": grade_dist,
        "onset_day": onset_dist,
        "duration_days": _get_ae_duration(event),
        "risk_modifiers": [],
        "cumulative": cumulative,
        "reversible": reversible,
    }


def _normalize_grade_distribution(sev: dict, freq_pct: float) -> dict[str, float]:
    """Normalize severity grades to proportions summing to ~1.0.

    Current format: grade_1=7.8, grade_2=4.5 (absolute %s out of freq_pct=13.0)
    Target format:  "1": 0.60, "2": 0.346 (proportions of total incidence)
    """
    grades = {}
    total = 0.0
    for key in ["grade_1", "grade_2", "grade_3", "grade_4"]:
        val = sev.get(key, 0)
        if val and val > 0:
            total += val
            grade_num = key.split("_")[1]
            grades[grade_num] = val

    # Normalize: divide each by total (not by freq_pct) to sum to 1.0
    # If total is 0 (all grades zero/missing), use a conservative default
    if total <= 0 or not grades:
        result = {"1": 0.30, "2": 0.30, "3": 0.25, "4": 0.10, "5": 0.05}
        return result

    denom = total
    result = {}
    for g, v in grades.items():
        proportion = round(v / denom, 4)
        if proportion > 0:
            result[g] = proportion

    return result


def _convert_ae_profile(adverse_events: list[dict]) -> list[dict]:
    """Convert all AEs to target format."""
    return [_convert_single_ae(ae) for ae in adverse_events]


# ---------------------------------------------------------------------------
# Dose modification rules — from AE triggers
# ---------------------------------------------------------------------------

def _parse_grade_from_condition(condition: str) -> int | None:
    """Extract grade threshold from trigger condition string.

    Examples: "grade >= 3", "grade >= 2 and incidence > 3%"
    """
    m = re.search(r"grade\s*>=?\s*(\d+)", condition, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _build_dose_mod_rules(adverse_events: list[dict]) -> list[dict]:
    """Extract dose modification rules from AE triggers.

    Groups triggers by AE name, builds grade_actions mapping.
    """
    rules_by_ae: dict[str, dict] = {}

    for ae in adverse_events:
        event = ae.get("event", "")
        triggers = ae.get("triggers", [])
        if not triggers:
            continue

        for trigger in triggers:
            target = trigger.get("target_ae", "")
            if target not in ("Dose reduction", "Treatment discontinuation"):
                continue

            condition = trigger.get("condition", "")
            grade = _parse_grade_from_condition(condition)
            if grade is None:
                continue

            snake_event = _to_snake_case(event)
            if snake_event not in rules_by_ae:
                rules_by_ae[snake_event] = {
                    "ae_term": snake_event,
                    "grade_actions": {},
                    "dose_reduction_levels": [1.0, 0.75, 0.50],
                    "rechallenge_criteria": "If toxicity resolves to Grade 1 or less, may resume at next lower dose level.",
                }

            grade_str = str(grade)
            if target == "Dose reduction":
                if grade_str not in rules_by_ae[snake_event]["grade_actions"]:
                    rules_by_ae[snake_event]["grade_actions"][grade_str] = "DRUG INTERRUPTED"
            elif target == "Treatment discontinuation":
                rules_by_ae[snake_event]["grade_actions"][grade_str] = "DRUG WITHDRAWN"

    result = list(rules_by_ae.values())

    has_default = any(r.get("ae_term") == "default" for r in result)
    if not has_default:
        result.insert(0, copy.deepcopy(DEFAULT_DOSE_MOD_RULE))

    return result


# ---------------------------------------------------------------------------
# AE cascade rules — from triggers referencing other AEs
# ---------------------------------------------------------------------------

def _build_ae_cascade_rules(adverse_events: list[dict]) -> list[dict]:
    """Extract AE cascade rules from triggers where target_ae is another AE.

    Falls back to common oncology cascades when LLM data yields none.
    """
    cascades = []
    for ae in adverse_events:
        event = ae.get("event", "")
        triggers = ae.get("triggers", [])
        for trigger in triggers:
            target = trigger.get("target_ae", "")
            if target in ("Dose reduction", "Treatment discontinuation", ""):
                continue
            condition = trigger.get("condition", "")
            grade = _parse_grade_from_condition(condition)
            prob = trigger.get("probability_pct", 50)
            cascades.append({
                "trigger_ae": _to_snake_case(event),
                "grade_threshold": grade if grade else 3,
                "target_ae": _to_snake_case(target),
                "multiplier": round(prob / 20, 2),
            })

    if not cascades:
        return copy.deepcopy(DEFAULT_AE_CASCADE_RULES)
    return cascades


# ---------------------------------------------------------------------------
# Administration schedule
# ---------------------------------------------------------------------------

def _normalize_route(route_str: str) -> str:
    """Normalize route string to target schema enum value."""
    cleaned = route_str.strip().lower()
    # Handle multi-route (e.g., "intravenously or subcutaneously") — take first
    if " or " in cleaned:
        cleaned = cleaned.split(" or ")[0].strip()
    # Remove adverbial suffixes
    cleaned = cleaned.replace("intravenously", "intravenous").replace("subcutaneously", "subcutaneous")
    return ROUTE_MAP.get(cleaned, "INTRAVENOUS")


def _parse_dose_value(dose_str: str) -> tuple[float | None, str]:
    """Extract numeric dose value and unit from dose string.

    Returns (value, unit) or (None, original_string).
    """
    m = re.match(r"([\d.]+)\s*(.*)", dose_str.strip())
    if m:
        try:
            return float(m.group(1)), m.group(2).strip()
        except ValueError:
            pass
    return None, dose_str


def _parse_cycle_days_schedule(schedule: str, cycle_days: int) -> list[int]:
    """Parse schedule string into list of administration days.

    Examples:
        "Day 1" -> [1]
        "Day 1, 8" -> [1, 8]
        "Day 1, 8, 15" -> [1, 8, 15]
        "Every 3 weeks" -> [1]
        "weekly" -> [1]  (within one cycle)
        "Days 1-5" -> [1, 2, 3, 4, 5]
        "Day 1-14" -> [1, 2, ..., 14]
    """
    if not schedule:
        return [1]

    s = schedule.strip()

    # "Day X, Y, Z" or "Days X, Y, Z"
    m = re.findall(r"(\d+)", s)
    if "day" in s.lower() and m:
        # Check for range pattern "Day X-Y" or "Days X-Y"
        range_match = re.search(r"(\d+)\s*[-–]\s*(\d+)", s)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            return list(range(start, end + 1))
        return [int(x) for x in m]

    # Generic fallback
    return [1]


def _infer_dosing_schedule(schedule_str: str) -> str:
    """Infer QD/BID/TID from schedule string."""
    s = schedule_str.lower()
    if "twice" in s or "bid" in s or "two times" in s:
        return "BID"
    if "three times" in s or "tid" in s:
        return "TID"
    return "QD"  # default: once daily


def _infer_continuous_days(schedule_str: str, cycle_days: int) -> int | None:
    """Extract continuous days per cycle for intermittent oral dosing.

    Returns number of dosing days if less than cycle length (intermittent),
    or None if continuous (every day of cycle).
    """
    # Match patterns like "Day 1-14", "Days 1-14", "D1-14"
    m = re.search(r"[Dd](?:ays?)?\s*(\d+)\s*[-–]\s*(\d+)", schedule_str)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        days = end - start + 1
        if days < cycle_days:
            return days
    # Match "14 days" pattern (e.g., "14 days on, 7 days off")
    m = re.search(r"(\d+)\s*days?\s*(?:on|continuous)", schedule_str.lower())
    if m:
        days = int(m.group(1))
        if days < cycle_days:
            return days
    return None  # continuous (every day of cycle)


def _parse_infusion_duration(schedule_str: str) -> int:
    """Parse infusion duration from schedule string.

    Looks for patterns like "over 120 minutes", "over 30 min", "60-min infusion".
    Returns duration in minutes, or 30 as default.
    """
    if not schedule_str:
        return 30
    m = re.search(r"over\s+(\d+)\s*min", schedule_str, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*[-–]?\s*min(?:ute)?s?\s*infusion", schedule_str, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"infusion\s+(?:over\s+)?(\d+)\s*min", schedule_str, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 30


def _build_admin_schedule(regimen: list[dict], schema_type: str = "") -> list[dict]:
    """Build administration_schedule from regimen entries.

    Adds type-specific fields based on schema_type:
    - IV items: infusion_duration_minutes (required)
    - ORAL items: daily_dosing_schedule, continuous_days_per_cycle
    - SC items: no extra fields
    """
    schedule = []
    for entry in regimen:
        drug = entry.get("drug", "")
        dose_str = entry.get("dose", "")
        route_str = entry.get("route", "IV")
        cycle_days = entry.get("cycle_days", 21)
        sched_str = entry.get("schedule", "")

        route = _normalize_route(route_str)
        dose_val, dose_unit = _parse_dose_value(dose_str)
        admin_days = _parse_cycle_days_schedule(sched_str, cycle_days)

        item: dict[str, Any] = {
            "drug_name": drug,
            "dose_per_administration": dose_str,
            "route": route,
            "cycle_days": admin_days,
        }
        if dose_val is not None:
            item["dose_value"] = dose_val
            item["dose_unit"] = dose_unit

        # Route-specific fields
        if route == "INTRAVENOUS":
            item["infusion_duration_minutes"] = _parse_infusion_duration(sched_str)
        elif route == "ORAL":
            item["daily_dosing_schedule"] = _infer_dosing_schedule(sched_str)
            cont_days = _infer_continuous_days(sched_str, cycle_days)
            item["continuous_days_per_cycle"] = cont_days

        schedule.append(item)

    # For monotherapy schemas, deduplicate same-drug entries (LLM sometimes
    # emits multiple regimen rows for different dose levels of one drug).
    if schema_type.endswith("_monotherapy") and len(schedule) > 1:
        seen: set[str] = set()
        deduped = []
        for item in schedule:
            name = item.get("drug_name", "")
            if name not in seen:
                seen.add(name)
                deduped.append(item)
        schedule = deduped

    return schedule


# ---------------------------------------------------------------------------
# Cycle length extraction
# ---------------------------------------------------------------------------

def _extract_cycle_length(regimen: list[dict]) -> int:
    """Extract cycle_length_days from first regimen entry."""
    if regimen:
        return regimen[0].get("cycle_days", 21)
    return 21


# ---------------------------------------------------------------------------
# Comorbidities
# ---------------------------------------------------------------------------

def _convert_comorbidities(comorbidities: list[dict]) -> list[dict]:
    """Convert comorbidities to target format: scale to 0-1, restructure modifiers."""
    result = []
    for comorb in comorbidities:
        condition = comorb.get("condition") or ""
        prev_pct = comorb.get("prevalence_pct") or 0
        modifiers = comorb.get("ae_risk_modifiers") or []

        entry: dict[str, Any] = {
            "condition": _to_snake_case(condition),
            "base_probability": round(prev_pct / 100, 4),
        }

        if modifiers:
            entry["conditional_modifiers"] = [
                {
                    "if_condition": m.get("ae", ""),
                    "multiplier": m.get("risk_multiplier", 1.0),
                }
                for m in modifiers
            ]

        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Efficacy
# ---------------------------------------------------------------------------

def _convert_efficacy(efficacy: dict) -> dict:
    """Convert efficacy to target format: scale to 0-1, wrap survival in distributions.

    Uses LLM-provided 95% CI bounds to derive distribution parameters.
    Omits fields when data is absent rather than substituting 0.
    """
    orr = efficacy.get("overall_response_rate_pct")
    cr = efficacy.get("complete_response_rate_pct")
    pfs = efficacy.get("median_pfs_months")
    pfs_ci_low = efficacy.get("median_pfs_ci_low")
    pfs_ci_high = efficacy.get("median_pfs_ci_high")
    os_ = efficacy.get("median_os_months")
    os_ci_low = efficacy.get("median_os_ci_low")
    os_ci_high = efficacy.get("median_os_ci_high")

    result: dict[str, Any] = {}

    if orr is not None and orr > 0:
        result["overall_response_rate"] = round(orr / 100, 4)
    else:
        # Schema requires overall_response_rate — use 0 when no data available
        result["overall_response_rate"] = 0.0
    if cr is not None and cr > 0:
        result["complete_response_rate"] = round(cr / 100, 4)

    if pfs is not None and pfs > 0:
        pfs_dist = _survival_dist(pfs, pfs_ci_low, pfs_ci_high)
        if pfs_dist is not None:
            result["progression_free_survival_months"] = pfs_dist
        else:
            # No CI — store median only (no manufactured distribution)
            result["median_pfs_months"] = pfs
    if os_ is not None and os_ > 0:
        os_dist = _survival_dist(os_, os_ci_low, os_ci_high)
        if os_dist is not None:
            result["overall_survival_months"] = os_dist
        else:
            result["median_os_months"] = os_

    return result


def _survival_dist(
    median: float, ci_low: float | None, ci_high: float | None,
) -> dict | None:
    """Build an exponential survival distribution from median and 95% CI.

    For exponential survival data, std is proportional to the mean (not
    derived from CI width, which represents estimate uncertainty, not
    patient-level variance). Empirically std ≈ 0.65 × median in censored
    trial data. min/max span the plausible individual outcome range.

    Returns None when median is missing or non-positive.
    """
    if not median or median <= 0:
        return None

    std = round(median * 0.65, 2)
    min_val = 0.03  # earliest plausible event
    max_val = round(median * 3.5, 1)  # ~97.5th percentile tail
    return _make_numeric_dist(
        mean=median,
        std=std,
        min_val=min_val,
        max_val=max_val,
        dist="exponential",
    )


# ---------------------------------------------------------------------------
# Disease baseline
# ---------------------------------------------------------------------------

def _get_disease_baseline(indication: str, efficacy: dict | None = None) -> dict:
    """Look up disease baseline by indication keyword.

    When efficacy ORR/CR are available, derive tumor_response_distribution
    from them for internal consistency with the efficacy section.
    """
    ind_lower = indication.lower()
    baseline = None
    for keyword, bl in DISEASE_BASELINES.items():
        if keyword in ind_lower:
            baseline = copy.deepcopy(bl)
            break
    if baseline is None:
        baseline = get_default_baseline()

    # Override tumor_response_distribution from efficacy ORR/CR when available
    if efficacy:
        orr_pct = efficacy.get("overall_response_rate_pct")
        cr_pct = efficacy.get("complete_response_rate_pct")
        if orr_pct is not None and orr_pct > 0:
            orr = orr_pct / 100.0  # convert to proportion
            cr = (cr_pct / 100.0) if cr_pct and cr_pct > 0 else orr * 0.1
            pr = orr - cr
            remainder = 1.0 - orr
            # Split remainder: ~60% SD, ~40% PD for typical oncology
            sd = round(remainder * 0.6, 4)
            pd = round(remainder * 0.4, 4)
            baseline["tumor_response_distribution"] = {
                "CR": round(cr, 4),
                "PR": round(pr, 4),
                "SD": sd,
                "PD": pd,
            }

    return baseline


# ---------------------------------------------------------------------------
# Supportive care
# ---------------------------------------------------------------------------

def _get_supportive_care_rules(ae_profile: list[dict]) -> list[dict]:
    """Generate supportive care rules based on AE profile entries."""
    rules = []
    seen = set()
    for ae in ae_profile:
        term = ae.get("ae_term", "")
        # ae_term is already snake_case from _convert_single_ae
        for keyword, treatments in SUPPORTIVE_CARE_MAP.items():
            if keyword in term and term not in seen:
                rules.append({
                    "ae_term": term,
                    "treatments": copy.deepcopy(treatments),
                })
                seen.add(term)
                break
    return rules


# ---------------------------------------------------------------------------
# Mortality model
# ---------------------------------------------------------------------------

def _get_mortality_model(indication: str) -> dict:
    """Look up mortality model by indication keyword."""
    ind_lower = indication.lower()
    for keyword, model in MORTALITY_MODELS.items():
        if keyword in ind_lower:
            return copy.deepcopy(model)
    return get_default_mortality()
