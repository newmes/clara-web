"""Hospital Record Mapper

Extracts hospital-only (HR) view from a full ground-truth day record.
Used to create separate _hospital.jsonl files during simulation.

The hospital record only contains data that would be available to the
treating institution: observed labs/vitals, detected AEs, and documented
subjective status — all constrained by observation timing.
"""

from __future__ import annotations

import json
from typing import Any


def extract_hospital_record(cdash_record: dict) -> dict:
    """Convert a full GT day record into hospital-only format.

    Args:
        cdash_record: Full CDASH-mapped day record (from map_day_record)

    Returns:
        Hospital-only record with the same day/patient/cycle structure
        but only hospital-observable data.
    """
    hr_data = cdash_record.get("hospital_record", {})
    hr_obj = hr_data.get("objective", {}) if hr_data else {}
    hr_subj = hr_data.get("subjective") if hr_data else None
    obs_events = cdash_record.get("observation_events", [])
    obs_types = hr_data.get("observation_types", []) if hr_data else []

    # Base record
    record = {
        "patient_id": cdash_record.get("patient_id"),
        "day": cdash_record.get("day"),
        "cycle": cdash_record.get("cycle"),
        "cycle_day": cdash_record.get("cycle_day"),
        "is_observation_day": bool(obs_types),
        "observation_types": obs_types,
    }

    # Objective (hospital-observed)
    objective = {}

    # Location + treatment status (always known)
    gt_obj = cdash_record.get("objective", {})
    objective["location"] = hr_obj.get("location", gt_obj.get("location"))
    objective["treatment_status"] = hr_obj.get(
        "treatment_status", gt_obj.get("treatment_status"))

    # Drug info (always known — hospital administers drugs)
    for key, val in gt_obj.items():
        if isinstance(val, dict) and "cumulative_dose_mg" in val:
            objective[key] = val

    # Labs: from HR (may be stale)
    objective["labs"] = hr_obj.get("labs", {})
    objective["labs_stale_days"] = hr_obj.get("labs_stale_days", 0)

    # Vitals: from HR (may be stale)
    objective["vitals"] = hr_obj.get("vitals", {})
    objective["vitals_stale_days"] = hr_obj.get("vitals_stale_days", 0)

    # ECOG: from HR (may be stale)
    objective["ecog"] = hr_obj.get("ecog")

    # Tumor: from HR (only updated on RECIST scans)
    objective["tumor"] = hr_obj.get("tumor")

    # Active AEs: only detected ones
    objective["active_aes"] = hr_obj.get("active_aes", [])

    record["objective"] = objective

    # AE domain (hospital-detected only)
    hr_ae_terms = {ae.get("ae", "") for ae in objective["active_aes"]}
    gt_aes = cdash_record.get("AE", [])
    record["AE"] = [
        ae for ae in gt_aes
        if ae.get("AETERM", "") in hr_ae_terms
    ]

    # EC (exposure): always known (hospital administers)
    record["EC"] = cdash_record.get("EC", [])

    # CM (concomitant meds): always known (hospital prescribes)
    record["CM"] = cdash_record.get("CM", [])

    # VS (vitals): only if observation day, otherwise stale/None
    if obs_types:
        record["VS"] = cdash_record.get("VS", {})
    else:
        # Return last known vitals from HR
        record["VS"] = _vitals_from_hr(hr_obj.get("vitals", {}))

    # LB (labs): only if observation day, otherwise stale/None
    if obs_types:
        record["LB"] = cdash_record.get("LB", {})
    else:
        record["LB"] = _labs_from_hr(hr_obj.get("labs", {}))

    # DS (disposition): always known
    record["DS"] = cdash_record.get("DS")

    # RS, TU: only if scheduled scan
    if "scheduled_scan" in obs_types:
        record["RS"] = cdash_record.get("RS")
        record["TU"] = cdash_record.get("TU")
    else:
        record["RS"] = None
        record["TU"] = None

    # DD (death details): always known
    record["DD"] = cdash_record.get("DD")

    # PE, EG: only on observation days
    if obs_types:
        record["PE"] = cdash_record.get("PE")
        record["EG"] = cdash_record.get("EG")
    else:
        record["PE"] = None
        record["EG"] = None

    # Subjective: only from HR (may be null on non-visit days)
    record["subjective"] = hr_subj

    # Care record: include if present (video calls are hospital-observable)
    record["care_record"] = cdash_record.get("care_record", [])

    # Observation events
    record["observation_events"] = obs_events

    # Mood state: NOT included (internal/GT only)
    # _sim metadata: NOT included (internal/GT only)

    return record


def _vitals_from_hr(hr_vitals: dict) -> dict:
    """Convert HR vitals format to VS-like format for stale data."""
    if not hr_vitals:
        return {}
    return {
        "TEMP_VSORRES": hr_vitals.get("BT"),
        "SYSBP_VSORRES": hr_vitals.get("SBP"),
        "DIABP_VSORRES": hr_vitals.get("DBP"),
        "PULSE_VSORRES": hr_vitals.get("HR"),
        "RESP_VSORRES": hr_vitals.get("RR"),
        "OXYSAT_VSORRES": hr_vitals.get("SpO2"),
        "WEIGHT_VSORRES": hr_vitals.get("weight_kg"),
        "_stale": True,
    }


def _labs_from_hr(hr_labs: dict) -> dict:
    """Convert HR labs format to LB-like format for stale data."""
    if not hr_labs:
        return {}
    results = {}
    for name, info in hr_labs.items():
        if isinstance(info, dict):
            results[name] = {
                "LBORRES": info.get("value"),
                "LBORRESU": info.get("unit", ""),
                "_stale": True,
            }
        else:
            results[name] = {"LBORRES": info, "_stale": True}
    return {"results": results, "_stale": True}
