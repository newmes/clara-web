"""
Shared data helpers, constants, and utility functions for viewer views.
"""
import json
import logging
from functools import lru_cache
from pathlib import Path

from django.conf import settings

from ..crf_aggregator import _read_jsonl_cached, _load_patient_json

logger = logging.getLogger(__name__)

# ─── Data helpers ─────────────────────────────────────────────

DATA_DIR = settings.DATA_DIR
MAP_ASSETS_DIR = Path(settings.BASE_DIR) / "static_dirs" / "assets" / "map"
PINNED_RUN_ID = "20260224_183746_Padcev___Pembrolizumab_100pt_126d"


def _get_runs():
    """Available simulation runs, newest first."""
    runs_dir = DATA_DIR / "runs"
    if not runs_dir.exists():
        return []
    runs = []
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if d.is_dir() and (d / "simulations").exists():
            modes = []
            natural_files = [f for f in (d / "simulations").glob("*_natural.jsonl")
                           if "_hospital" not in f.stem]
            care_ai_files = [f for f in (d / "simulations").glob("*_care_ai.jsonl")
                           if "_hospital" not in f.stem]
            if natural_files:
                modes.append("natural")
            if care_ai_files:
                modes.append("care_ai")

            run_info = {
                "id": d.name,
                "path": str(d),
                "modes": modes,
                "status": "completed",
            }

            # Check for run_meta.json
            meta_path = d / "run_meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    run_info["drug_name"] = meta.get("drug_name", "")
                    run_info["indication"] = meta.get("indication", "")
                    run_info["n_patients"] = meta.get("n_patients")
                    run_info["total_days"] = meta.get("total_days")
                    run_info["status"] = meta.get("status", "completed")
                    run_info["started_at"] = meta.get("started_at")
                except Exception:
                    pass

            # Count patients
            patients_dir = d / "patients"
            if patients_dir.exists() and "n_patients" not in run_info:
                run_info["n_patients"] = len(list(patients_dir.glob("*.json")))

            runs.append(run_info)

    # Also include runs that are still generating (simulations/ might not exist yet)
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if d.is_dir() and not (d / "simulations").exists():
            meta_path = d / "run_meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if meta.get("status") in ("running", "generating_patients",
                                               "starting"):
                        runs.insert(0, {
                            "id": d.name,
                            "path": str(d),
                            "modes": [],
                            "status": meta.get("status", "running"),
                            "drug_name": meta.get("drug_name", ""),
                            "n_patients": meta.get("n_patients"),
                            "total_days": meta.get("total_days"),
                        })
                except Exception:
                    pass

    return runs


def _get_run_path(run_id: str) -> Path:
    return DATA_DIR / "runs" / run_id


_warmed_keys: set[tuple] = set()


def _warm_cache(run_path: Path, patient_ids: list[str], mode: str):
    """Pre-load JSONL + patient JSON into LRU cache.

    First call for a (run, mode) pair reads all files (~2-3s for 100 patients).
    Subsequent calls are instant (set check -> skip).
    This shifts the file I/O cost from being spread across many API calls
    to a single upfront cost on first page load.
    """
    key = (str(run_path), mode)
    if key in _warmed_keys:
        return
    _warmed_keys.add(key)

    sim_dir = run_path / "simulations"
    if not sim_dir.exists():
        return

    for pid in patient_ids:
        fpath = sim_dir / f"{pid}_{mode}.jsonl"
        if fpath.exists():
            _read_jsonl_cached(fpath)
        _load_patient_json(run_path, pid)


def _get_day_index(run_path: Path, patient_id: str, mode: str) -> dict[int, dict]:
    """Build {day: record} index from cached JSONL. O(1) day lookup."""
    fpath = run_path / "simulations" / f"{patient_id}_{mode}.jsonl"
    if not fpath.exists():
        return {}
    records = _read_jsonl_cached(fpath)
    return {r.get("day"): r for r in records}


def _load_patient_profile(run_path: Path, patient_id: str) -> dict:
    """Load patient JSON profile (demographics, persona, etc.)."""
    result = _load_patient_json(run_path, patient_id)
    return result if result is not None else {}


@lru_cache(maxsize=64)
def _load_run_meta_cached(fpath_str: str, _mtime: float) -> dict:
    with open(fpath_str, encoding="utf-8") as fh:
        return json.load(fh)


def _load_run_meta(run_path: Path) -> dict:
    """Load run_meta.json for this run (mtime-cached)."""
    f = run_path / "run_meta.json"
    if not f.exists():
        return {}
    try:
        return _load_run_meta_cached(str(f), f.stat().st_mtime)
    except Exception:
        return {}


@lru_cache(maxsize=32)
def _load_rule_set_cached(fpath_str: str, _mtime: float) -> dict:
    with open(fpath_str, encoding="utf-8") as fh:
        return json.load(fh)


def _load_rule_set(run_path: Path) -> dict:
    """Load rule_set.json for this run (mtime-cached)."""
    f = run_path / "rule_set.json"
    if not f.exists():
        return {}
    try:
        return _load_rule_set_cached(str(f), f.stat().st_mtime)
    except Exception:
        return {}


def _extract_lab_ranges(run_path: Path, mode: str = "natural") -> dict:
    """Extract lab reference ranges from LB data (LBORNRLO/LBORNRHI).

    Used when rule_set.json is missing or has no lab_reference_ranges.
    Reads the first patient's first day with LB data and builds a ranges dict.
    """
    from frontend.viewer.crf_aggregator import LAB_ABBREVIATIONS, _lab_display_name
    sim_dir = run_path / "simulations"
    if not sim_dir.exists():
        return {}
    # Find any JSONL file to extract ranges from
    for fpath in sorted(sim_dir.glob(f"*_{mode}.jsonl")):
        if "_hospital" in fpath.stem:
            continue
        records = _read_jsonl_cached(fpath)
        for record in records:
            lb = record.get("LB")
            if not lb or not lb.get("LBPERF"):
                continue
            results = lb.get("results", {})
            ranges = {}
            for test_name, vals in results.items():
                lo = vals.get("LBORNRLO")
                hi = vals.get("LBORNRHI")
                if lo is not None or hi is not None:
                    display = _lab_display_name(test_name)
                    ranges[display] = {
                        "unit": vals.get("LBORRESU", ""),
                        "normal_range": {"min": lo, "max": hi},
                        "LLN": lo,
                        "ULN": hi,
                    }
            if ranges:
                return ranges
    return {}


@lru_cache(maxsize=32)
def _list_patients_cached(run_path_str: str, _sim_mtime: float, _pat_mtime: float) -> list[str]:
    run_path = Path(run_path_str)
    ids = set()
    sim_dir = run_path / "simulations"
    if sim_dir.exists():
        for f in sim_dir.glob("*_natural.jsonl"):
            if "_hospital" in f.stem:
                continue
            pid = f.stem.replace("_natural", "")
            ids.add(pid)
        for f in sim_dir.glob("*_care_ai.jsonl"):
            if "_hospital" in f.stem:
                continue
            pid = f.stem.replace("_care_ai", "")
            ids.add(pid)
    # Fallback: count from patients/ directory
    if not ids:
        patients_dir = run_path / "patients"
        if patients_dir.exists():
            for f in patients_dir.glob("*.json"):
                ids.add(f.stem)
    return sorted(ids)


def _list_patients(run_path: Path) -> list[str]:
    """List patient IDs from simulation files (exclude _hospital variants).
    Falls back to patients/ directory if no simulation files exist yet.
    Directory-mtime cached."""
    sim_dir = run_path / "simulations"
    patients_dir = run_path / "patients"
    sim_mtime = sim_dir.stat().st_mtime if sim_dir.exists() else 0
    pat_mtime = patients_dir.stat().st_mtime if patients_dir.exists() else 0
    return _list_patients_cached(str(run_path), sim_mtime, pat_mtime)


def _load_day_for_patient(run_path: Path, patient_id: str, day: int,
                          mode: str = "natural") -> dict | None:
    """Load a single day's data for a patient from cached JSONL.

    Uses _get_day_index() for O(1) dict lookup instead of linear file scan.
    If the requested day is beyond the last entry AND the patient is
    deceased on the last day, return the death-day record so that
    downstream code still sees location='DECEASED'.
    """
    day_index = _get_day_index(run_path, patient_id, mode)
    if not day_index:
        return None
    record = day_index.get(day)
    if record is not None:
        return record
    # Day not found -- check if patient died before this day
    max_day = max(day_index.keys())
    if max_day < day:
        last = day_index[max_day]
        if (last.get("objective") or {}).get("location", "") == "DECEASED":
            return last
    return None


def _load_all_days_for_patient(run_path: Path, patient_id: str,
                               mode: str = "natural") -> list[dict]:
    """Load all days for a patient (cached via _read_jsonl_cached)."""
    fpath = run_path / "simulations" / f"{patient_id}_{mode}.jsonl"
    if not fpath.exists():
        return []
    return list(_read_jsonl_cached(fpath))


def _find_last_hr_observation(run_path: Path, patient_id: str, day: int,
                              mode: str = "natural") -> tuple[dict, int]:
    """Find the last day with non-empty hospital_record labs/vitals.

    Returns (hr_objective, stale_days). If none found, returns ({}, 0).
    Uses cached data with reverse search -- no file I/O.
    """
    day_index = _get_day_index(run_path, patient_id, mode)
    if not day_index:
        return {}, 0
    for d_num in sorted((d for d in day_index if d < day), reverse=True):
        record = day_index[d_num]
        hr = record.get("hospital_record", {})
        hr_obj = hr.get("objective", {})
        if hr_obj.get("labs") or hr_obj.get("vitals"):
            return hr_obj, day - d_num
    return {}, 0


def _count_days(run_path: Path, mode: str | None = None) -> int:
    """Find max day across all patients.

    Uses cached JSONL data -- checks only last record per file.
    If mode is specified, only count that mode's files.
    Otherwise, count across all modes.
    """
    sim_dir = run_path / "simulations"
    if not sim_dir.exists():
        return 0
    max_day = 0
    patterns = [f"*_{mode}.jsonl"] if mode else ["*_natural.jsonl", "*_care_ai.jsonl"]
    for pattern in patterns:
        for fpath in sim_dir.glob(pattern):
            if "_hospital" in fpath.stem:
                continue
            records = _read_jsonl_cached(fpath)
            if records:
                last_day = records[-1].get("day", 0)
                if last_day > max_day:
                    max_day = last_day
    return max_day


def _is_hr_tumor_stuck(all_days: list[dict], current_day: int) -> bool:
    """Detect if HR tumor data is stuck (never updated from baseline).

    Legacy runs generated before the RECIST fix have HR tumor frozen at
    the Day 1 value. We detect this by checking if HR tumor is identical
    across multiple visit days.
    """
    tumor_vals = set()
    visit_count = 0
    for d in all_days:
        d_num = d.get("day", 0)
        if d_num > current_day:
            break
        hr = d.get("hospital_record", {})
        obs_types = hr.get("observation_types", [])
        hr_obj = hr.get("objective", {})
        hr_t = hr_obj.get("tumor")
        if obs_types and hr_t:
            pct = hr_t.get("estimated_change_pct")
            if pct is not None:
                tumor_vals.add(round(pct, 2))
                visit_count += 1
    # If we've had 3+ visit days with tumor data and they all have the same value,
    # it's almost certainly stuck
    return visit_count >= 3 and len(tumor_vals) <= 1


def _extract_day_events(day_data: dict) -> list[dict]:
    """Extract notable events from a day's data for the summary panel."""
    events = []
    pid = day_data.get("patient_id", "?")

    # AE events
    for ae in day_data.get("AE", []):
        ae_term = ae.get("AETERM", "unknown")
        grade = ae.get("_grade", "?")
        status = ae.get("_status", "")
        days_active = ae.get("_days_active", 0)

        if days_active <= 1:
            events.append({
                "type": "ae_onset",
                "severity": "high" if grade >= 3 else "medium",
                "icon": "\U0001f534" if grade >= 3 else "\U0001f7e1",
                "text": f"{pid}: {ae_term} Grade {grade} onset",
            })
        elif "worsened" in status:
            events.append({
                "type": "ae_worsened",
                "severity": "high" if grade >= 3 else "medium",
                "icon": "\U0001f534" if grade >= 3 else "\U0001f7e1",
                "text": f"{pid}: {ae_term} worsened to Grade {grade}",
            })

    # Resolved AEs (checking _status)
    for ae in day_data.get("AE", []):
        if ae.get("_status") == "resolved":
            events.append({
                "type": "ae_resolved",
                "severity": "low",
                "icon": "\U0001f7e2",
                "text": f"{pid}: {ae.get('AETERM', '?')} resolved",
            })

    # Dose modifications
    for ec in day_data.get("EC", []):
        if ec.get("ECDOSADJ"):
            drug = ec.get("ECREFID", "?")
            adj = ec.get("ECADJ", "modified")
            events.append({
                "type": "dose_mod",
                "severity": "medium",
                "icon": "\U0001f48a",
                "text": f"{pid}: {drug} {adj}",
            })

    # Treatment administration
    for ec in day_data.get("EC", []):
        if ec.get("ECTRTCMP") and not ec.get("ECDOSADJ"):
            drug = ec.get("ECREFID", "?")
            events.append({
                "type": "treatment",
                "severity": "info",
                "icon": "\U0001f489",
                "text": f"{pid}: {drug} administered",
            })

    # RECIST scan
    rs = day_data.get("RS")
    if rs:
        if isinstance(rs, list):
            for r in rs:
                events.append({
                    "type": "recist",
                    "severity": "info",
                    "icon": "\U0001f4cb",
                    "text": f"{pid}: RECIST scan \u2014 {r.get('RSORRESU', '?')}",
                })
        elif isinstance(rs, dict):
            events.append({
                "type": "recist",
                "severity": "info",
                "icon": "\U0001f4cb",
                "text": f"{pid}: RECIST scan \u2014 {rs.get('RSORRESU', '?')}",
            })

    # Discontinuation
    ds = day_data.get("DS")
    if ds:
        events.append({
            "type": "discontinuation",
            "severity": "high",
            "icon": "\u26d4",
            "text": f"{pid}: Discontinued \u2014 {ds.get('DSDECOD', '?')}",
        })

    # Video call / Care record
    for cr in day_data.get("care_record", []):
        assessment = cr.get("nurse_assessment", {})
        severity_level = assessment.get("severity_level", "green")
        summary_text = assessment.get("summary", "Care AI interaction")
        sev_map = {"green": "info", "yellow": "info", "orange": "medium", "red": "high"}
        icon_map = {"green": "\U0001f4f9", "yellow": "\U0001f4f9", "orange": "\U0001f7e0", "red": "\U0001f534"}
        events.append({
            "type": "video_call",
            "severity": sev_map.get(severity_level, "info"),
            "icon": icon_map.get(severity_level, "\U0001f4f9"),
            "text": f"{pid}: Video call [{severity_level.upper()}] \u2014 {summary_text}",
        })
        for action in cr.get("actions", []):
            act = action.get("action", "")
            if act not in ("no_action", "monitor_closely"):
                events.append({
                    "type": "care_action",
                    "severity": "medium",
                    "icon": "\U0001fa7a",
                    "text": f"{pid}: Care AI \u2192 {act}: {action.get('reason', '')}",
                })

    # Observation events
    for obs in day_data.get("observation_events", []):
        obs_type = obs.get("type", "")
        if obs_type == "self_report":
            events.append({
                "type": "self_report", "severity": "info", "icon": "\U0001f4de",
                "text": f"{pid}: Self-reported symptoms to clinic",
            })
        elif obs_type == "er_visit":
            events.append({
                "type": "er_visit", "severity": "high", "icon": "\U0001f691",
                "text": f"{pid}: Emergency room visit",
            })

    return events


def _patient_summary(profile: dict, day_data: dict | None,
                     view_mode: str = "gt",
                     run_path: Path = None, mode: str = "natural") -> dict:
    """Build a summary dict for a patient card.

    Args:
        view_mode: "gt" for Ground Truth, "hr" for Hospital Record
        run_path: needed for HR carry-forward when hospital_record is empty
        mode: simulation mode for HR carry-forward
    """
    dm = profile.get("DM", {})
    persona = profile.get("persona", {})

    summary = {
        "patient_id": profile.get("patient_id", "?"),
        "age": dm.get("AGE", "?"),
        "sex": dm.get("SEX", "?"),
        "race": dm.get("RACE", ""),
        "persona_type": persona.get("type", "unknown"),
        "persona_desc": persona.get("description", ""),
        "view_mode": view_mode,
    }

    if day_data:
        hr_data = day_data.get("hospital_record", {})
        hr_obj = hr_data.get("objective", {})
        gt_obj = day_data.get("objective", {})
        obs_types = hr_data.get("observation_types", [])
        is_visit = ("scheduled_visit" in obs_types or "er_visit" in obs_types)

        if view_mode == "hr":
            # -- Hospital Record mode: ONLY what the hospital knows --
            # Carry-forward: if current HR has no labs/vitals, find last known
            if not hr_obj.get("labs") and not hr_obj.get("vitals") and run_path:
                pid = profile.get("patient_id", "?")
                cur_day = day_data.get("day", 0)
                last_hr_obj, stale = _find_last_hr_observation(
                    run_path, pid, cur_day, mode)
                if last_hr_obj:
                    hr_obj = dict(hr_obj) if hr_obj else {}
                    if not hr_obj.get("labs"):
                        hr_obj["labs"] = last_hr_obj.get("labs", {})
                        hr_obj["labs_stale_days"] = stale
                    if not hr_obj.get("vitals"):
                        hr_obj["vitals"] = last_hr_obj.get("vitals", {})
                        hr_obj["vitals_stale_days"] = stale
                    for k in ("active_aes", "treatment_status", "ecog",
                              "location", "tumor"):
                        if k not in hr_obj and k in last_hr_obj:
                            hr_obj[k] = last_hr_obj[k]

            summary["location"] = hr_obj.get("location", gt_obj.get("location", "?"))
            summary["treatment_status"] = hr_obj.get(
                "treatment_status", gt_obj.get("treatment_status", "?"))
            summary["ecog"] = hr_obj.get("ecog", "?")
            tumor = hr_obj.get("tumor") or {}
            summary["tumor_change_pct"] = tumor.get("estimated_change_pct")
            summary["is_visit_day"] = is_visit
            summary["observation_types"] = obs_types

            # AEs: only detected
            active_aes = []
            for ae in hr_obj.get("active_aes", []):
                active_aes.append({
                    "term": ae.get("ae", "?"),
                    "grade": ae.get("grade", "?"),
                    "days_active": ae.get("days_active", 0),
                    "detected_day": ae.get("detected_day"),
                    "detection_delay": ae.get("detection_delay"),
                    "channel": ae.get("channel", ""),
                })
            summary["active_aes"] = active_aes

            # Labs: from HR (stale on non-visit days)
            hr_labs = hr_obj.get("labs", {})
            summary["labs"] = {}
            for name, info in hr_labs.items():
                if isinstance(info, dict):
                    summary["labs"][name] = info.get("value")
                else:
                    summary["labs"][name] = info
            summary["labs_stale_days"] = hr_obj.get("labs_stale_days", 0)

            # Vitals: from HR (stale on non-visit days)
            hr_vitals = hr_obj.get("vitals", {})
            summary["vitals"] = {
                "temp": hr_vitals.get("BT"),
                "bp": f"{hr_vitals.get('SBP', '?')}/{hr_vitals.get('DBP', '?')}",
                "hr": hr_vitals.get("HR"),
                "spo2": hr_vitals.get("SpO2"),
                "rr": hr_vitals.get("RR"),
                "weight": hr_vitals.get("weight_kg"),
            }
            summary["vitals_stale_days"] = hr_obj.get("vitals_stale_days", 0)

            # Subjective: only from HR (empty on non-visit days)
            hr_subj = hr_data.get("subjective", {})
            if hr_subj:
                summary["awareness"] = hr_subj.get("overall_awareness", "?")
                summary["symptoms_perceived"] = hr_subj.get(
                    "symptoms_patient_perceives", [])
            else:
                summary["awareness"] = "UNKNOWN"
                summary["symptoms_perceived"] = []

        else:
            # -- Ground Truth mode: full picture --
            summary["location"] = gt_obj.get("location", "?")
            summary["treatment_status"] = gt_obj.get("treatment_status", "?")
            summary["ecog"] = gt_obj.get("ecog", "?")
            gt_tumor = gt_obj.get("tumor") or {}
            summary["tumor_change_pct"] = gt_tumor.get(
                "estimated_change_pct", 0)

            active_aes = []
            for ae in day_data.get("AE", []):
                if ae.get("AEONGO") or ae.get("_status", "").startswith("active"):
                    active_aes.append({
                        "term": ae.get("AETERM", "?"),
                        "grade": ae.get("_grade", "?"),
                        "days_active": ae.get("_days_active", 0),
                    })
            summary["active_aes"] = active_aes

            # All labs from GT
            lb = day_data.get("LB", {}).get("results", {})
            summary["labs"] = {}
            for name, info in lb.items():
                if isinstance(info, dict):
                    summary["labs"][name] = info.get("LBORRES")
                else:
                    summary["labs"][name] = info

            # Vitals from GT
            vs = day_data.get("VS", {})
            summary["vitals"] = {
                "temp": vs.get("TEMP_VSORRES"),
                "bp": f"{vs.get('SYSBP_VSORRES', '?')}/{vs.get('DIABP_VSORRES', '?')}",
                "hr": vs.get("PULSE_VSORRES"),
                "spo2": vs.get("_SpO2") or vs.get("OXYSAT_VSORRES"),
                "rr": vs.get("RESP_VSORRES"),
                "weight": vs.get("WEIGHT_VSORRES"),
            }

            # Subjective from GT
            subj = day_data.get("subjective", {})
            summary["awareness"] = subj.get("overall_awareness", "?")
            summary["symptoms_perceived"] = subj.get(
                "symptoms_patient_perceives", [])

        # Sim metadata
        sim = day_data.get("_sim", {})
        summary["generation_mode"] = sim.get("generation_mode", "?")
        summary["mortality_risk"] = sim.get("mortality_risk", 0)

        # -- New: Mood state --
        summary["mood"] = day_data.get("mood_state", {})

        # -- New: Hospital record (what the hospital knows) --
        raw_hr = day_data.get("hospital_record", {})
        raw_hr_obj = raw_hr.get("objective", {})
        # Carry-forward: if HR has no labs/vitals, look back
        if not raw_hr_obj.get("labs") and not raw_hr_obj.get("vitals") and run_path:
            pid_cf = profile.get("patient_id", "?")
            cur_day_cf = day_data.get("day", 0)
            cf_hr_obj, cf_stale = _find_last_hr_observation(
                run_path, pid_cf, cur_day_cf, mode)
            if cf_hr_obj:
                merged_hr = dict(raw_hr)
                merged_obj = dict(raw_hr_obj) if raw_hr_obj else {}
                if not merged_obj.get("labs"):
                    merged_obj["labs"] = cf_hr_obj.get("labs", {})
                    merged_obj["labs_stale_days"] = cf_stale
                if not merged_obj.get("vitals"):
                    merged_obj["vitals"] = cf_hr_obj.get("vitals", {})
                    merged_obj["vitals_stale_days"] = cf_stale
                for k in ("active_aes", "treatment_status", "ecog",
                          "location", "tumor"):
                    if k not in merged_obj and k in cf_hr_obj:
                        merged_obj[k] = cf_hr_obj[k]
                merged_hr["objective"] = merged_obj
                raw_hr = merged_hr
        summary["hospital_record"] = raw_hr

        # -- New: Observation events --
        summary["observation_events"] = day_data.get("observation_events", [])

        # -- New: Care AI record --
        care_records = day_data.get("care_record", [])
        if care_records:
            cr = care_records[0] if isinstance(care_records, list) else care_records
            summary["care_record"] = {
                "severity_level": cr.get("nurse_assessment", {}).get("severity_level", ""),
                "summary": cr.get("nurse_assessment", {}).get("summary", ""),
                "actions": cr.get("actions", []),
                "detection": cr.get("detection", {}),
                "turns": cr.get("turns", []),
                "terminated_early": cr.get("terminated_early", False),
                "mood_snapshot": cr.get("mood_snapshot", {}),
                "interaction_quality": cr.get("interaction_quality", {}),
                "grade_distortion": cr.get("grade_distortion", 0),
            }
        else:
            summary["care_record"] = None

    return summary
