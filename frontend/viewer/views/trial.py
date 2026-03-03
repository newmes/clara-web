"""
Trial viewer views: trial_viewer, patient_state, api_day_data,
api_run_meta, api_patient_timeline, sse_stream.
"""
import json
import time

from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render

from ._helpers import (
    _get_run_path, _list_patients, _warm_cache, _count_days,
    _load_rule_set, _load_run_meta, _load_patient_profile,
    _load_day_for_patient, _load_all_days_for_patient,
    _find_last_hr_observation, _is_hr_tumor_stuck,
    _extract_day_events, _patient_summary, _extract_lab_ranges,
    logger,
)
from .map import _ensure_map_for_run
from .sim_api import _live_sims


def trial_viewer(request, run_id: str, day: int = 1):
    """Main trial viewer page -- Generative Agents demo style."""
    run_path = _get_run_path(run_id)
    if not run_path.exists():
        return HttpResponse("Run not found", status=404)

    # Check which modes are available
    sim_dir = run_path / "simulations"
    if not sim_dir.exists():
        sim_dir.mkdir(parents=True, exist_ok=True)
    available_modes = []
    natural_files = [f for f in sim_dir.glob("*_natural.jsonl")
                   if "_hospital" not in f.stem]
    care_files = [f for f in sim_dir.glob("*_care_ai.jsonl")
                if "_hospital" not in f.stem]
    if care_files:
        available_modes.append("care_ai")
    if natural_files:
        available_modes.append("natural")

    # Default to first available mode if requested mode doesn't exist
    mode = request.GET.get("mode", "")
    if mode not in available_modes:
        mode = available_modes[0] if available_modes else "natural"

    view_mode = request.GET.get("view", "hr")  # default Hospital Record

    patient_ids = _list_patients(run_path)
    _warm_cache(run_path, patient_ids, mode)
    total_days = _count_days(run_path, mode)
    rule_set = _load_rule_set(run_path)

    # Ensure map is generated for this run
    try:
        _ensure_map_for_run(run_path, len(patient_ids))
    except Exception:
        pass  # non-critical; map will fall back to default

    # Load profiles + collect events in single loop
    patients = []
    all_events = []
    for pid in patient_ids:
        profile = _load_patient_profile(run_path, pid)
        day_data = _load_day_for_patient(run_path, pid, day, mode)
        patients.append(_patient_summary(
            profile, day_data, view_mode, run_path=run_path, mode=mode))
        if day_data:
            all_events.extend(_extract_day_events(day_data))

    # Sort events: high severity first
    severity_order = {"high": 0, "medium": 1, "info": 2, "low": 3}
    all_events.sort(key=lambda e: severity_order.get(e["severity"], 9))

    # Cycle info
    cycle_length = 21
    if rule_set:
        td = rule_set.get("trial_design", {})
        cycle_length = td.get("cycle_length_days", 21)
    cycle = (day - 1) // cycle_length + 1
    cycle_day = (day - 1) % cycle_length + 1

    drug_name = rule_set.get("drug_name", "Unknown")
    indication = rule_set.get("indication", "")

    # Live simulation detection
    is_live = request.GET.get("live") == "1" or run_id in _live_sims

    context = {
        "run_id": run_id,
        "day": day,
        "total_days": total_days,
        "cycle": cycle,
        "cycle_day": cycle_day,
        "cycle_length": cycle_length,
        "drug_name": drug_name,
        "indication": indication,
        "mode": mode,
        "view_mode": view_mode,
        "available_modes": available_modes,
        "available_modes_json": json.dumps(available_modes),
        "patients": patients,
        "patients_json": json.dumps(patients),
        "events": all_events,
        "events_json": json.dumps(all_events),
        "patient_ids": patient_ids,
        "patient_ids_json": json.dumps(patient_ids),
        "is_live": is_live,
        "model_name": _load_run_meta(run_path).get("model", ""),
        "lab_ranges_json": json.dumps(rule_set.get("lab_reference_ranges", {}) or _extract_lab_ranges(run_path)),
    }
    return render(request, "trial/trial.html", context)


def patient_state(request, run_id: str, patient_id: str, day: int = None):
    """Patient detail page -- like Generative Agents persona_state."""
    run_path = _get_run_path(run_id)
    if not run_path.exists():
        return HttpResponse("Run not found", status=404)

    # Auto-detect mode if not specified
    sim_dir = run_path / "simulations"
    avail = []
    if [f for f in sim_dir.glob("*_natural.jsonl") if "_hospital" not in f.stem]:
        avail.append("natural")
    if [f for f in sim_dir.glob("*_care_ai.jsonl") if "_hospital" not in f.stem]:
        avail.append("care_ai")
    mode = request.GET.get("mode", "")
    if mode not in avail:
        mode = avail[0] if avail else "natural"

    view_mode = request.GET.get("view", "hr")

    profile = _load_patient_profile(run_path, patient_id)
    all_days = _load_all_days_for_patient(run_path, patient_id, mode)

    if not day and all_days:
        day = all_days[-1].get("day", 1)
    current_day_data = None
    for d in all_days:
        if d.get("day") == day:
            current_day_data = d
            break

    # Django templates disallow underscore-prefixed attributes.
    # Remap _grade, _status, _days_active etc. into template-safe keys.
    if current_day_data:
        current_day_data = dict(current_day_data)

        # -- Compute RECIST scan schedule (for HR tumor reconstruction) --
        rule_path = run_path / "rule_set.json"
        cycle_len = 21
        if rule_path.exists():
            try:
                rs = json.loads(rule_path.read_text(encoding="utf-8"))
                cycle_len = rs.get("trial_design", {}).get("cycle_length_days", 21)
            except Exception:
                pass
        total_d = len(all_days)
        first_scan = cycle_len * 2 + 7  # e.g. 49
        scan_interval = cycle_len * 2   # e.g. 42
        recist_scan_days = set()
        sd = first_scan
        while sd <= total_d:
            recist_scan_days.add(sd)
            sd += scan_interval

        if view_mode == "hr":
            # -- Hospital Record mode: replace GT fields with HR data --
            hr = current_day_data.get("hospital_record", {})
            hr_obj = hr.get("objective", {})
            obs_types = hr.get("observation_types", [])

            # If current day has empty hospital_record, search backwards
            # for the last day with actual HR data (carry-forward logic).
            if not hr_obj.get("labs") and not hr_obj.get("vitals"):
                last_hr_day_num = None
                last_hr = {}
                last_hr_obj = {}
                for prev_d in reversed(all_days):
                    pd_num = prev_d.get("day", 0)
                    if pd_num >= day:
                        continue
                    prev_hr = prev_d.get("hospital_record", {})
                    prev_hr_obj = prev_hr.get("objective", {})
                    if prev_hr_obj.get("labs") or prev_hr_obj.get("vitals"):
                        last_hr_day_num = pd_num
                        last_hr = prev_hr
                        last_hr_obj = prev_hr_obj
                        break
                if last_hr_day_num is not None:
                    stale_days = day - last_hr_day_num
                    # Merge: use last known data but keep current day's
                    # obs_types and any AEs if present
                    if not hr_obj.get("labs"):
                        hr_obj = dict(hr_obj)
                        hr_obj["labs"] = last_hr_obj.get("labs", {})
                        hr_obj["labs_stale_days"] = stale_days
                    if not hr_obj.get("vitals"):
                        hr_obj = dict(hr_obj)
                        hr_obj["vitals"] = last_hr_obj.get("vitals", {})
                        hr_obj["vitals_stale_days"] = stale_days
                    if not hr_obj.get("active_aes"):
                        hr_obj.setdefault("active_aes",
                                          last_hr_obj.get("active_aes", []))
                    for k in ("treatment_status", "ecog", "location", "tumor"):
                        if k not in hr_obj and k in last_hr_obj:
                            hr_obj[k] = last_hr_obj[k]

            # AEs: only hospital-detected
            hr_aes = []
            for ae in hr_obj.get("active_aes", []):
                hr_aes.append({
                    "AETERM": ae.get("ae", ""),
                    "grade": ae.get("grade", 0),
                    "status": "active",
                    "days_active": ae.get("days_active", 0),
                    "channel": ae.get("channel", ""),
                    "detection_delay": ae.get("detection_delay"),
                    "detected_day": ae.get("detected_day"),
                    "AEONGO": True,
                })
            current_day_data["AE"] = hr_aes
            current_day_data["safe_AE"] = hr_aes

            # Labs: from HR (stale values)
            hr_labs = hr_obj.get("labs", {})
            hr_results = {}
            for name, info in hr_labs.items():
                val = info.get("value") if isinstance(info, dict) else info
                trend = info.get("trend", "") if isinstance(info, dict) else ""
                hr_results[name] = {
                    "LBORRES": val,
                    "LBORRESU": info.get("unit", "") if isinstance(info, dict) else "",
                    "_trend": trend,
                }
            current_day_data["LB"] = {
                "results": hr_results,
                "labs_stale_days": hr_obj.get("labs_stale_days", 0),
            }

            # Vitals: from HR
            hr_vitals = hr_obj.get("vitals", {})
            current_day_data["VS"] = {
                "TEMP_VSORRES": hr_vitals.get("BT"),
                "SYSBP_VSORRES": hr_vitals.get("SBP"),
                "DIABP_VSORRES": hr_vitals.get("DBP"),
                "PULSE_VSORRES": hr_vitals.get("HR"),
                "RESP_VSORRES": hr_vitals.get("RR"),
                "OXYSAT_VSORRES": hr_vitals.get("SpO2"),
                "WEIGHT_VSORRES": hr_vitals.get("weight_kg"),
                "vitals_stale_days": hr_obj.get("vitals_stale_days", 0),
            }

            # Subjective: from HR (empty on non-visit days)
            hr_subj = hr.get("subjective", {})
            if hr_subj:
                current_day_data["subjective"] = hr_subj
            else:
                current_day_data["subjective"] = {
                    "overall_awareness": "UNKNOWN",
                    "symptoms_patient_perceives": [],
                }

            # Objective: replace tumor and ecog with HR values
            gt_obj = current_day_data.get("objective", {})
            hr_objective = dict(gt_obj)
            hr_objective["treatment_status"] = hr_obj.get(
                "treatment_status", gt_obj.get("treatment_status"))
            hr_objective["ecog"] = hr_obj.get(
                "ecog", gt_obj.get("ecog"))
            hr_objective["_ecog_note"] = "Last clinical assessment"
            hr_objective["_labs_stale_days"] = hr_obj.get("labs_stale_days", 0)
            hr_objective["_vitals_stale_days"] = hr_obj.get("vitals_stale_days", 0)

            # -- HR Tumor: reconstruct from scan schedule if stuck --
            hr_tumor = hr_obj.get("tumor")
            hr_tumor_stuck = _is_hr_tumor_stuck(all_days, day)
            if hr_tumor_stuck:
                # Legacy run: HR tumor never updated properly.
                # Reconstruct: find the last RECIST scan day <= current day
                # and use GT tumor from that day.
                last_scan_tumor = None
                last_scan_day_num = None
                for d in all_days:
                    d_num = d.get("day", 0)
                    if d_num > day:
                        break
                    if d_num in recist_scan_days:
                        gt_t = d.get("objective", {}).get("tumor", {})
                        if gt_t and gt_t.get("estimated_change_pct") is not None:
                            last_scan_tumor = gt_t
                            last_scan_day_num = d_num
                if last_scan_tumor:
                    hr_objective["tumor"] = last_scan_tumor
                    hr_objective["_tumor_note"] = f"RECIST scan (Day {last_scan_day_num})"
                else:
                    hr_objective["tumor"] = hr_tumor
                    hr_objective["_tumor_note"] = "Baseline only \u2014 no RECIST scan yet"
            else:
                hr_objective["tumor"] = hr_tumor
                if hr_tumor:
                    hr_objective["_tumor_note"] = "Last RECIST scan"
                else:
                    hr_objective["_tumor_note"] = "No scan performed yet"

            current_day_data["objective"] = hr_objective

            # Location for display
            current_day_data["_display_location"] = hr_obj.get("location", gt_obj.get("location", "HOME"))
            current_day_data["_hr_obs_types"] = obs_types
            current_day_data["_hr_is_visit"] = ("scheduled_visit" in obs_types or "er_visit" in obs_types)
        else:
            # GT mode: standard safe_AE mapping
            safe_aes = []
            for ae in current_day_data.get("AE", []):
                safe_ae = {k.lstrip("_") if k.startswith("_") else k: v
                           for k, v in ae.items()}
                safe_aes.append(safe_ae)
            current_day_data["safe_AE"] = safe_aes

            # GT mode: normalise VS keys for template compatibility
            gt_vs = current_day_data.get("VS")
            if isinstance(gt_vs, dict):
                if "_SpO2" in gt_vs and "OXYSAT_VSORRES" not in gt_vs:
                    gt_vs["OXYSAT_VSORRES"] = gt_vs["_SpO2"]

            # Location for display
            gt_obj = current_day_data.get("objective", {})
            current_day_data["_display_location"] = gt_obj.get("location", "HOME")
            hr = current_day_data.get("hospital_record", {})
            obs_types = hr.get("observation_types", [])
            current_day_data["_hr_is_visit"] = ("scheduled_visit" in obs_types or "er_visit" in obs_types)
            current_day_data["_hr_obs_types"] = obs_types

    # Filter: only show data up to and including the current viewing day
    visible_days = [d for d in all_days if d.get("day", 0) <= day]

    if view_mode == "hr":
        # Hospital Record: AEs, labs, vitals from hospital_record only
        ae_timeline = []
        _hr_ae_seen = {}  # track per-AE to avoid duplicates per day
        for d in visible_days:
            day_num = d.get("day", 0)
            hr = d.get("hospital_record", {}).get("objective", {})
            for ae in hr.get("active_aes", []):
                ae_term = ae.get("ae")
                detected_day = ae.get("detected_day", day_num)
                resolved_day = ae.get("resolved_day")
                status = ae.get("status", "active")

                ae_timeline.append({
                    "day": day_num,
                    "term": ae_term,
                    "grade": ae.get("grade"),
                    "status": status,
                    "days_active": ae.get("days_active", 0),
                    "channel": ae.get("channel", ""),
                    "detection_delay": ae.get("detection_delay"),
                    "detected_day": detected_day,
                    "resolved_day": resolved_day,
                    "onset_day": ae.get("onset_day"),
                })

        lab_trends = {}
        for d in visible_days:
            day_num = d.get("day", 0)
            hr = d.get("hospital_record", {}).get("objective", {})
            hr_labs = hr.get("labs", {})
            for lab_name, lab_info in hr_labs.items():
                if lab_name not in lab_trends:
                    lab_trends[lab_name] = []
                val = lab_info.get("value") if isinstance(lab_info, dict) else lab_info
                if val is not None:
                    lab_trends[lab_name].append({
                        "day": day_num, "value": val, "unit": "",
                    })
    else:
        # Ground Truth: full data
        ae_timeline = []
        for d in visible_days:
            day_num = d.get("day", 0)
            for ae in d.get("AE", []):
                ae_timeline.append({
                    "day": day_num,
                    "term": ae.get("AETERM"),
                    "grade": ae.get("_grade"),
                    "status": ae.get("_status"),
                    "days_active": ae.get("_days_active"),
                })

        lab_trends = {}
        for d in visible_days:
            day_num = d.get("day", 0)
            results = d.get("LB", {}).get("results", {})
            for lab_name, lab_val in results.items():
                if lab_name not in lab_trends:
                    lab_trends[lab_name] = []
                lab_trends[lab_name].append({
                    "day": day_num,
                    "value": lab_val.get("LBORRES"),
                    "unit": lab_val.get("LBORRESU", ""),
                })

    # Build event log (memory stream, like generative agents)
    event_log = []
    for d in visible_days:
        day_num = d.get("day", 0)
        hr_data = d.get("hospital_record", {})
        obs_types = hr_data.get("observation_types", [])

        if view_mode == "gt":
            day_events = _extract_day_events(d)
            for evt in day_events:
                event_log.append({
                    "day": day_num,
                    "type": evt["type"],
                    "text": evt["text"],
                    "icon": evt["icon"],
                })
        else:
            # HR mode: only show events on observation days
            if obs_types:
                hr_aes = hr_data.get("objective", {}).get("active_aes", [])
                for ae in hr_aes:
                    event_log.append({
                        "day": day_num,
                        "type": "ae_detected",
                        "text": f"Detected: {ae.get('ae', '?')} Grade {ae.get('grade', '?')} ({ae.get('channel', '')})",
                        "icon": "\u26a0\ufe0f",
                    })
                if "scheduled_visit" in obs_types:
                    event_log.append({
                        "day": day_num, "type": "visit",
                        "text": "Scheduled clinic visit", "icon": "\U0001f3e5",
                    })

        # Care records (visible in both modes -- these are hospital actions)
        for cr in d.get("care_record", []):
            event_log.append({
                "day": day_num,
                "type": "care",
                "text": cr.get("summary", "Video call"),
                "icon": "\U0001f4f9",
            })

    # Format BMI to 1 decimal
    bmi_raw = profile.get("emr", {}).get("demographics", {}).get("bmi", "?")
    bmi_display = f"{bmi_raw:.1f}" if isinstance(bmi_raw, (int, float)) else str(bmi_raw)

    # Build mood trajectory
    mood_trajectory = []
    for d in visible_days:
        m = d.get("mood_state", {})
        if m:
            mood_trajectory.append({"day": d.get("day", 0), **m})

    context = {
        "run_id": run_id,
        "patient_id": patient_id,
        "day": day,
        "mode": mode,
        "view_mode": view_mode,
        "available_modes": avail,
        "available_modes_json": json.dumps(avail),
        "profile": profile,
        "profile_json": json.dumps(profile),
        "current_day": current_day_data,
        "current_day_json": json.dumps(current_day_data),
        "ae_timeline_json": json.dumps(ae_timeline),
        "lab_trends_json": json.dumps(lab_trends),
        "mood_trajectory_json": json.dumps(mood_trajectory),
        "event_log": event_log,
        "total_days": len(all_days),
        "bmi_display": bmi_display,
        "model_name": _load_run_meta(run_path).get("model", ""),
    }
    return render(request, "patient_state/patient_state.html", context)


# --- JSON API ---

def api_run_meta(request, run_id: str):
    """Run metadata: patients, total days, drug info."""
    run_path = _get_run_path(run_id)
    if not run_path.exists():
        return JsonResponse({"error": "not found"}, status=404)

    mode = request.GET.get("mode", None)
    patient_ids = _list_patients(run_path)
    total_days = _count_days(run_path, mode)
    rule_set = _load_rule_set(run_path)

    return JsonResponse({
        "run_id": run_id,
        "patient_ids": patient_ids,
        "total_days": total_days,
        "drug_name": rule_set.get("drug_name", ""),
        "indication": rule_set.get("indication", ""),
        "cycle_length": rule_set.get("trial_design", {}).get(
            "cycle_length_days", 21),
    })


_missing_runs: set = set()

def api_day_data(request, run_id: str, day: int):
    """All patients' data for a specific day -- for AJAX day navigation."""
    if run_id in _missing_runs:
        return JsonResponse({"error": "not found", "stop_polling": True}, status=404)
    run_path = _get_run_path(run_id)
    if not run_path.exists():
        _missing_runs.add(run_id)
        return JsonResponse({"error": "not found", "stop_polling": True}, status=404)

    mode = request.GET.get("mode", "natural")
    view_mode = request.GET.get("view", "hr")
    patient_ids = _list_patients(run_path)
    _warm_cache(run_path, patient_ids, mode)
    patients = []
    all_events = []

    for pid in patient_ids:
        profile = _load_patient_profile(run_path, pid)
        day_data = _load_day_for_patient(run_path, pid, day, mode)
        patients.append(_patient_summary(
            profile, day_data, view_mode, run_path=run_path, mode=mode))
        if day_data:
            all_events.extend(_extract_day_events(day_data))

    severity_order = {"high": 0, "medium": 1, "info": 2, "low": 3}
    all_events.sort(key=lambda e: severity_order.get(e["severity"], 9))

    return JsonResponse({
        "day": day,
        "patients": patients,
        "events": all_events,
    })


def api_patient_timeline(request, run_id: str, patient_id: str):
    """Full timeline for a patient -- for charts and detailed view."""
    run_path = _get_run_path(run_id)
    if not run_path.exists():
        return JsonResponse({"error": "not found"}, status=404)

    mode = request.GET.get("mode", "natural")
    all_days = _load_all_days_for_patient(run_path, patient_id, mode)
    profile = _load_patient_profile(run_path, patient_id)

    return JsonResponse({
        "patient_id": patient_id,
        "profile": profile,
        "days": all_days,
    })


# --- SSE (Server-Sent Events) for auto-play ---

def sse_stream(request, run_id: str):
    """
    Concordia-style SSE endpoint for auto-play mode.
    Client sends speed via query param: ?speed=1 (days per second).
    Streams day data as events.
    """
    run_path = _get_run_path(run_id)
    if not run_path.exists():
        return HttpResponse("Run not found", status=404)

    speed = float(request.GET.get("speed", "1"))
    start_day = int(request.GET.get("start", "1"))
    mode = request.GET.get("mode", "natural")
    view_mode = request.GET.get("view", "hr")
    total_days = _count_days(run_path, mode)
    patient_ids = _list_patients(run_path)
    _warm_cache(run_path, patient_ids, mode)

    def event_stream():
        for day in range(start_day, total_days + 1):
            patients = []
            all_events = []
            for pid in patient_ids:
                profile = _load_patient_profile(run_path, pid)
                day_data = _load_day_for_patient(run_path, pid, day, mode)
                patients.append(_patient_summary(
                    profile, day_data, view_mode,
                    run_path=run_path, mode=mode))
                if day_data:
                    all_events.extend(_extract_day_events(day_data))

            payload = json.dumps({
                "day": day,
                "patients": patients,
                "events": all_events,
            })
            yield f"event: day\ndata: {payload}\n\n"

            if speed > 0:
                time.sleep(1.0 / speed)

        yield "event: done\ndata: {}\n\n"

    response = StreamingHttpResponse(
        event_stream(), content_type="text/event-stream"
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response
