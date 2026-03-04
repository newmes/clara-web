"""Context Manager — Daily Agent의 Event Day LLM 호출을 위한 컨텍스트 압축

Day 100이면 100일치 데이터를 전부 넘길 수 없으므로,
patient_summary (누적 요약) + recent_days (최근 N일 전체)로 압축한다.
"""


def compress_history(
    patient_data: dict,
    day_results: list[dict],
    recent_n: int = 5,
) -> dict:
    """일별 결과 리스트를 Daily Agent Event Day LLM 호출용으로 압축한다.

    Args:
        patient_data: Patient Agent output (환자 초기값, 기저값 참조)
        day_results: 지금까지 생성된 Progression output 리스트 (시간순)
        recent_n: 전체 데이터를 넘길 최근 일수

    Returns:
        {"patient_summary": {...}, "recent_days": [...]}
    """
    if not day_results:
        return {
            "patient_summary": _make_initial_summary(patient_data),
            "recent_days": [],
        }

    recent_days = day_results[-recent_n:]

    latest = day_results[-1]
    obj = latest.get("objective", {})

    # active AEs 수집
    active_aes = obj.get("active_aes", [])

    # 누적 투약 정보
    drug_a = obj.get("drug_a", {})
    drug_b = obj.get("drug_b", {})

    # care_record 이력 요약 (전체 기간에서 비어있지 않은 것만)
    care_records = []
    for dr in day_results:
        cr = dr.get("care_record", [])
        if cr:
            care_records.extend(cr)

    # 주요 이벤트 타임라인 (AE onset/resolution, grade changes)
    events = _extract_events(day_results)

    patient_summary = {
        "total_days_elapsed": latest.get("day", 0),
        "current_cycle": latest.get("cycle", 1),
        "current_location": obj.get("location", "HOME"),
        "treatment_status": obj.get("treatment_status", "on_treatment"),
        "active_aes": [
            {"ae": ae["ae"], "grade": ae["grade"], "onset_day": ae["onset_day"], "status": ae["status"]}
            for ae in active_aes
        ],
        "drug_a_summary": {
            "cumulative_dose_mg": drug_a.get("cumulative_dose_mg", 0),
            "dose_level": drug_a.get("dose_level", 1.0),
            "last_administered_day": drug_a.get("last_administered_day"),
            "next_scheduled_day": drug_a.get("next_scheduled_day"),
        },
        "drug_b_cumulative_days": drug_b.get("cumulative_days", 0),
        "latest_labs": obj.get("labs", {}),
        "latest_vitals": obj.get("vitals", {}),
        "care_record_summary": care_records[-10:] if care_records else [],
        "key_events": events[-15:],
    }

    return {
        "patient_summary": patient_summary,
        "recent_days": recent_days,
    }


def _make_initial_summary(patient_data: dict) -> dict:
    """Day 1 이전: 기저값에서 초기 요약 생성."""
    emr = patient_data.get("emr", {})
    return {
        "total_days_elapsed": 0,
        "current_cycle": 1,
        "current_location": patient_data.get("initial_state", {}).get("location", "HOME"),
        "treatment_status": "screening",
        "active_aes": [],
        "drug_a_summary": {
            "cumulative_dose_mg": 0,
            "dose_level": 1.0,
            "last_administered_day": None,
            "next_scheduled_day": 1,
        },
        "drug_b_cumulative_days": 0,
        "latest_labs": emr.get("baseline_labs", {}),
        "latest_vitals": emr.get("baseline_vitals", {}),
        "care_record_summary": [],
        "key_events": [],
    }


def _extract_events(day_results: list[dict]) -> list[dict]:
    """전체 기간에서 주요 이벤트를 추출한다."""
    events = []
    prev_aes = {}

    for dr in day_results:
        day = dr.get("day", 0)
        obj = dr.get("objective", {})

        for ae in obj.get("active_aes", []):
            ae_name = ae["ae"]
            status = ae.get("status", "")
            grade = ae.get("grade", 0)

            if status == "new_onset":
                events.append({"day": day, "event": f"AE onset: {ae_name} G{grade}"})
            elif ae_name in prev_aes and grade != prev_aes[ae_name]:
                direction = "↑" if grade > prev_aes[ae_name] else "↓"
                events.append({"day": day, "event": f"AE grade change: {ae_name} G{prev_aes[ae_name]}→G{grade} {direction}"})
            elif status == "resolved":
                events.append({"day": day, "event": f"AE resolved: {ae_name}"})

            prev_aes[ae_name] = grade

        # care_record events
        for cr in dr.get("care_record", []):
            for action in cr.get("actions", []):
                events.append({
                    "day": day,
                    "event": f"Care action: {action.get('action', '?')} - {action.get('detail', '')}",
                })

        # treatment status changes
        ts = obj.get("treatment_status", "")
        if ts in ("held", "discontinued") and events and "treatment" not in events[-1].get("event", ""):
            events.append({"day": day, "event": f"Treatment: {ts}"})

    return events
