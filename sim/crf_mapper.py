"""CDASH CRF Mapper — 시뮬레이션 내부 표현 → CDASH 표준 필드 변환.

data_to_generate/json/*.json 스키마 기준으로 필드명을 매핑한다.
내부 로직은 기존 필드명(ae, SBP 등)을 유지하고, 출력 시점에서 일괄 변환.

사용:
    from sim.crf_mapper import map_day_record, map_patient_record
    cdash_record = map_day_record(raw_record, patient_data)
"""

from __future__ import annotations

import math
import re
from typing import Any


def _safe_float(val) -> float | None:
    """LLM이 '65.0 (kg)' 같은 문자열을 반환할 때 숫자만 추출."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        m = re.search(r'-?\d+\.?\d*', val)
        return float(m.group()) if m else None
    return None


# ══════════════════════════════════════════════════════
# A. ROUTE / DOSFRM 코드 매핑
# ══════════════════════════════════════════════════════

_ROUTE_MAP = {
    "PO": "ORAL", "ORAL": "ORAL",
    "IV": "INTRAVENOUS", "INTRAVENOUS": "INTRAVENOUS",
    "SC": "SUBCUTANEOUS", "SUBCUTANEOUS": "SUBCUTANEOUS",
    "IM": "INTRAMUSCULAR", "INTRAMUSCULAR": "INTRAMUSCULAR",
    "TOPICAL": "TOPICAL", "TOP": "TOPICAL",
    "RECTAL": "RECTAL",
    "NASAL": "NASAL",
    "RESPIRATORY (INHALATION)": "RESPIRATORY (INHALATION)",
    "INHALATION": "RESPIRATORY (INHALATION)",
    "TRANSDERMAL": "TRANSDERMAL",
    "VAGINAL": "VAGINAL",
    "N/A": "OTHER",
}

_DOSFRM_MAP = {
    "ORAL": "TABLET",
    "INTRAVENOUS": "INJECTION",
    "SUBCUTANEOUS": "INJECTION",
    "INTRAMUSCULAR": "INJECTION",
    "TOPICAL": "LIQUID",
    "RESPIRATORY (INHALATION)": "INHALANT",
    "TRANSDERMAL": "PATCH",
    "OTHER": "OTHER",
}

# Lab category classification
_LAB_CATEGORY = {
    "hemoglobin": "HEMATOLOGY", "ANC": "HEMATOLOGY", "platelets": "HEMATOLOGY",
    "creatinine": "CHEMISTRY", "eGFR": "CHEMISTRY", "alt": "CHEMISTRY",
    "ast": "CHEMISTRY", "total_bilirubin": "CHEMISTRY", "albumin": "CHEMISTRY",
    "sodium": "CHEMISTRY", "LDH": "CHEMISTRY",
    "glucose_fasting": "CHEMISTRY", "HbA1c": "CHEMISTRY",
    "TSH": "ENDOCRINE",
}


# ══════════════════════════════════════════════════════
# B. 환자 레코드 변환 (1회, DM/MH)
# ══════════════════════════════════════════════════════

def map_patient_record(patient: dict) -> dict:
    """환자 데이터를 CDASH DM + MH 형식으로 변환한다.

    Returns:
        {
            "patient_id": str,
            "DM": {...},
            "MH": [...],
            "emr": {...},   # 원본 유지 (시뮬레이션 내부용)
            "persona": {...},
        }
    """
    emr = patient.get("emr", {})
    demo = emr.get("demographics", {})
    mh_list = emr.get("medical_history", [])

    # ── DM (Demographics) ──
    normalized_race = _normalize_race(demo.get("race", ""))
    dm = {
        "AGE": demo.get("age"),
        "AGEU": "YEARS",
        "SEX": demo.get("sex"),
        "RACE": normalized_race,
        "RACEOTH": demo.get("race", "") if normalized_race == "OTHER" else "",
        "ETHNIC": "NOT REPORTED",  # 시뮬레이션에서 별도 생성 안 함
    }

    # ── MH (Medical History) ──
    mh_records = []
    for item in mh_list:
        mh_rec = {
            "MHYN": True,
            "MHDAT": None,  # 수집일 (스크리닝)
            "MHTERM": item.get("condition", ""),
            "MHSTDAT": None,  # 정확한 시작일 없음 (baseline)
            "MHONGO": item.get("ongoing", True),
            "MHENDAT": None,
        }
        mh_records.append(mh_rec)

    return {
        "patient_id": patient.get("patient_id"),
        "DM": dm,
        "MH": mh_records,
        "emr": emr,  # 내부용 원본
        "persona": patient.get("persona"),
        "initial_state": patient.get("initial_state"),
    }


def _normalize_race(race: str) -> str:
    """race 값을 CDASH codelist로 정규화."""
    r = race.upper().strip()
    if "WHITE" in r or "CAUCASIAN" in r:
        return "WHITE"
    elif "BLACK" in r or "AFRICAN" in r:
        return "BLACK OR AFRICAN AMERICAN"
    elif "ASIAN" in r:
        return "ASIAN"
    elif "NATIVE" in r and "HAWAIIAN" in r:
        return "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER"
    elif "INDIAN" in r or "ALASKA" in r:
        return "AMERICAN INDIAN OR ALASKA NATIVE"
    elif r in ("NOT REPORTED", "UNKNOWN", "OTHER", ""):
        return r or "NOT REPORTED"
    return "OTHER"


# ══════════════════════════════════════════════════════
# C. 일별 레코드 변환 (매일)
# ══════════════════════════════════════════════════════

def map_day_record(record: dict, patient: dict | None = None) -> dict:
    """1일치 시뮬레이션 레코드를 CDASH 도메인 구조로 변환.

    Args:
        record: daily_agent 출력 (내부 필드명)
        patient: 환자 데이터 (DM/MH/baseline용, optional)

    Returns:
        CDASH 표준 필드명으로 변환된 레코드
    """
    obj = record.get("objective", {})
    day = record.get("day", 0)

    mapped = {
        "patient_id": record.get("patient_id"),
        "day": day,
        "cycle": record.get("cycle"),
        "cycle_day": record.get("cycle_day"),
    }

    # ── 사망/약물 상태 추출 (AE 매핑에 필요) ──
    location = obj.get("location", "HOME")
    is_deceased = location == "DECEASED"
    ds = record.get("ds_record")
    death_cause = ds.get("DSTERM", "") if ds and ds.get("DSDECOD") == "DEATH" else ""

    # 약물 상태: objective에서 동적 키로 추출
    held_drugs: set[str] = set()
    discontinued_drugs: set[str] = set()
    dose_reduced_drugs: set[str] = set()
    for key, val in obj.items():
        if isinstance(val, dict) and "cumulative_dose_mg" in val:
            if val.get("treatment_held"):
                held_drugs.add(key)
            if val.get("treatment_discontinued"):
                discontinued_drugs.add(key)
            if val.get("dose_level", 1.0) < 1.0:
                dose_reduced_drugs.add(key)

    # ── 각 도메인 변환 ──
    mapped["AE"] = _map_ae(
        obj.get("active_aes", []), day,
        is_deceased=is_deceased, death_cause=death_cause,
        held_drugs=held_drugs, discontinued_drugs=discontinued_drugs,
        dose_reduced_drugs=dose_reduced_drugs,
    )
    mapped["EC"] = _map_ec(record.get("ec_records", []), day)
    mapped["CM"] = _map_cm(record.get("cm_records", []))
    mapped["VS"] = _map_vs(obj.get("vitals", {}), day, patient)
    mapped["LB"] = _map_lb(obj.get("labs", {}), day)
    mapped["DS"] = _map_ds(record.get("ds_record"), day)
    mapped["RS"] = _map_rs(record.get("recist_scan"), record.get("recist_history"), day)
    mapped["TU"] = _map_tu(record.get("recist_scan"), patient, day)
    mapped["DD"] = _map_dd(record.get("ds_record"), day)
    mapped["PE"] = _map_pe(record, day)
    mapped["EG"] = _map_eg(day)

    # ── 치료 상태 (objective 구조 유지) ──
    mapped["objective"] = _map_objective(obj, record)

    # ── subjective (Care AI용 — CDASH 외) ──
    mapped["subjective"] = record.get("subjective")

    # ── 시뮬레이션 내부 메타 (underscore prefix) ──
    mapped["_sim"] = {
        "generation_mode": record.get("_generation_mode"),
        "events": record.get("_events"),
        "mortality_risk": record.get("_mortality_risk"),
        "mortality_channels": record.get("_mortality_channels"),
        "discontinuation_risks": record.get("_discontinuation_risks"),
    }

    # care_record (Care AI용)
    mapped["care_record"] = record.get("care_record", [])

    # ── Observation Model 결과 (GT/HR 분리) ──
    if record.get("hospital_record") is not None:
        mapped["hospital_record"] = record["hospital_record"]
    if record.get("observation_events") is not None:
        mapped["observation_events"] = record["observation_events"]
    if record.get("mood_state") is not None:
        mapped["mood_state"] = record["mood_state"]

    return mapped


# ── AE Domain ──────────────────────────────────────

def _map_ae(active_aes: list[dict], day: int,
            is_deceased: bool = False, death_cause: str = "",
            held_drugs: set | None = None,
            discontinued_drugs: set | None = None,
            dose_reduced_drugs: set | None = None) -> list[dict]:
    """active_aes → CDASH AE 레코드.

    Args:
        active_aes: 활성 AE 목록
        day: 현재 시뮬레이션 일수
        is_deceased: 이 날 환자가 사망했는가
        death_cause: 사인 ("treatment_toxicity" 등)
        held_drugs: 현재 보류 중인 약물 세트
        discontinued_drugs: 중단된 약물 세트
        dose_reduced_drugs: 감량된 약물 세트
    """
    if not active_aes:
        return []

    held_drugs = held_drugs or set()
    discontinued_drugs = discontinued_drugs or set()
    dose_reduced_drugs = dose_reduced_drugs or set()

    records = []
    for ae in active_aes:
        status = ae.get("status", "active_stable")
        is_resolved = status == "resolved"
        grade = ae.get("grade", 1)
        ae_term = ae.get("ae", ae.get("AETERM", ""))

        # ── AEACN: 용량 조절 반영 (올바른 귀인) ──
        # AEACN은 "이 특정 AE 때문에 취해진 조치"를 나타냄
        # daily_agent에서 ae_dose_actions로 설정된 값이 있으면 우선 사용
        aeacn = ae.get("AEACN")
        if aeacn is None:
            # AEACN이 설정되지 않았으면 기본값 (이 AE가 조치를 야기하지 않았음)
            aeacn = "DOSE NOT CHANGED"

        # ── AEOUT: 결과 반영 ──
        aeout = ae.get("AEOUT")
        if aeout is None:
            if is_deceased and death_cause == "treatment_toxicity":
                aeout = "FATAL"
            elif is_resolved:
                aeout = "RECOVERED/RESOLVED"
            elif grade <= ae.get("peak_grade", grade) and ae.get("days_active", 0) > 0:
                if is_resolved:
                    aeout = "RECOVERED/RESOLVED"
                else:
                    aeout = "NOT RECOVERED/NOT RESOLVED"
            else:
                aeout = "NOT RECOVERED/NOT RESOLVED"

        # ── AESDTH: 사망 관련 ──
        aesdth = ae.get("AESDTH")
        if aesdth is None:
            if grade >= 5:
                aesdth = True
            elif is_deceased and death_cause == "treatment_toxicity" and grade >= 4:
                aesdth = True
            else:
                aesdth = False

        # ── AEENDAT ──
        aeendat = ae.get("resolved_day", ae.get("AEENDAT"))
        if is_resolved and aeendat is None:
            aeendat = day

        rec = {
            # core fields
            "AETERM": ae_term,
            "AESTDAT": ae.get("onset_day", ae.get("AESTDAT")),
            "AEONGO": not is_resolved,
            "AEENDAT": aeendat,
            "AESEV": ae.get("AESEV", _grade_to_severity(grade)),
            "AESER": ae.get("AESER", grade >= 3),
            "AEREL": ae.get("AEREL", True),
            "AEACN": aeacn,
            "AEACNOTH": ae.get("AEACNOTH"),
            "AEOUT": aeout,
            "AESDTH": aesdth,
            "AESLIFE": ae.get("AESLIFE", grade >= 4),
            "AESHOSP": ae.get("AESHOSP", grade >= 3),
            "AESDISAB": ae.get("AESDISAB", False),
            "AESCONG": ae.get("AESCONG", False),
            "AESMIE": ae.get("AESMIE", False),
            # metadata
            "AEYN": True,
            # 시뮬레이션 추가 필드 (CDASH 외)
            "_grade": grade,
            "_status": status,
            "_days_active": ae.get("days_active", 0),
            "_visual": ae.get("visual"),
        }
        records.append(rec)
    return records


def _grade_to_severity(grade: int) -> str:
    """CTCAE Grade → AESEV 매핑 (CDISC 표준)."""
    return {
        1: "MILD",
        2: "MODERATE",
        3: "SEVERE",
        4: "LIFE-THREATENING",
        5: "FATAL",
    }.get(grade, "SEVERE")


# ── EC Domain ──────────────────────────────────────

def _map_ec(ec_records: list[dict], day: int) -> list[dict]:
    """ec_records → CDASH EC 레코드."""
    if not ec_records:
        return []

    records = []
    for ec in ec_records:
        dose_unit = ec.get("ECDOSU", "mg")
        # mg/kg는 CDASH codelist에 없으므로 그대로 유지 (OTHER 처리)
        ecdosu = dose_unit if dose_unit in ("mg", "g", "ug", "mL", "IU", "CAPSULE", "TABLET", "PUFF") else dose_unit

        rec = {
            "ECREFID": ec.get("drug_name", ec.get("ECREFID", "")),
            "ECSTDAT": ec.get("ECSTDAT", day),
            "ECENDAT": ec.get("ECSTDAT", day),  # 당일 투여이므로 동일
            "ECDSTXT": str(ec.get("ECDSTXT", "")),
            "ECDOSU": ecdosu,
            "ECDOSFRQ": ec.get("ECDOSFRQ", ""),
            "ECROUTE": ec.get("ECROUTE", "INTRAVENOUS"),
            "ECDOSADJ": ec.get("ECDOSADJ", False),
            "ECADJ": ec.get("ECADJ", ""),
            "ECTRTCMP": ec.get("ECTRTCMP", True),
            # 중단 관련
            "ECCINTD": None,
            "ECCINTDU": None,
            # metadata
            "ECITRPYN": ec.get("ECITRPYN", False),
            # 시뮬레이션 추가
            "_dose_mg": ec.get("dose_mg"),
            "_cumulative_dose_mg": ec.get("cumulative_dose_mg"),
            "_dose_level": ec.get("dose_level"),
        }

        # hold인 경우 중단 기간 계산
        if ec.get("hold_reason"):
            rec["ECITRPYN"] = True
            rec["ECADJ"] = ec.get("hold_reason", "")

        records.append(rec)
    return records


# ── CM Domain ──────────────────────────────────────

def _map_cm(cm_records: list[dict]) -> list[dict]:
    """cm_records → CDASH CM 레코드."""
    if not cm_records:
        return []

    records = []
    for cm in cm_records:
        # CMDOSE "50mg" → CMDSTXT "50" + CMDOSU "mg" 분리
        cmdstxt, cmdosu = _split_dose(
            cm.get("CMDSTXT", cm.get("CMDOSE", "")),
            cm.get("CMDOSU", ""),
        )

        # CMROUTE codelist 정규화
        raw_route = cm.get("CMROUTE", "OTHER")
        cmroute = _ROUTE_MAP.get(raw_route.upper(), raw_route)

        # CMDOSFRM (제형) 추론
        cmdosfrm = _DOSFRM_MAP.get(cmroute, "OTHER")

        rec = {
            "CMTRT": cm.get("CMTRT", ""),
            "CMINDC": cm.get("CMINDC", ""),
            "CMDSTXT": cmdstxt,
            "CMDOSU": cmdosu,
            "DOSUO": "",  # 용량 단위 기타
            "CMDOSFRM": cmdosfrm,
            "DOSFRMO": "",  # 제형 기타
            "CMDOSFRQ": cm.get("CMDOSFRQ", ""),
            "DOSFRQO": "",  # 투약 빈도 기타
            "CMROUTE": cmroute,
            "ROUTEO": "",  # 투여 경로 기타
            "CMSTDAT": cm.get("CMSTDAT"),
            "CMONGO": cm.get("CMONGO", True),
            "CMENDAT": cm.get("CMENDAT"),
            # metadata
            "CMYN": True,
            # 시뮬레이션 내부
            "_baseline": cm.get("_baseline", False),
        }
        records.append(rec)
    return records


def _split_dose(dose_str: str, unit_hint: str) -> tuple[str, str]:
    """'50mg' → ('50', 'mg'), '1' + 'mg/kg' → ('1', 'mg/kg')"""
    if unit_hint:
        # 이미 분리되어 있으면 그대로
        return str(dose_str), unit_hint

    s = str(dose_str).strip()
    if not s:
        return "", ""

    # 숫자 + 단위 분리: "50mg", "200.5mg", "1mg/kg"
    m = re.match(r"^([\d.]+)\s*(.*)$", s)
    if m:
        return m.group(1), m.group(2) or "mg"
    return s, ""


# ── VS Domain ──────────────────────────────────────

def _map_vs(vitals: dict, day: int, patient: dict | None = None) -> dict:
    """vitals → CDASH VS 레코드."""
    if not vitals:
        return {}

    # HEIGHT: baseline weight와 BMI에서 역산 (고정값 — 당일 weight 아닌 baseline 사용)
    height = None
    if patient:
        emr = patient.get("emr", {})
        bmi = _safe_float(emr.get("demographics", {}).get("bmi"))
        baseline_weight = _safe_float(emr.get("baseline_vitals", {}).get("weight_kg"))
        if bmi and baseline_weight and bmi > 0:
            height_m = math.sqrt(baseline_weight / bmi)
            height = round(height_m * 100, 1)  # cm

    rec = {
        # core fields
        "VSPERF": True,
        "VSDAT": day,
        "SYSBP_VSORRES": vitals.get("SBP"),
        "DIABP_VSORRES": vitals.get("DBP"),
        "PULSE_VSORRES": vitals.get("HR"),
        "RESP_VSORRES": vitals.get("RR"),
        "TEMP_VSORRES": vitals.get("BT"),
        "WEIGHT_VSORRES": vitals.get("weight_kg"),
        "HEIGHT_VSORRES": height,
        # metadata — 단위
        "SYSBP_VSORRESU": "mmHg",
        "DIABP_VSORRESU": "mmHg",
        "PULSE_VSORRESU": "beats/min",
        "RESP_VSORRESU": "breaths/min",
        "TEMP_VSORRESU": "C",
        "WEIGHT_VSORRESU": "kg",
        "HEIGHT_VSORRESU": "cm",
        # metadata — 측정 조건 (기본값)
        "BP_VSPOS": "SITTING",
        "BP_VSLOC": "BRACHIAL ARTERY",
        "PULSE_VSLOC": "PERIPHERAL ARTERY",
        "TEMP_VSLOC": "ORAL CAVITY",
        # 시뮬레이션 추가 (SpO2 — CDASH VS에 없으나 임상 필수)
        "_SpO2": vitals.get("SpO2"),
        "_SpO2_unit": "%",
    }
    return rec


# ── LB Domain ──────────────────────────────────────

def _map_lb(labs: dict, day: int) -> dict:
    """labs → CDASH LB 레코드."""
    if not labs:
        return {}

    rec = {
        "LBCAT": "CHEMISTRY",  # 전체 분류 (개별 항목별로도 results 내에 포함)
        "LBPERF": True,
        "LBDAT": day,
        "LBTIM": None,
        "results": {},
    }

    for lab_name, lab_data in labs.items():
        if isinstance(lab_data, dict):
            val = lab_data.get("value")
            unit = lab_data.get("unit", "")
            trend = lab_data.get("trend", "stable")
        elif isinstance(lab_data, (int, float)):
            val = lab_data
            unit = ""
            trend = "stable"
        else:
            continue

        entry = {
            "LBORRES": val,
            "LBORRESU": unit,
            "LBCAT": _LAB_CATEGORY.get(lab_name, "OTHER"),
            "_trend": trend,
        }
        if isinstance(lab_data, dict) and lab_data.get("LBBLFL") == "Y":
            entry["LBBLFL"] = "Y"
        elif day == 0:
            entry["LBBLFL"] = "Y"
        rec["results"][lab_name] = entry

    return rec


# ── DS Domain ──────────────────────────────────────

def _map_ds(ds_record: dict | None, day: int) -> dict | None:
    """ds_record → CDASH DS 레코드."""
    if not ds_record:
        return None

    return {
        "DSDECOD": ds_record.get("DSDECOD", ""),
        "DSTERM": ds_record.get("DSTERM", ""),
        "DSSTDAT": ds_record.get("DSSTDTC", ds_record.get("DSSTDAT", day)),
        # 시뮬레이션 추가
        "_mortality_channels": ds_record.get("mortality_channels"),
    }


# ── RS Domain (Disease Response) ──────────────────

def _map_rs(recist_scan: dict | None, recist_history: list | None, day: int) -> dict | None:
    """recist_scan → CDASH RS 레코드."""
    if not recist_scan:
        return None

    cat = recist_scan.get("recist_category", "NE")

    # NTRGRESP: 비표적병변 반응 (간단 파생 — 표적 기반 추론)
    if cat == "CR":
        ntrgresp = "CR"
    elif cat == "PD":
        ntrgresp = "PD"
    else:
        ntrgresp = "NON-CR/NON-PD"

    # OVRLRESP: 전체 반응 (RECIST 1.1 기준)
    ovrlresp = cat  # 단순화: 표적 = 전체 (신규 병변 없다는 가정)

    # BESTRESP: 누적 최적 반응
    bestresp = cat
    if recist_history:
        response_rank = {"CR": 4, "PR": 3, "SD": 2, "NE": 1, "PD": 0}
        best_rank = 0
        for h in recist_history:
            r = response_rank.get(h.get("recist_category", "NE"), 1)
            if r > best_rank:
                best_rank = r
                bestresp = h.get("recist_category", "NE")

    return {
        "RSCAT": "RECIST 1.1",
        "RSPERF": True,
        "RSREASND": "",  # 미수행 사유 (수행했으므로 빈값)
        "RSEVAL": "INVESTIGATOR",
        "RSEVALID": "RADIOLOGIST 1",
        "TRGRESP_RSORRES": cat,
        "NTRGRESP_RSORRES": ntrgresp,
        "OVRLRESP_RSORRES": ovrlresp,
        "BESTRESP_RSORRES": bestresp,
        # 시뮬레이션 추가
        "_tumor_change_pct": recist_scan.get("tumor_change_pct"),
        "_nadir_pct": recist_scan.get("nadir_pct"),
        "_description": recist_scan.get("description"),
    }


# ── TU/TR Domain (Tumor Identification/Results) ──

def _map_tu(recist_scan: dict | None, patient: dict | None, day: int) -> list[dict] | None:
    """RECIST 스캔일에 병변별 TU/TR 레코드 생성."""
    if not recist_scan:
        return None

    lesions = []
    if patient:
        emr = patient.get("emr", {})
        for key_path in [
            emr.get("diagnosis", {}),
            emr.get("baseline_tumor", {}),
        ]:
            tl = key_path.get("target_lesions", [])
            if tl:
                lesions = tl
                break

    if not lesions:
        lesions = [{"site": "UNKNOWN", "size_mm": 20.0}]

    pct = recist_scan.get("tumor_change_pct", 0)
    records = []
    for i, lesion in enumerate(lesions, 1):
        baseline_mm = lesion.get("diameter_mm",
                      lesion.get("size_mm",
                      lesion.get("baseline_mm", 20.0)))
        current_mm = round(baseline_mm * (1 + pct / 100), 1)
        site = lesion.get("tumor_site",
               lesion.get("site",
               lesion.get("location", "UNKNOWN")))

        rec = {
            "TULNKID": f"T{i:02d}",
            "TULOC": site.upper() if site else "UNKNOWN",
            "TULAT": lesion.get("laterality", ""),
            "TUDIR": "",
            "TULOCDTL": lesion.get("detail", ""),
            "TUMETHOD": "CT SCAN",
            "TUDAT": day,
            "TUEVAL": "INVESTIGATOR",
            "TUEVALID": "RADIOLOGIST 1",
            "TRORRES": str(current_mm),
            "TRORRESU": "mm",
            "TRSTAT": "",
            "TRREASND": "",
            "TUYN": True,
            # 시뮬레이션 추가
            "_baseline_mm": baseline_mm,
            "_change_pct": pct,
        }
        records.append(rec)

    return records


# ── DD Domain (Death Details) ──────────────────────

_DEATH_CAUSE_MAP = {
    "disease_progression": "DISEASE UNDER STUDY",
    "treatment_toxicity": "ADVERSE EVENT",
    "clinical_deterioration": "CLINICAL DETERIORATION",
    "infection": "INFECTION",
    "cardiac": "CARDIAC DISORDER",
    "other": "OTHER",
}


def _map_dd(ds_record: dict | None, day: int) -> dict | None:
    """사망 시 DD (Death Details) 레코드."""
    if not ds_record or ds_record.get("DSDECOD") != "DEATH":
        return None

    dsterm = ds_record.get("DSTERM", "")

    cause = None
    if "primary cause:" in dsterm:
        cause = dsterm.split("primary cause:")[-1].strip().rstrip(")")

    if not cause or cause == "UNKNOWN":
        cause = _DEATH_CAUSE_MAP.get(dsterm)

    if not cause:
        channels = ds_record.get("mortality_channels") or {}
        scored = {k: v for k, v in channels.items() if not k.startswith("_")}
        if scored:
            top_channel = max(scored, key=scored.get)
            cause = _DEATH_CAUSE_MAP.get(top_channel, top_channel.upper().replace("_", " "))

    if not cause:
        cause = "UNKNOWN"

    relatedness = "NOT RELATED"
    if dsterm == "treatment_toxicity":
        relatedness = "RELATED"
    elif dsterm == "clinical_deterioration":
        relatedness = "POSSIBLY RELATED"

    return {
        "DDYN": True,
        "DDDAT": day,
        "DTHDAT": day,
        "PRCDTH_DDORRES": cause,
        "DTHRELD_DDORRES": relatedness,
        "AUTOPIND_DDORRES": False,
    }


# ── PE Domain (Physical Examination) ──────────────

def _map_pe(record: dict, day: int) -> dict | None:
    """투약일(OUTPATIENT/INPATIENT)에만 PE 수행."""
    location = record.get("objective", {}).get("location", "HOME")
    if location in ("OUTPATIENT", "INPATIENT"):
        return {
            "PEPERF": True,
            "PEDAT": day,
        }
    return None


# ── EG Domain (ECG) ──────────────────────────────

def _map_eg(day: int) -> dict | None:
    """Day 1 (스크리닝)과 이후 cycle Day 1에만 ECG 수행.
    간단한 규칙: Day 1 또는 21의 배수 + 1일."""
    if day in (0, 1):
        return {
            "EGPERF": True,
            "EGREFID": "",
            "EGMETHOD": "12 LEAD STANDARD",
            "EGPOS": "SUPINE",
            "EGDAT": day,
        }
    return None


# ── Objective (치료 상태) 변환 ──────────────────────

def _map_objective(obj: dict, record: dict) -> dict:
    """objective 구조를 정리 — 약물 추적, location, treatment_status."""
    mapped_obj = {
        "location": obj.get("location", "HOME"),
        "treatment_status": obj.get("treatment_status", "on_treatment"),
        "ecog": obj.get("ecog"),
        "tumor": obj.get("tumor"),
    }

    # 약물별 추적 데이터 (동적 키)
    for key, val in obj.items():
        if isinstance(val, dict) and "cumulative_dose_mg" in val:
            # 약물 추적 dict
            mapped_obj[key] = val

    return mapped_obj