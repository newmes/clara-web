"""Bridge between ClinicalTrialEngine simulation output and Doc Agent CRFData.

Our simulation stores day-numbers (int) and boolean flags. Doc Agent expects
datetime.date objects and "Y"/"N" strings. This adapter converts simulation
JSONL + patient profile into a validated CRFData instance for a single SAE.

Usage:
    from sim.doc_agent.sim_to_crf_adapter import build_crf_for_sae

    crf = build_crf_for_sae(
        patient_profile=patient_dict,
        day_records=list_of_daily_dicts,
        target_ae_term="neutropenia",
        sim_start_date=date(2026, 1, 6),
    )
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Optional

from .schemas.crf import (
    AEDomain,
    CMDomain,
    CMRecord,
    CRFData,
    DADomain,
    DDDomain,
    DMDomain,
    ECDomain,
    InvestigatorInfo,
    LBDomain,
    LBRecord,
    MHDomain,
    MHRecord,
    VSDomain,
    VSRecord,
)

logger = logging.getLogger(__name__)


def _day_to_date(day: Any, start: date) -> Optional[date]:
    """Convert simulation day number to actual date. Returns None for invalid input."""
    if day is None:
        return None
    try:
        return start + timedelta(days=int(day) - 1)
    except (TypeError, ValueError):
        return None


def _bool_to_yn(val: Any) -> str:
    """Convert Python bool / truthy to 'Y' or 'N'."""
    if isinstance(val, str):
        return val.upper() if val.upper() in ("Y", "N") else ("Y" if val else "N")
    return "Y" if val else "N"


def _normalize_severity(sev: str, grade: int) -> str:
    """Map AESEV to the 3-value literal the doc_agent schema accepts."""
    upper = (sev or "").upper()
    if upper in ("MILD", "MODERATE", "SEVERE"):
        return upper
    if grade >= 3 or upper in ("LIFE-THREATENING", "FATAL"):
        return "SEVERE"
    if grade == 2:
        return "MODERATE"
    return "MILD"


def _normalize_aeacn(val: str) -> str:
    """Ensure AEACN is a valid literal value."""
    valid = {
        "DRUG WITHDRAWN", "DRUG INTERRUPTED", "DOSE REDUCED",
        "DOSE INCREASED", "DOSE NOT CHANGED", "UNKNOWN", "NOT APPLICABLE",
    }
    upper = (val or "").upper().strip()
    return upper if upper in valid else "DOSE NOT CHANGED"


def _normalize_aeout(val: str) -> str:
    """Ensure AEOUT is a valid literal value."""
    valid = {
        "RECOVERED/RESOLVED", "RECOVERING/RESOLVING",
        "NOT RECOVERED/NOT RESOLVED", "RECOVERED/RESOLVED WITH SEQUELAE",
        "FATAL", "UNKNOWN",
    }
    upper = (val or "").upper().strip()
    return upper if upper in valid else "NOT RECOVERED/NOT RESOLVED"


def _normalize_sex(sex: str) -> str:
    """Map simulation sex values to the doc_agent literal."""
    s = (sex or "").upper().strip()
    if s in ("M", "MALE"):
        return "Male"
    if s in ("F", "FEMALE"):
        return "Female"
    return "Male"


# ─────────────────────────────────────────────────────────
# Domain Builders
# ─────────────────────────────────────────────────────────

def _build_dm(profile: dict) -> DMDomain:
    """Build Demographics domain from patient profile."""
    dm = profile.get("DM", {})
    emr_demo = profile.get("emr", {}).get("demographics", {})
    return DMDomain(
        SUBJID=profile.get("patient_id", "UNK"),
        AGE=dm.get("AGE") or emr_demo.get("age", 0),
        SEX=_normalize_sex(dm.get("SEX") or emr_demo.get("sex", "")),
        RACE=dm.get("RACE", ""),
        ETHNIC=dm.get("ETHNIC", "NOT REPORTED"),
    )


def _build_ae(
    ae_record: dict,
    start: date,
) -> AEDomain:
    """Build AE domain from a single target SAE record."""
    grade = ae_record.get("_grade", 3)
    grade_str = str(min(grade, 5)) if grade else None

    return AEDomain(
        AETERM=ae_record.get("AETERM", ""),
        AESTDAT=_day_to_date(ae_record.get("AESTDAT"), start) or start,
        AEENDAT=_day_to_date(ae_record.get("AEENDAT"), start),
        AESEV=_normalize_severity(ae_record.get("AESEV", ""), grade),
        AESER=_bool_to_yn(ae_record.get("AESER", grade >= 3)),
        AETOXGR=grade_str,
        AEREL=str(ae_record.get("AEREL", "Possible")),
        AEACN=_normalize_aeacn(ae_record.get("AEACN", "")),
        AEOUT=_normalize_aeout(ae_record.get("AEOUT", "")),
        AESDTH=_bool_to_yn(ae_record.get("AESDTH", False)),
        AESLIFE=_bool_to_yn(ae_record.get("AESLIFE", grade >= 4)),
        AESHOSP=_bool_to_yn(ae_record.get("AESHOSP", grade >= 3)),
        AESDISAB=_bool_to_yn(ae_record.get("AESDISAB", False)),
        AESCONG=_bool_to_yn(ae_record.get("AESCONG", False)),
        AESMIE=_bool_to_yn(ae_record.get("AESMIE", False)),
    )


def _build_ec_list(
    day_records: list[dict],
    start: date,
) -> list[ECDomain]:
    """Aggregate all EC records into exposure periods."""
    periods: dict[str, dict] = {}

    for rec in day_records:
        for ec in rec.get("EC", []):
            drug = ec.get("ECREFID", "study_drug")
            day_num = ec.get("ECSTDAT", rec.get("day", 1))
            dose_txt = ec.get("ECDSTXT", "")
            freq = ec.get("ECDOSFRQ", "")
            route = ec.get("ECROUTE", "INTRAVENOUS")
            dose_adj = ec.get("ECDOSADJ", False)

            if drug not in periods:
                periods[drug] = {
                    "drug": drug,
                    "dose": dose_txt,
                    "freq": freq,
                    "route": route,
                    "start_day": day_num,
                    "end_day": day_num,
                    "adj": None,
                }
            else:
                periods[drug]["end_day"] = day_num
                if dose_adj and not periods[drug]["adj"]:
                    periods[drug]["adj"] = ec.get("ECADJ", "Dose adjustment")

    result: list[ECDomain] = []
    for p in periods.values():
        dose_parts = []
        if p["dose"]:
            dose_parts.append(f"{p['dose']} mg")
        result.append(ECDomain(
            ECDSTXT=" ".join(dose_parts) if dose_parts else "Per protocol",
            ECDOSFRQ=p["freq"],
            ECROUTE=p["route"],
            ECSTDAT=_day_to_date(p["start_day"], start) or start,
            ECENDAT=_day_to_date(p["end_day"], start),
            ECDOSADJ=p["adj"],
        ))

    if not result:
        result.append(ECDomain(
            ECDSTXT="Per protocol",
            ECSTDAT=start,
        ))
    return result


def _build_lb(
    day_records: list[dict],
    ae_onset_day: int,
    start: date,
) -> LBDomain:
    """Collect lab records at key timepoints: baseline(day 1), pre-AE, onset, post-AE, latest."""
    records: list[LBRecord] = []
    seen: set[str] = set()

    last_day = max((r.get("day", 0) for r in day_records), default=1)
    key_days = {1, ae_onset_day - 1, ae_onset_day, ae_onset_day + 7, last_day}
    key_days = {d for d in key_days if d >= 1}

    relevant_days = [
        r for r in day_records if r.get("day", 0) in key_days
    ]

    for rec in relevant_days:
        lb = rec.get("LB", {})
        lab_results = lb.get("results", {})
        day_num = lb.get("LBDAT", rec.get("day", 1))

        for lab_name, lab_data in lab_results.items():
            key = f"{lab_name}_{day_num}"
            if key in seen:
                continue
            seen.add(key)

            if isinstance(lab_data, dict):
                val = lab_data.get("LBORRES")
                unit = lab_data.get("LBORRESU", "")
            else:
                val = lab_data
                unit = ""

            if val is None:
                continue

            records.append(LBRecord(
                LBTESTCD=lab_name.upper().replace("_", ""),
                LBTEST=lab_name.replace("_", " ").title(),
                LBORRES=str(val),
                LBORRESU=unit,
                LBDAT=_day_to_date(day_num, start) or start,
            ))

    return LBDomain(records=records)


def _build_cm(
    day_records: list[dict],
    start: date,
) -> CMDomain:
    """Deduplicate CM records across all days."""
    seen: set[str] = set()
    records: list[CMRecord] = []

    for rec in day_records:
        for cm in rec.get("CM", []):
            name = cm.get("CMTRT", "")
            if not name or name in seen:
                continue
            seen.add(name)

            is_baseline = cm.get("_baseline", False)
            records.append(CMRecord(
                CMTRT=name,
                CMINDC=cm.get("CMINDC", ""),
                CMDSTXT=f"{cm.get('CMDSTXT', '')} {cm.get('CMDOSU', '')}".strip(),
                CMSTDAT=_day_to_date(cm.get("CMSTDAT"), start),
                CMENDAT=_day_to_date(cm.get("CMENDAT"), start),
                CMCAT="BASELINE" if is_baseline else "AE_TREATMENT",
            ))
    return CMDomain(records=records)


def _build_mh(profile: dict) -> MHDomain:
    """Build medical history from patient profile."""
    records: list[MHRecord] = []
    for mh in profile.get("MH", []):
        term = mh.get("MHTERM", "")
        if not term:
            continue
        records.append(MHRecord(
            MHTERM=term,
            MHONGO=_bool_to_yn(mh.get("MHONGO", True)),
        ))
    return MHDomain(records=records)


def _build_vs(
    day_records: list[dict],
    ae_onset_day: int,
    start: date,
    profile: dict,
) -> VSDomain:
    """Build vitals from day closest to AE onset."""
    weight: Optional[float] = None
    height: Optional[float] = None
    records: list[VSRecord] = []

    target = None
    for rec in day_records:
        if rec.get("day", 0) == ae_onset_day:
            target = rec
            break
    if target is None and day_records:
        target = min(day_records, key=lambda r: abs(r.get("day", 0) - ae_onset_day))

    if target:
        vs = target.get("VS", {})
        day_num = vs.get("VSDAT", target.get("day", 1))
        vs_date = _day_to_date(day_num, start) or start

        vital_map = [
            ("SYSBP_VSORRES", "SYSBP", "mmHg"),
            ("DIABP_VSORRES", "DIABP", "mmHg"),
            ("PULSE_VSORRES", "PULSE", "beats/min"),
            ("RESP_VSORRES", "RESP", "breaths/min"),
            ("TEMP_VSORRES", "TEMP", "C"),
        ]
        for src_key, code, unit in vital_map:
            val = vs.get(src_key)
            if val is not None:
                records.append(VSRecord(
                    VSTESTCD=code,
                    VSORRES=str(val),
                    VSORRESU=unit,
                    VSDAT=vs_date,
                ))

        weight = vs.get("WEIGHT_VSORRES")
        height = vs.get("HEIGHT_VSORRES")

    return VSDomain(records=records, WEIGHT=weight, HEIGHT=height)


def _build_dd(
    day_records: list[dict],
    start: date,
) -> DDDomain:
    """Build death details if patient died."""
    for rec in reversed(day_records):
        ds = rec.get("DS") or rec.get("ds_record")
        if ds and ds.get("DSDECOD") == "DEATH":
            dd_data = rec.get("DD", {})
            death_day = dd_data.get("DTHDAT") or ds.get("DSSTDAT") or rec.get("day")
            cause = dd_data.get("PRCDTH_DDORRES") or ds.get("DSTERM", "")
            return DDDomain(
                DTHDAT=_day_to_date(death_day, start),
                PRCDTH=cause,
                AUTOPIND="N",
            )
    return DDDomain()


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────

def find_serious_aes(day_records: list[dict]) -> list[dict]:
    """Scan simulation JSONL records for all AEs with grade >= 3 or AESER=True.

    Returns list of dicts: {ae_record, day, patient_id}.
    """
    results = []
    seen: set[str] = set()

    for rec in day_records:
        for ae in rec.get("AE", []):
            term = ae.get("AETERM", "")
            grade = ae.get("_grade", 0)
            aeser = ae.get("AESER", False)

            if grade >= 3 or aeser:
                key = term
                if key not in seen:
                    seen.add(key)
                    results.append({
                        "ae_record": ae,
                        "day": rec.get("day"),
                        "patient_id": rec.get("patient_id"),
                    })
    return results


def build_crf_for_sae(
    patient_profile: dict,
    day_records: list[dict],
    target_ae_term: str,
    sim_start_date: date = date(2026, 1, 6),
    target_ae_day: Optional[int] = None,
) -> Optional[CRFData]:
    """Convert simulation data into CRFData for a specific SAE.

    Args:
        patient_profile: Patient profile dict (from map_patient_record or raw)
        day_records: All daily JSONL records for this patient (already mapped via crf_mapper)
        target_ae_term: AETERM of the SAE to report
        sim_start_date: Day 1 of the simulation as a calendar date
        target_ae_day: Optional onset day to disambiguate if same AE appears multiple times

    Returns:
        CRFData if the SAE is found, None otherwise
    """
    target_ae = None
    for rec in day_records:
        for ae in rec.get("AE", []):
            term = ae.get("AETERM", "")
            grade = ae.get("_grade", 0)
            onset = ae.get("AESTDAT", rec.get("day"))

            term_match = term.lower().replace("_", " ") == target_ae_term.lower().replace("_", " ")
            rec_day = rec.get("day")
            day_match = target_ae_day is None or onset == target_ae_day or rec_day == target_ae_day

            if term_match and day_match and (grade >= 3 or ae.get("AESER")):
                if target_ae is None or ae.get("_grade", 0) > target_ae.get("_grade", 0):
                    target_ae = ae

    if target_ae is None:
        logger.warning("SAE '%s' not found in patient records", target_ae_term)
        return None

    ae_onset_day = target_ae.get("AESTDAT", 1)

    return CRFData(
        dm=_build_dm(patient_profile),
        ae=_build_ae(target_ae, sim_start_date),
        ec=_build_ec_list(day_records, sim_start_date),
        lb=_build_lb(day_records, ae_onset_day, sim_start_date),
        cm=_build_cm(day_records, sim_start_date),
        mh=_build_mh(patient_profile),
        vs=_build_vs(day_records, ae_onset_day, sim_start_date, patient_profile),
        dd=_build_dd(day_records, sim_start_date),
        da=DADomain(),
        investigator=InvestigatorInfo(),
    )
