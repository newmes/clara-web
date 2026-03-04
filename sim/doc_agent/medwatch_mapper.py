"""Map CRF domain data to MedWatch FDA Form 3500A fields."""
from __future__ import annotations

from datetime import date
from typing import Optional
from uuid import uuid4

from .config import Settings
from .schemas.agent_output import SentinelOutput
from .schemas.crf import CMRecord, CRFData
from .schemas.medwatch import (
    MedWatch3500A,
    SectionA,
    SectionB,
    SectionC,
    SectionE,
    SectionG,
)


def _format_dose_freq_route(crf: CRFData) -> str:
    """Combine first exposure record (dose at AE onset) into C2 dose/frequency/route string."""
    if not crf.ec:
        return ""
    first = crf.ec[0]
    parts = [first.ECDSTXT]
    if first.ECDOSFRQ:
        parts.append(first.ECDOSFRQ)
    if first.ECROUTE:
        parts.append(first.ECROUTE)
    return ", ".join(parts)


def _format_therapy_dates(crf: CRFData) -> tuple[Optional[date], Optional[date]]:
    """Return earliest start and latest end across all exposure periods."""
    if not crf.ec:
        return None, None
    start = min(ec.ECSTDAT for ec in crf.ec)
    ends = [ec.ECENDAT for ec in crf.ec if ec.ECENDAT is not None]
    end = max(ends) if ends else None
    return start, end


def _format_lab_data(crf: CRFData) -> str:
    """Summarize lab records into B6 text, grouped by date."""
    if not crf.lb.records:
        return ""
    from collections import OrderedDict
    by_date: OrderedDict[str, list[str]] = OrderedDict()
    for r in crf.lb.records:
        if r.LBTESTCD and r.LBTESTCD.upper() == "REQUIREDLABS":
            continue
        ref = ""
        if r.LBORNRLO and r.LBORNRHI:
            ref = f" (ref {r.LBORNRLO}-{r.LBORNRHI})"
        unit = f" {r.LBORRESU}" if r.LBORRESU else ""
        date_key = str(r.LBDAT) if r.LBDAT else "Unknown"
        by_date.setdefault(date_key, []).append(
            f"{r.LBTESTCD}: {r.LBORRES}{unit}{ref}"
        )
    sections: list[str] = []
    for dt, items in by_date.items():
        sections.append(f"[{dt}]\n" + "\n".join(items))
    return "\n\n".join(sections)


def classify_cm_records(
    crf: CRFData,
) -> tuple[list[CMRecord], list[CMRecord]]:
    """Classify CM records into baseline vs AE treatment medications.

    Classification priority:
    1. If CMCAT is explicitly set, use it directly.
    2. If CMCAT is None, apply heuristics:
       a. CMSTDAT >= AESTDAT → AE_TREATMENT
       b. CMINDC contains AETERM token → AE_TREATMENT
       c. Otherwise → BASELINE

    Returns:
        (baseline_meds, ae_treatment_meds)
    """
    baseline: list[CMRecord] = []
    ae_treatment: list[CMRecord] = []

    ae_start = crf.ae.AESTDAT
    ae_term_lower = crf.ae.AETERM.lower()
    ae_tokens = set(ae_term_lower.split())

    for rec in crf.cm.records:
        # Priority 1: Explicit CMCAT
        if rec.CMCAT == "AE_TREATMENT":
            ae_treatment.append(rec)
            continue
        if rec.CMCAT == "BASELINE":
            baseline.append(rec)
            continue

        # Priority 2: Heuristics (CMCAT is None)
        # 2a: Start date on or after AE onset
        if rec.CMSTDAT is not None and rec.CMSTDAT >= ae_start:
            ae_treatment.append(rec)
            continue

        # 2b: Indication contains AE term tokens
        if rec.CMINDC:
            indc_lower = rec.CMINDC.lower()
            if any(token in indc_lower for token in ae_tokens if len(token) > 3):
                ae_treatment.append(rec)
                continue

        # 2c: Default → BASELINE
        baseline.append(rec)

    return baseline, ae_treatment


def _format_concomitant_meds(baseline_meds: list[CMRecord]) -> str:
    """Summarize baseline concomitant medications into C9 text.

    Only baseline medications are included in C9.
    AE treatment medications are excluded (they appear in B5 narrative).
    """
    if not baseline_meds:
        return ""
    parts: list[str] = []
    for r in baseline_meds:
        entry = r.CMTRT
        if r.CMINDC:
            entry += f" ({r.CMINDC})"
        parts.append(entry)
    return "; ".join(parts)


def _format_medical_history(crf: CRFData) -> str:
    """Summarize medical history into B7 text."""
    if not crf.mh.records:
        return ""
    parts: list[str] = []
    for r in crf.mh.records:
        entry = r.MHTERM
        if r.MHONGO == "Y":
            entry += " (ongoing)"
        parts.append(entry)
    return "; ".join(parts)


_AEOUT_TO_OUTCOME = {
    "RECOVERED/RESOLVED": "Recovered/resolved",
    "RECOVERING/RESOLVING": "Recovering/resolving",
    "NOT RECOVERED/NOT RESOLVED": "Not recovered/not resolved",
    "RECOVERED/RESOLVED WITH SEQUELAE": "Recovered/resolved with sequelae",
    "FATAL": "Fatal",
    "UNKNOWN": "Unknown",
}


def _derive_indication(crf: CRFData, fallback: str) -> str:
    """Derive drug indication from medical history cancer diagnosis."""
    for r in crf.mh.records:
        term_lower = r.MHTERM.lower()
        if "cancer" in term_lower or "carcinoma" in term_lower:
            return r.MHTERM
    return fallback


def _generate_report_number(protocol: str) -> str:
    """Auto-generate a unique manufacturer report number."""
    short_id = uuid4().hex[:8].upper()
    return f"{protocol}-{short_id}"


def map_crf_to_medwatch(
    crf: CRFData,
    settings: Settings,
    sentinel_output: Optional[SentinelOutput] = None,
) -> MedWatch3500A:
    """Map CRF domain data + protocol settings to MedWatch 3500A form.

    Fields B5 (narrative), C7 (dechallenge), and C8 (rechallenge) are left
    as empty strings for the AI to fill via structured generation.
    """
    # --- Section A: Patient Information ---
    section_a = SectionA(
        patient_id=crf.dm.SUBJID,
        age=crf.dm.AGE,
        sex=crf.dm.SEX,
        weight=crf.vs.WEIGHT,
        ethnicity=crf.dm.ETHNIC,
        race=crf.dm.RACE,
    )

    # --- Section B: Adverse Event ---
    therapy_start, _ = _format_therapy_dates(crf)
    section_b = SectionB(
        report_type="Adverse Event",
        seriousness_death=crf.ae.AESDTH == "Y",
        seriousness_life_threatening=crf.ae.AESLIFE == "Y",
        seriousness_hospitalization=crf.ae.AESHOSP == "Y",
        hospitalization_start=crf.ae.AEHOSPSTDAT,
        hospitalization_end=crf.ae.AEHOSPENDAT,
        seriousness_disability=crf.ae.AESDISAB == "Y",
        seriousness_congenital=crf.ae.AESCONG == "Y",
        seriousness_other=crf.ae.AESMIE == "Y",
        death_date=crf.dd.DTHDAT,
        onset_date=crf.ae.AESTDAT,
        report_date=date.today(),
        outcome=_AEOUT_TO_OUTCOME.get(crf.ae.AEOUT, "Unknown"),
        narrative="",  # AI fills this
        lab_data=_format_lab_data(crf),
        medical_history=_format_medical_history(crf),
    )

    # --- Section C: Suspect Product ---
    therapy_start, therapy_end = _format_therapy_dates(crf)
    baseline_meds, ae_treatment_meds = classify_cm_records(crf)
    section_c = SectionC(
        drug_name=f"{settings.DRUG_NAME} ({settings.DRUG_MANUFACTURER})",
        dose_frequency_route=_format_dose_freq_route(crf),
        therapy_start=therapy_start,
        therapy_end=therapy_end,
        indication=_derive_indication(crf, settings.INDICATION),
        product_type=settings.PRODUCT_TYPE,
        lot_number=crf.da.LOT_NUMBER or "",
        expiry_date=crf.da.EXPIRY_DATE,
        dechallenge="",  # AI fills this
        rechallenge="",  # AI fills this
        concomitant_meds=_format_concomitant_meds(baseline_meds),
    )

    # --- Section E: Reporter (map from investigator info if available) ---
    inv = crf.investigator
    section_e = SectionE(
        reporter_name=inv.name or "",
        reporter_address=inv.address or settings.SPONSOR_ADDRESS,
        reporter_phone=inv.phone or "",
        reporter_email=inv.email or "",
        reporter_qualification=settings.REPORTER_QUALIFICATION,
        reported_to_fda="No",
    )

    # --- Section G: Manufacturer ---
    section_g = SectionG(
        sponsor_contact=settings.SPONSOR_CONTACT,
        source=settings.REPORT_SOURCE,
        awareness_date=date.today(),
        ind_type=settings.IND_TYPE,
        ind_number=settings.IND_NUMBER,
        initial_followup="Initial",
        ae_term=crf.ae.AETERM,
        report_number=_generate_report_number(settings.PROTOCOL_NUMBER),
    )

    # --- ILD flag ---
    # Priority: Sentinel detection > AETERM text matching fallback
    ild_flag = False
    if sentinel_output is not None:
        ild_flag = sentinel_output.ild_detected
    if not ild_flag:
        aeterm_lower = (crf.ae.AETERM or "").strip().lower()
        ild_flag = aeterm_lower in ("pneumonitis", "interstitial lung disease", "ild")

    return MedWatch3500A(
        section_a=section_a,
        section_b=section_b,
        section_c=section_c,
        section_e=section_e,
        section_g=section_g,
        ild_flag=ild_flag,
        non_serious_flag=(crf.ae.AESER == "N"),
    )
