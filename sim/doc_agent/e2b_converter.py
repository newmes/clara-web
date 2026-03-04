"""Convert MedWatch 3500A + CRF data to E2B(R3) XML format."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from lxml import etree

from .code_maps import (
    ACTION_CL15,
    OUTCOME_CL11,
    RECHALLENGE_CL16,
    SEX_E2B,
)
from .config import Settings
from .schemas.agent_output import MedDRACode
from .medwatch_mapper import classify_cm_records
from .schemas.crf import CRFData
from .schemas.medwatch import MedWatch3500A

# XSD schema path (loaded once on first validation call)
_XSD_PATH = Path(__file__).parent / "templates" / "ich_icsr_v2.1.xsd"
_schema_cache: Optional[etree.XMLSchema] = None


def _get_schema() -> etree.XMLSchema:
    """Load and cache the ICH ICSR XSD schema."""
    global _schema_cache
    if _schema_cache is None:
        xsd_doc = etree.parse(str(_XSD_PATH))
        _schema_cache = etree.XMLSchema(xsd_doc)
    return _schema_cache


def validate_e2b_xml(xml_str: str) -> tuple[bool, list[str]]:
    """Validate E2B XML against ICH ICSR XSD schema.

    Returns:
        (is_valid, errors) — True with empty list if valid,
        False with list of error messages otherwise.
    """
    try:
        doc = etree.fromstring(xml_str.encode("utf-8"))
    except etree.XMLSyntaxError as e:
        return False, [f"XML syntax error: {e}"]

    schema = _get_schema()
    is_valid = schema.validate(doc)

    if is_valid:
        return True, []

    errors = [str(err) for err in schema.error_log]
    return False, errors


def _iso_date(d: Optional[date]) -> str:
    """Format date as YYYYMMDD for E2B."""
    if d is None:
        return ""
    return d.strftime("%Y%m%d")


def _sub(parent: etree._Element, tag: str, text: str = "") -> etree._Element:
    """Add a subelement with optional text."""
    el = etree.SubElement(parent, tag)
    if text:
        el.text = str(text)
    return el


def convert_to_e2b_xml(
    medwatch: MedWatch3500A,
    crf: CRFData,
    meddra_code: MedDRACode,
    settings: Settings,
    rechallenge_code: Optional[int] = None,
) -> str:
    """Build E2B(R3) ICSR XML from MedWatch 3500A + CRF supplemental data.

    Returns pretty-printed XML string with declaration.
    """
    # Root: ichicsr
    root = etree.Element("ichicsr", lang="en")

    # --- N: Transmission Headers ---
    header = _sub(root, "ichicsrmessageheader")
    _sub(header, "messagetype", "ichicsr")
    _sub(header, "messageformatversion", "2.1")
    _sub(header, "messageformatrelease", "2.0")
    _sub(header, "messagenumb", uuid4().hex[:12].upper())
    _sub(header, "messagesenderidentifier", settings.SPONSOR_NAME)
    _sub(header, "messagereceiveridentifier", "FDA")
    _sub(header, "messagedateformat", "204")
    _sub(header, "messagedate", datetime.now().strftime("%Y%m%d%H%M%S"))

    # --- safetyreport (one per case) ---
    report = _sub(root, "safetyreport")

    # C.1: Case Identification
    _sub(report, "safetyreportversion", "1")
    _sub(report, "safetyreportid", medwatch.section_g.report_number)
    _sub(report, "primarysourcecountry", "US")
    _sub(report, "occurcountry", "US")
    _sub(report, "transmissiondateformat", "102")
    _sub(report, "transmissiondate", _iso_date(date.today()))
    _sub(report, "reporttype", "1")  # 1 = Spontaneous/IND
    _sub(report, "serious", "1" if crf.ae.AESER == "Y" else "2")

    # Seriousness criteria
    if crf.ae.AESDTH == "Y":
        _sub(report, "seriousnessdeath", "1")
    if crf.ae.AESLIFE == "Y":
        _sub(report, "seriousnesslifethreatening", "1")
    if crf.ae.AESHOSP == "Y":
        _sub(report, "seriousnesshospitalization", "1")
    if crf.ae.AESDISAB == "Y":
        _sub(report, "seriousnessdisabling", "1")
    if crf.ae.AESCONG == "Y":
        _sub(report, "seriousnesscongenitalanomali", "1")
    if crf.ae.AESMIE == "Y":
        _sub(report, "seriousnessother", "1")

    _sub(report, "receivedateformat", "102")
    _sub(report, "receivedate", _iso_date(medwatch.section_g.awareness_date))
    _sub(report, "receiptdateformat", "102")
    _sub(report, "receiptdate", _iso_date(date.today()))
    _sub(report, "fulfilexpeditecriteria", "1")

    # C.2: Primary Source (Reporter)
    primary_source = _sub(report, "primarysource")
    _sub(primary_source, "reportergivename", medwatch.section_e.reporter_name or "Investigator")
    _sub(primary_source, "qualification", "1")  # 1 = Physician

    # C.3: Sender
    sender = _sub(report, "sender")
    _sub(sender, "sendertype", "2")  # 2 = Pharmaceutical company
    _sub(sender, "senderorganization", settings.SPONSOR_NAME)

    # C.4: Receiver
    receiver = _sub(report, "receiver")
    _sub(receiver, "receivertype", "2")  # 2 = Regulatory authority
    _sub(receiver, "receiverorganization", "FDA")

    # C.5: Study Identification
    report_duplicate = _sub(report, "reportduplicate")
    _sub(report_duplicate, "duplicatesource", settings.SPONSOR_NAME)
    _sub(report_duplicate, "duplicatenumb", medwatch.section_g.report_number)

    # Study registration
    study_reg = _sub(report, "studyregistration")
    _sub(study_reg, "studyregistrationnumb", settings.PROTOCOL_NUMBER)
    _sub(study_reg, "studyregistrationcountry", "US")

    # IND number
    if settings.IND_NUMBER:
        _sub(report, "authoritynumb", settings.IND_NUMBER)
        _sub(report, "companynumb", medwatch.section_g.report_number)

    # --- D: Patient ---
    patient = _sub(report, "patient")
    _sub(patient, "patientinitial", crf.dm.SUBJID[:3] if crf.dm.SUBJID else "UNK")

    # D.2: Age
    if crf.dm.AGE:
        _sub(patient, "patientonsetage", str(crf.dm.AGE))
        _sub(patient, "patientonsetageunit", "801")  # 801 = Year

    # D.4: Weight and Height
    if crf.vs.WEIGHT:
        _sub(patient, "patientweight", str(crf.vs.WEIGHT))
    if crf.vs.HEIGHT:
        _sub(patient, "patientheight", str(crf.vs.HEIGHT))

    # D.5: Sex
    sex_code = SEX_E2B.get(crf.dm.SEX, 0)
    if sex_code:
        _sub(patient, "patientsex", str(sex_code))

    # D.7: Medical History
    for mh in crf.mh.records:
        med_hist = _sub(patient, "medicalhistoryepisode")
        _sub(med_hist, "patientepisodename", mh.MHTERM)
        if mh.MHSTDAT:
            _sub(med_hist, "patientmedicalstartdateformat", "102")
            _sub(med_hist, "patientmedicalstartdate", _iso_date(mh.MHSTDAT))
        if mh.MHONGO == "Y":
            _sub(med_hist, "patientmedicalcontinue", "1")

    # D.9: Death (if applicable)
    if crf.dd.DTHDAT:
        _sub(patient, "patientdeathdateformat", "102")
        _sub(patient, "patientdeathdate", _iso_date(crf.dd.DTHDAT))
        if crf.dd.PRCDTH:
            death_cause = _sub(patient, "patientdeathcause")
            _sub(death_cause, "patientdeathreport", crf.dd.PRCDTH)
        if crf.dd.AUTOPIND:
            _sub(patient, "patientautopsyyesno", "1" if crf.dd.AUTOPIND == "Y" else "2")

    # --- E.i: Reaction/Event ---
    reaction = _sub(patient, "reaction")
    _sub(reaction, "primarysourcereaction", crf.ae.AETERM)

    # E.i.2.1b: MedDRA coding (Required)
    if meddra_code.llt:
        _sub(reaction, "reactionmeddraversionllt", "26.1")
        _sub(reaction, "reactionmeddrallt", meddra_code.llt)
    if meddra_code.pt:
        _sub(reaction, "reactionmeddraversionpt", "26.1")
        _sub(reaction, "reactionmeddrapt", meddra_code.pt)

    # E.i.4: Start date
    _sub(reaction, "reactionstartdateformat", "102")
    _sub(reaction, "reactionstartdate", _iso_date(crf.ae.AESTDAT))

    # E.i.5: End date (CRF supplementation)
    if crf.ae.AEENDAT:
        _sub(reaction, "reactionenddateformat", "102")
        _sub(reaction, "reactionenddate", _iso_date(crf.ae.AEENDAT))

    # E.i.7: Outcome (CL11)
    outcome_code = OUTCOME_CL11.get(crf.ae.AEOUT)
    if outcome_code:
        _sub(reaction, "reactionoutcome", str(outcome_code))

    # --- F.r: Tests/Procedures (Lab results) ---
    for lab in crf.lb.records:
        test = _sub(patient, "test")
        _sub(test, "testdateformat", "102")
        _sub(test, "testdate", _iso_date(lab.LBDAT))
        _sub(test, "testname", lab.LBTEST or lab.LBTESTCD)
        _sub(test, "testresult", lab.LBORRES)
        if lab.LBORRESU:
            _sub(test, "testunit", lab.LBORRESU)
        if lab.LBORNRLO:
            _sub(test, "lowtestrange", lab.LBORNRLO)
        if lab.LBORNRHI:
            _sub(test, "hightestrange", lab.LBORNRHI)

    # --- G.k: Drug Information ---
    # Suspect drug
    drug = _sub(patient, "drug")
    _sub(drug, "drugcharacterization", "1")  # 1 = Suspect
    _sub(drug, "medicinalproduct", settings.DRUG_NAME)

    # Dosage — use first EC record (dose at AE onset)
    if crf.ec:
        first_ec = crf.ec[0]
        dosage = _sub(drug, "drugstructuredosagenumb", first_ec.ECDSTXT)
        if first_ec.ECDOSFRQ:
            _sub(drug, "drugdosagetext", f"{first_ec.ECDSTXT} {first_ec.ECDOSFRQ}")
        if first_ec.ECROUTE:
            _sub(drug, "drugadministrationroute", first_ec.ECROUTE)

        # Therapy dates — first dose start, latest end
        _sub(drug, "drugstartdateformat", "102")
        _sub(drug, "drugstartdate", _iso_date(first_ec.ECSTDAT))
        ends = [ec.ECENDAT for ec in crf.ec if ec.ECENDAT is not None]
        if ends:
            _sub(drug, "drugenddateformat", "102")
            _sub(drug, "drugenddate", _iso_date(max(ends)))

    # Indication — derive from medical history
    _sub(drug, "drugindication", medwatch.section_c.indication)

    # Action taken (CL15)
    action_code = ACTION_CL15.get(crf.ae.AEACN)
    if action_code:
        _sub(drug, "actiondrug", str(action_code))

    # Rechallenge (CL16)
    rc_code = rechallenge_code or RECHALLENGE_CL16.get("Does not apply", 4)
    _sub(drug, "drugrecurreadministration", str(rc_code))

    # Lot number
    if crf.da.LOT_NUMBER:
        _sub(drug, "drugbatchnumb", crf.da.LOT_NUMBER)

    # Concomitant medications (drugcharacterization=2)
    # Element order must match ICH ICSR DTD: dosagetext before indication
    # Classify CM records: baseline vs AE treatment
    baseline_meds, ae_treatment_meds = classify_cm_records(crf)

    # Baseline medications: drugindication = original CMINDC
    for cm in baseline_meds:
        con_drug = _sub(patient, "drug")
        _sub(con_drug, "drugcharacterization", "2")  # 2 = Concomitant
        _sub(con_drug, "medicinalproduct", cm.CMTRT)
        if cm.CMDSTXT:
            _sub(con_drug, "drugdosagetext", cm.CMDSTXT)
        if cm.CMSTDAT:
            _sub(con_drug, "drugstartdateformat", "102")
            _sub(con_drug, "drugstartdate", _iso_date(cm.CMSTDAT))
        if cm.CMENDAT:
            _sub(con_drug, "drugenddateformat", "102")
            _sub(con_drug, "drugenddate", _iso_date(cm.CMENDAT))
        if cm.CMINDC:
            _sub(con_drug, "drugindication", cm.CMINDC)

    # AE treatment medications: drugindication = AETERM (treatment purpose)
    for cm in ae_treatment_meds:
        con_drug = _sub(patient, "drug")
        _sub(con_drug, "drugcharacterization", "2")  # 2 = Concomitant
        _sub(con_drug, "medicinalproduct", cm.CMTRT)
        if cm.CMDSTXT:
            _sub(con_drug, "drugdosagetext", cm.CMDSTXT)
        if cm.CMSTDAT:
            _sub(con_drug, "drugstartdateformat", "102")
            _sub(con_drug, "drugstartdate", _iso_date(cm.CMSTDAT))
        if cm.CMENDAT:
            _sub(con_drug, "drugenddateformat", "102")
            _sub(con_drug, "drugenddate", _iso_date(cm.CMENDAT))
        _sub(con_drug, "drugindication", crf.ae.AETERM)

    # --- H: Narrative ---
    summary = _sub(patient, "summary")
    _sub(summary, "narrativeincludeclinical", medwatch.section_b.narrative)

    # Reporter's comment (causality from investigator)
    if crf.ae.AEREL:
        _sub(summary, "reportercomment", f"Investigator assessment: {crf.ae.AEREL}")

    # Sender's diagnosis
    if meddra_code.pt:
        sender_diag = _sub(summary, "senderdiagnosis")
        _sub(sender_diag, "senderdiagnosismeddraversion", "26.1")
        _sub(sender_diag, "senderdiagnosis", meddra_code.pt)

    # Pretty print
    tree = etree.ElementTree(root)
    return etree.tostring(
        tree,
        pretty_print=True,
        xml_declaration=True,
        encoding="UTF-8",
    ).decode("utf-8")
