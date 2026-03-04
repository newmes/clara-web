"""Doc Agent — Agno Workflow for MedWatch 3500A + E2B(R3) generation."""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any, Optional

from agno.agent.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.workflow.step import Step, StepInput, StepOutput
from agno.workflow.workflow import Workflow

from .code_maps import RECHALLENGE_CL16
from .e2b_converter import convert_to_e2b_xml
from .meddra_coder import code_meddra
from .medwatch_mapper import classify_cm_records, map_crf_to_medwatch
from .prompts import (
    B5_NARRATIVE_PROMPT,
    C7_DECHALLENGE_PROMPT,
    C8_RECHALLENGE_PROMPT,
)
from .config import Settings
from .schemas.agent_output import (
    DechallengeResult,
    MedDRACode,
    RechallengeResult,
    SentinelOutput,
)
from .schemas.crf import CRFData
from .schemas.medwatch import MedWatch3500A

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt formatting helpers
# ---------------------------------------------------------------------------

def _format_date(d: Optional[date]) -> str:
    """Format date as DD-MMM-YYYY or 'N/A'."""
    if d is None:
        return "N/A"
    return d.strftime("%d-%b-%Y").upper()


def _format_dm_data(crf: CRFData) -> str:
    dm = crf.dm
    return (
        f"Subject ID: {dm.SUBJID}\n"
        f"Age: {dm.AGE} years\n"
        f"Sex: {dm.SEX}"
    )


def _format_ec_data(crf: CRFData, drug_name: str = "", indication: str = "") -> str:
    lines = []
    if drug_name:
        lines.append(f"Suspect drug: {drug_name}")
    if indication:
        lines.append(f"Indication: {indication}")
    for i, ec in enumerate(crf.ec, 1):
        period = f"Exposure period {i}: " if len(crf.ec) > 1 else ""
        lines.append(
            f"{period}{ec.ECDSTXT} {ec.ECDOSFRQ} {ec.ECROUTE}\n"
            f"  Start: {_format_date(ec.ECSTDAT)}\n"
            f"  End: {_format_date(ec.ECENDAT)}"
        )
        if ec.ECDOSADJ:
            lines.append(f"  Dose adjustment: {ec.ECDOSADJ}")
    return "\n".join(lines) if lines else "N/A"


def _format_ae_data(crf: CRFData) -> str:
    ae = crf.ae
    sae_flags = []
    if ae.AESDTH == "Y":
        sae_flags.append("Death")
    if ae.AESLIFE == "Y":
        sae_flags.append("Life-threatening")
    if ae.AESHOSP == "Y":
        sae_flags.append("Hospitalization")
    if ae.AESDISAB == "Y":
        sae_flags.append("Disability")
    if ae.AESCONG == "Y":
        sae_flags.append("Congenital anomaly")
    if ae.AESMIE == "Y":
        sae_flags.append("Other medically important")

    toxgr_line = f"\nCTCAE Grade: {ae.AETOXGR}" if ae.AETOXGR else ""

    result = (
        f"Term: {ae.AETERM}\n"
        f"Onset: {_format_date(ae.AESTDAT)}\n"
        f"End: {_format_date(ae.AEENDAT)}\n"
        f"Severity: {ae.AESEV}{toxgr_line}\n"
        f"Serious: {ae.AESER}\n"
        f"Causality: {ae.AEREL}\n"
        f"Action taken: {ae.AEACN}\n"
        f"Outcome: {ae.AEOUT}\n"
        f"SAE criteria: {', '.join(sae_flags) if sae_flags else 'None'}"
    )
    # Hospitalization dates
    if ae.AESHOSP == "Y":
        hosp_start = _format_date(ae.AEHOSPSTDAT) if ae.AEHOSPSTDAT else "same as AE onset"
        hosp_end = _format_date(ae.AEHOSPENDAT) if ae.AEHOSPENDAT else "ongoing"
        result += f"\nHospitalization: {hosp_start} to {hosp_end}"
    return result


# Mapping: AE term keywords → relevant lab test codes
_AE_RELEVANT_LABS: dict[str, set[str]] = {
    "hyperglycemia": {"GLUCOSEFASTING", "GLUCOSE", "HBA1C"},
    "neutropenia": {"ANC", "WBC"},
    "febrile_neutropenia": {"ANC", "WBC"},
    "thrombocytopenia": {"PLATELETS", "PLT"},
    "anemia": {"HEMOGLOBIN", "HGB", "HCT"},
    "hepatotoxicity": {"ALT", "AST", "TOTALBILIRUBIN", "ALP", "ALBUMIN"},
    "nephrotoxicity": {"CREATININE", "EGFR", "BUN"},
    "hypothyroidism": {"TSH", "FT4", "FT3"},
    "hyperthyroidism": {"TSH", "FT4", "FT3"},
    "pneumonitis": {"LDH", "KL6", "SPO2"},
    "dyspnoea": {"LDH", "SPO2", "HEMOGLOBIN", "HGB", "CREATININE", "ALBUMIN"},
    "dyspnea": {"LDH", "SPO2", "HEMOGLOBIN", "HGB", "CREATININE", "ALBUMIN"},
    "cough": {"LDH", "WBC", "ANC", "CRP", "SPO2"},
    "stomatitis": {"ANC", "WBC", "ALBUMIN"},
    "fatigue": {"HEMOGLOBIN", "HGB", "TSH", "CREATININE", "EGFR", "SODIUM", "ALBUMIN"},
    "diarrhea": {"SODIUM", "POTASSIUM", "CREATININE", "EGFR", "ALBUMIN"},
    "colitis": {"ANC", "WBC", "CRP", "ALBUMIN", "SODIUM", "POTASSIUM"},
    "rash": {"ANC", "WBC"},
    "nausea": {"SODIUM", "POTASSIUM", "CREATININE"},
    "infusion_related_reaction": {"ANC", "WBC", "CRP"},
}


def _get_relevant_lab_codes(ae_term: str) -> set[str] | None:
    """Return relevant lab codes for an AE term, or None to include all."""
    term_lower = ae_term.lower().replace(" ", "_").replace("-", "_")
    for key, codes in _AE_RELEVANT_LABS.items():
        if key in term_lower or term_lower in key:
            return codes
    return None


def _format_lb_data(crf: CRFData, ae_term: str = "") -> str:
    if not crf.lb.records:
        return "No lab data available"
    relevant_codes = _get_relevant_lab_codes(ae_term) if ae_term else None
    lines = []
    for r in crf.lb.records:
        # Filter to AE-relevant labs if mapping exists
        if relevant_codes and r.LBTESTCD.upper() not in relevant_codes:
            continue
        ref = ""
        if r.LBORNRLO and r.LBORNRHI:
            ref = f" (ref: {r.LBORNRLO}-{r.LBORNRHI})"
        unit = f" {r.LBORRESU}" if r.LBORRESU else ""
        name = r.LBTEST or r.LBTESTCD
        lines.append(f"{name}: {r.LBORRES}{unit}{ref} [{_format_date(r.LBDAT)}]")
    return "\n".join(lines) if lines else "No relevant lab data"


def _format_mh_data(crf: CRFData) -> str:
    if not crf.mh.records:
        return "No relevant medical history"
    lines = []
    for r in crf.mh.records:
        ongoing = " (ongoing)" if r.MHONGO == "Y" else ""
        lines.append(f"- {r.MHTERM}{ongoing}")
    return "\n".join(lines)


def _format_vs_data(crf: CRFData) -> str:
    parts = []
    if crf.vs.WEIGHT:
        parts.append(f"Weight: {crf.vs.WEIGHT} kg")
    if crf.vs.HEIGHT:
        parts.append(f"Height: {crf.vs.HEIGHT} cm")
    for r in crf.vs.records:
        parts.append(f"{r.VSTESTCD}: {r.VSORRES} {r.VSORRESU}")
    return "\n".join(parts) if parts else "No vital signs data"


def _format_cm_data(crf: CRFData) -> str:
    if not crf.cm.records:
        return "No concomitant medications"
    baseline_meds, ae_treatment_meds = classify_cm_records(crf)
    lines = []
    if baseline_meds:
        lines.append("Baseline medications:")
        for r in baseline_meds:
            entry = f"- {r.CMTRT}"
            if r.CMDSTXT:
                entry += f" {r.CMDSTXT}"
            if r.CMINDC:
                entry += f" (for {r.CMINDC})"
            lines.append(entry)
    if ae_treatment_meds:
        if lines:
            lines.append("")
        lines.append("AE treatment medications:")
        for r in ae_treatment_meds:
            entry = f"- {r.CMTRT}"
            if r.CMDSTXT:
                entry += f" {r.CMDSTXT}"
            if r.CMINDC:
                entry += f" (for {r.CMINDC})"
            lines.append(entry)
    return "\n".join(lines) if lines else "No concomitant medications"


def _format_dd_data(crf: CRFData) -> str:
    if not crf.dd.DTHDAT:
        return "N/A (patient alive)"
    lines = [f"Death date: {_format_date(crf.dd.DTHDAT)}"]
    if crf.dd.PRCDTH:
        lines.append(f"Cause: {crf.dd.PRCDTH}")
    if crf.dd.AUTOPIND:
        lines.append(f"Autopsy: {crf.dd.AUTOPIND}")
    return "\n".join(lines)


def _format_imaging_data(crf: CRFData) -> str:
    if not crf.imaging.records:
        return "No imaging studies available"
    lines = []
    for r in crf.imaging.records:
        lines.append(
            f"{r.IMG_MODALITY} {r.IMG_REGION} [{_format_date(r.IMG_DAT)}]:\n"
            f"  Findings: {r.IMG_FINDINGS}\n"
            f"  Impression: {r.IMG_IMPRESSION}"
        )
        extras = []
        if r.IMG_PLEFF:
            extras.append(f"Pleural effusion: {r.IMG_PLEFF}")
        if r.IMG_CONSOL:
            extras.append(f"Consolidation: {r.IMG_CONSOL}")
        if r.IMG_READER:
            extras.append(f"Reader: {r.IMG_READER}")
        if extras:
            lines.append("  " + ", ".join(extras))
    return "\n".join(lines)


def _format_pft_data(crf: CRFData) -> str:
    if not crf.pft.records:
        return "No pulmonary function test data available"
    baseline = [r for r in crf.pft.records if r.PFT_BLFL == "Y"]
    post_ae = [r for r in crf.pft.records if r.PFT_BLFL == "N"]
    lines = []
    if baseline:
        lines.append("Baseline:")
        for r in baseline:
            ref = ""
            if r.PFT_REFLO is not None and r.PFT_REFHI is not None:
                ref = f" (ref: {r.PFT_REFLO}-{r.PFT_REFHI})"
            lines.append(
                f"  {r.PFT_TEST} ({r.PFT_TESTCD}): {r.PFT_RESULT} {r.PFT_UNIT}{ref} "
                f"[{_format_date(r.PFT_DAT)}]"
            )
    if post_ae:
        lines.append("Post-AE:")
        for r in post_ae:
            ref = ""
            if r.PFT_REFLO is not None and r.PFT_REFHI is not None:
                ref = f" (ref: {r.PFT_REFLO}-{r.PFT_REFHI})"
            lines.append(
                f"  {r.PFT_TEST} ({r.PFT_TESTCD}): {r.PFT_RESULT} {r.PFT_UNIT}{ref} "
                f"[{_format_date(r.PFT_DAT)}]"
            )
    return "\n".join(lines)


def _format_microbiology_data(crf: CRFData) -> str:
    if not crf.microbiology.records:
        return "No microbiology data available"
    lines = []
    for r in crf.microbiology.records:
        line = f"{r.MB_SPECIMEN} — {r.MB_TEST} [{_format_date(r.MB_DAT)}]: {r.MB_RESULT}"
        if r.MB_ORGANISM:
            line += f"\n  Organism: {r.MB_ORGANISM}"
        if r.MB_SENSITIVITY:
            line += f"\n  Susceptibility: {r.MB_SENSITIVITY}"
        lines.append(line)
    return "\n".join(lines)


def _format_consultation_data(crf: CRFData) -> str:
    if not crf.consultation.records:
        return "No specialist consultation data available"
    lines = []
    for r in crf.consultation.records:
        line = f"{r.CONSULT_SPECIALTY} [{_format_date(r.CONSULT_DAT)}]:\n  {r.CONSULT_IMPRESSION}"
        if r.CONSULT_PHYSICIAN:
            line += f"\n  Consultant: {r.CONSULT_PHYSICIAN}"
        lines.append(line)
    return "\n".join(lines)


def _format_sentinel_output(sentinel: Optional[SentinelOutput], crf: Optional[CRFData] = None) -> str:
    if sentinel is None or not sentinel.ild_detected:
        return "No ILD-related clinical findings"
    lines = [
        f"ILD suspected (Grade {sentinel.ild_grade})",
    ]
    if crf is not None:
        lines.append(f"Reported AE term: {crf.ae.AETERM}")
    if sentinel.cxr_findings:
        lines.append(f"Chest imaging: {sentinel.cxr_findings}")
    if sentinel.differential_diagnosis:
        lines.append(f"Differential diagnosis considerations: {sentinel.differential_diagnosis}")
    if sentinel.kl6_value:
        lines.append(f"KL-6: {sentinel.kl6_value} U/mL")
    if sentinel.spo2_value:
        lines.append(f"SpO2: {sentinel.spo2_value}%")
    return "\n".join(lines)


def format_b5_prompt(
    crf: CRFData,
    settings: Settings,
    sentinel_output: Optional[SentinelOutput] = None,
) -> str:
    """Format the B5 Narrative prompt with all CRF data.

    Omits empty data sections to save tokens for the 4K-context model.
    """
    # Build data sections — only include non-empty ones
    _NO_DATA_PREFIXES = ("No ", "N/A", "No relevant", "No lab", "No vital",
                         "No concomitant", "No imaging", "No pulmonary",
                         "No microbiology", "No specialist", "No ILD")

    def _has_data(text: str) -> bool:
        return bool(text) and not any(text.startswith(p) for p in _NO_DATA_PREFIXES)

    sections: list[str] = []

    # Always-included sections
    sections.append(f"## Patient Demographics (DM)\n{_format_dm_data(crf)}")
    sections.append(f"## Study Drug Exposure (EC)\n{_format_ec_data(crf, drug_name=settings.DRUG_NAME, indication=settings.INDICATION)}")
    sections.append(f"## Adverse Event (AE)\n{_format_ae_data(crf)}")

    # Conditionally-included sections
    lb = _format_lb_data(crf, ae_term=crf.ae.AETERM)
    if _has_data(lb):
        sections.append(f"## Laboratory Results (LB)\n{lb}")
    mh = _format_mh_data(crf)
    if _has_data(mh):
        sections.append(f"## Medical History (MH)\n{mh}")
    vs = _format_vs_data(crf)
    if _has_data(vs):
        sections.append(f"## Vital Signs (VS)\n{vs}")
    cm = _format_cm_data(crf)
    if _has_data(cm):
        sections.append(f"## Concomitant Medications (CM)\n{cm}")
    dd = _format_dd_data(crf)
    if _has_data(dd):
        sections.append(f"## Death Details (DD)\n{dd}")
    sentinel = _format_sentinel_output(sentinel_output, crf)
    if _has_data(sentinel):
        sections.append(f"## ILD Clinical Findings\n{sentinel}")
    imaging = _format_imaging_data(crf)
    if _has_data(imaging):
        sections.append(f"## Imaging Studies\n{imaging}")
    pft = _format_pft_data(crf)
    if _has_data(pft):
        sections.append(f"## Pulmonary Function Tests (PFT)\n{pft}")
    micro = _format_microbiology_data(crf)
    if _has_data(micro):
        sections.append(f"## Microbiology Results\n{micro}")
    consult = _format_consultation_data(crf)
    if _has_data(consult):
        sections.append(f"## Specialist Consultations\n{consult}")

    data_block = "\n\n".join(sections)

    return B5_NARRATIVE_PROMPT.format(
        report_type="Initial",
        data_block=data_block,
    )


def format_c7_prompt(crf: CRFData) -> str:
    """Format the C7 Dechallenge prompt."""
    ec_mod_date = "N/A"
    if crf.ec and crf.ae.AEACN in ("DRUG WITHDRAWN", "DRUG INTERRUPTED"):
        # Use the first EC's end date (when drug was stopped/interrupted)
        for ec in crf.ec:
            if ec.ECENDAT:
                ec_mod_date = _format_date(ec.ECENDAT)
                break
    elif crf.ec and crf.ae.AEACN == "DOSE REDUCED":
        # Use second EC's start date (when reduced dose began)
        if len(crf.ec) > 1:
            ec_mod_date = _format_date(crf.ec[1].ECSTDAT)
        elif crf.ec[0].ECENDAT:
            ec_mod_date = _format_date(crf.ec[0].ECENDAT)

    return C7_DECHALLENGE_PROMPT.format(
        aeacn=crf.ae.AEACN,
        aeout=crf.ae.AEOUT,
        aestdat=_format_date(crf.ae.AESTDAT),
        aeendat=_format_date(crf.ae.AEENDAT),
        ecstdat=_format_date(crf.ec[0].ECSTDAT) if crf.ec else "N/A",
        ec_modification_date=ec_mod_date,
    )


def format_c8_prompt(crf: CRFData) -> str:
    """Format the C8 Rechallenge prompt."""
    ec_lines = []
    for i, ec in enumerate(crf.ec, 1):
        ec_lines.append(
            f"Period {i}: {ec.ECDSTXT} {ec.ECDOSFRQ}, "
            f"Start: {_format_date(ec.ECSTDAT)}, "
            f"End: {_format_date(ec.ECENDAT)}"
        )
    ec_history = "\n".join(ec_lines) if ec_lines else "Single exposure period only"

    ae_history = (
        f"- {crf.ae.AETERM}: onset {_format_date(crf.ae.AESTDAT)}, "
        f"end {_format_date(crf.ae.AEENDAT)}, outcome {crf.ae.AEOUT}"
    )

    return C8_RECHALLENGE_PROMPT.format(
        ec_history=ec_history,
        ae_history=ae_history,
        current_aeterm=crf.ae.AETERM,
    )


# ---------------------------------------------------------------------------
# Workflow Step Executors
# ---------------------------------------------------------------------------

def step_crf_mapping(step_input: StepInput) -> StepOutput:
    """Step 1: Map CRF data to MedWatch 3500A and prepare AI prompts."""
    data = step_input.input
    if isinstance(data, str):
        data = json.loads(data)

    crf = CRFData.model_validate(data["crf"])
    settings = Settings()

    sentinel_output = None
    if data.get("sentinel_output"):
        sentinel_output = SentinelOutput.model_validate(data["sentinel_output"])

    # Map CRF → MedWatch (B5/C7/C8 left blank for AI)
    medwatch = map_crf_to_medwatch(crf, settings, sentinel_output)

    # Format AI prompts
    b5_prompt = format_b5_prompt(crf, settings, sentinel_output)
    c7_prompt = format_c7_prompt(crf)
    c8_prompt = format_c8_prompt(crf)

    return StepOutput(
        step_name="crf_mapping",
        content={
            "medwatch": json.loads(medwatch.model_dump_json()),
            "crf": json.loads(crf.model_dump_json()),
            "sentinel_output": json.loads(sentinel_output.model_dump_json()) if sentinel_output else None,
            "b5_prompt": b5_prompt,
            "c7_prompt": c7_prompt,
            "c8_prompt": c8_prompt,
            "settings": {
                "vllm_base_url": settings.VLLM_BASE_URL,
                "vllm_model_id": settings.VLLM_MODEL_ID,
                "vllm_api_key": settings.VLLM_API_KEY,
                "max_tokens_narrative": settings.MAX_TOKENS_NARRATIVE,
                "temperature_narrative": settings.TEMPERATURE_NARRATIVE,
                "max_tokens_structured": settings.MAX_TOKENS_STRUCTURED,
                "temperature_structured": settings.TEMPERATURE_STRUCTURED,
            },
        },
    )


def step_ai_processing(step_input: StepInput) -> StepOutput:
    """Step 2: Run B5/C7/C8 agents in parallel via ThreadPoolExecutor."""
    data = step_input.previous_step_content
    s = data["settings"]

    model = OpenAILike(
        id=s["vllm_model_id"],
        base_url=s["vllm_base_url"],
        api_key=s["vllm_api_key"],
    )

    narrative_agent = Agent(
        name="b5_narrative",
        model=model,
        markdown=False,
    )
    dechallenge_agent = Agent(
        name="c7_dechallenge",
        model=model,
        output_schema=DechallengeResult,
        markdown=False,
    )
    rechallenge_agent = Agent(
        name="c8_rechallenge",
        model=model,
        output_schema=RechallengeResult,
        markdown=False,
    )

    def run_narrative():
        result = narrative_agent.run(data["b5_prompt"])
        text = str(result.content)
        # Strip markdown formatting that may leak from the LLM
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
        text = re.sub(r"^[-*]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"(?i)^(?:section b5:?\s*)?(?:clinical narrative):?\s*\n*", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def run_dechallenge():
        result = dechallenge_agent.run(data["c7_prompt"])
        if isinstance(result.content, DechallengeResult):
            return result.content.model_dump()
        # Fallback: try to parse JSON from string
        try:
            return json.loads(result.content)
        except (json.JSONDecodeError, TypeError):
            return {"c7_answer": "Unknown", "c7_rationale": str(result.content)}

    def run_rechallenge():
        # Deterministic guard: single exposure period → no rechallenge possible
        crf_data = data.get("crf", {})
        ec_list = crf_data.get("ec", [])
        if len(ec_list) < 2:
            return {
                "c8_answer": "Does not apply",
                "c8_rationale": "Single exposure period — drug was not re-administered",
                "e2b_code": 4,
            }
        result = rechallenge_agent.run(data["c8_prompt"])
        if isinstance(result.content, RechallengeResult):
            return result.content.model_dump()
        try:
            return json.loads(result.content)
        except (json.JSONDecodeError, TypeError):
            return {
                "c8_answer": "Does not apply",
                "c8_rationale": str(result.content),
                "e2b_code": 4,
            }

    with ThreadPoolExecutor(max_workers=3) as pool:
        b5_future = pool.submit(run_narrative)
        c7_future = pool.submit(run_dechallenge)
        c8_future = pool.submit(run_rechallenge)

        narrative = b5_future.result()
        dechallenge = c7_future.result()
        rechallenge = c8_future.result()

    return StepOutput(
        step_name="ai_processing",
        content={
            **data,  # pass through medwatch, crf, settings, etc.
            "narrative": narrative,
            "dechallenge": dechallenge,
            "rechallenge": rechallenge,
        },
    )


def step_assemble(step_input: StepInput) -> StepOutput:
    """Step 3: Merge AI outputs into the MedWatch 3500A form."""
    data = step_input.previous_step_content

    medwatch = MedWatch3500A.model_validate(data["medwatch"])

    # Insert AI-generated content
    narrative = data.get("narrative", "")
    if isinstance(narrative, dict):
        narrative = narrative.get("content", str(narrative))
    medwatch.section_b.narrative = str(narrative)

    dechallenge = data.get("dechallenge", {})
    if isinstance(dechallenge, dict):
        c7_answer = dechallenge.get("c7_answer", "Unknown")
        c7_rationale = dechallenge.get("c7_rationale", "")
        medwatch.section_c.dechallenge = f"{c7_answer} — {c7_rationale}"
    else:
        medwatch.section_c.dechallenge = str(dechallenge)

    rechallenge = data.get("rechallenge", {})
    if isinstance(rechallenge, dict):
        c8_answer = rechallenge.get("c8_answer", "Does not apply")
        c8_rationale = rechallenge.get("c8_rationale", "")
        medwatch.section_c.rechallenge = f"{c8_answer} — {c8_rationale}"
    else:
        medwatch.section_c.rechallenge = str(rechallenge)

    return StepOutput(
        step_name="assemble_3500a",
        content={
            "medwatch": json.loads(medwatch.model_dump_json()),
            "crf": data["crf"],
            "sentinel_output": data.get("sentinel_output"),
            "settings": data["settings"],
            "dechallenge": dechallenge,
            "rechallenge": rechallenge,
        },
    )


def step_e2b(step_input: StepInput) -> StepOutput:
    """Step 4: MedDRA coding + E2B(R3) XML generation."""
    data = step_input.previous_step_content

    medwatch = MedWatch3500A.model_validate(data["medwatch"])
    crf = CRFData.model_validate(data["crf"])
    settings = Settings()

    # MedDRA coding
    meddra_code = code_meddra(
        crf.ae.AETERM,
        base_url=settings.VLLM_BASE_URL,
        model_id=settings.VLLM_MODEL_ID,
        api_key=settings.VLLM_API_KEY,
    )

    # Rechallenge E2B code
    rechallenge = data.get("rechallenge", {})
    rechallenge_code = None
    if isinstance(rechallenge, dict):
        rechallenge_code = rechallenge.get("e2b_code")

    # Generate E2B XML
    e2b_xml = convert_to_e2b_xml(
        medwatch=medwatch,
        crf=crf,
        meddra_code=meddra_code,
        settings=settings,
        rechallenge_code=rechallenge_code,
    )

    return StepOutput(
        step_name="e2b_conversion",
        content={
            "medwatch": json.loads(medwatch.model_dump_json()),
            "e2b_xml": e2b_xml,
            "meddra_code": json.loads(meddra_code.model_dump_json()),
        },
    )


# ---------------------------------------------------------------------------
# Workflow Factory
# ---------------------------------------------------------------------------

def create_doc_workflow(
    vllm_base_url: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Workflow:
    """Create the Doc Agent Agno Workflow.

    Steps:
    1. CRF → MedWatch mapping (deterministic)
    2. AI processing: B5 Narrative + C7 Dechallenge + C8 Rechallenge (parallel)
    3. 3500A assembly (deterministic)
    4. MedDRA coding + E2B(R3) XML generation (deterministic)
    """
    return Workflow(
        name="doc_agent",
        description="Generate MedWatch FDA 3500A + E2B(R3) XML from CRF data",
        steps=[
            Step(name="crf_mapping", executor=step_crf_mapping),
            Step(name="ai_processing", executor=step_ai_processing),
            Step(name="assemble_3500a", executor=step_assemble),
            Step(name="e2b_conversion", executor=step_e2b),
        ],
    )


def run_doc_workflow(
    crf_data: CRFData,
    settings: Optional[Settings] = None,
    sentinel_output: Optional[SentinelOutput] = None,
) -> dict[str, Any]:
    """Run the Doc Agent workflow end-to-end.

    Returns:
        dict with keys: "medwatch" (MedWatch3500A), "e2b_xml" (str), "meddra_code" (MedDRACode)
    """
    if crf_data.ae.AESER == "N":
        logger.info("Non-serious AE (AESER=N) for %s — skipping MedWatch/E2B", crf_data.dm.SUBJID)
        return {
            "non_serious": True,
            "subjid": crf_data.dm.SUBJID,
            "aeterm": crf_data.ae.AETERM,
            "reason": "Non-serious AE — route to AE CRF form, not MedWatch 3500A",
        }

    if settings is None:
        settings = Settings()

    workflow = create_doc_workflow(
        vllm_base_url=settings.VLLM_BASE_URL,
        model_id=settings.VLLM_MODEL_ID,
    )

    # Prepare input
    workflow_input: dict[str, Any] = {
        "crf": json.loads(crf_data.model_dump_json()),
    }
    if sentinel_output:
        workflow_input["sentinel_output"] = json.loads(sentinel_output.model_dump_json())

    # Run workflow
    result = workflow.run(input=workflow_input)

    # Extract final output
    content = result.content
    if isinstance(content, dict):
        return {
            "medwatch": MedWatch3500A.model_validate(content["medwatch"]),
            "e2b_xml": content["e2b_xml"],
            "meddra_code": MedDRACode.model_validate(content["meddra_code"]),
        }

    raise ValueError(f"Unexpected workflow output: {type(content)}")
