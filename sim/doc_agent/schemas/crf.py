"""CRF (Case Report Form) domain models following CDASH standards."""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class DMDomain(BaseModel):
    """Demographics domain."""
    SUBJID: str = Field(..., description="Subject identifier")
    AGE: int = Field(..., description="Age in years")
    SEX: Literal["Male", "Female"] = Field(..., description="Sex")
    RACE: str = Field(default="", description="Race")
    ETHNIC: str = Field(default="", description="Ethnicity")


class AEDomain(BaseModel):
    """Adverse Event domain."""
    AETERM: str = Field(..., description="Reported term for the adverse event")
    AESTDAT: date = Field(..., description="AE start date")
    AEENDAT: Optional[date] = Field(default=None, description="AE end date")
    AESEV: Literal["MILD", "MODERATE", "SEVERE"] = Field(..., description="Severity")
    AESER: Literal["Y", "N"] = Field(..., description="Serious?")
    AETOXGR: Optional[Literal["1", "2", "3", "4", "5"]] = Field(
        default=None, description="CTCAE Grade (1-5)"
    )
    AEREL: str = Field(default="", description="Causality assessment")
    AEACN: Literal[
        "DRUG WITHDRAWN", "DRUG INTERRUPTED", "DOSE REDUCED",
        "DOSE INCREASED", "DOSE NOT CHANGED", "UNKNOWN", "NOT APPLICABLE"
    ] = Field(..., description="Action taken with study drug")
    AEOUT: Literal[
        "RECOVERED/RESOLVED", "RECOVERING/RESOLVING",
        "NOT RECOVERED/NOT RESOLVED", "RECOVERED/RESOLVED WITH SEQUELAE",
        "FATAL", "UNKNOWN"
    ] = Field(..., description="Outcome of AE")
    # SAE seriousness criteria flags
    AESDTH: Literal["Y", "N"] = Field(default="N", description="Results in death")
    AESLIFE: Literal["Y", "N"] = Field(default="N", description="Life threatening")
    AESHOSP: Literal["Y", "N"] = Field(default="N", description="Requires hospitalization")
    AESDISAB: Literal["Y", "N"] = Field(default="N", description="Persistent disability")
    AESCONG: Literal["Y", "N"] = Field(default="N", description="Congenital anomaly")
    AESMIE: Literal["Y", "N"] = Field(default="N", description="Other medically important")
    # Hospitalization dates (if AESHOSP=Y)
    AEHOSPSTDAT: Optional[date] = Field(default=None, description="Hospitalization start date")
    AEHOSPENDAT: Optional[date] = Field(default=None, description="Hospitalization end date")


class ECDomain(BaseModel):
    """Exposure (study drug) domain."""
    ECDSTXT: str = Field(..., description="Dose text (e.g., '5.4 mg/kg')")
    ECDOSFRQ: str = Field(default="", description="Dosing frequency (e.g., 'Q3W')")
    ECROUTE: str = Field(default="", description="Route (e.g., 'Intravenous')")
    ECSTDAT: date = Field(..., description="Exposure start date")
    ECENDAT: Optional[date] = Field(default=None, description="Exposure end date")
    ECDOSADJ: Optional[str] = Field(default=None, description="Dose adjustment reason")


class LBRecord(BaseModel):
    """Single lab result record."""
    LBTESTCD: str = Field(..., description="Lab test code (e.g., 'KL6')")
    LBTEST: str = Field(default="", description="Lab test name")
    LBORRES: str = Field(..., description="Result value")
    LBORRESU: str = Field(default="", description="Unit")
    LBORNRLO: Optional[str] = Field(default=None, description="Reference range low")
    LBORNRHI: Optional[str] = Field(default=None, description="Reference range high")
    LBDAT: date = Field(..., description="Lab collection date")


class LBDomain(BaseModel):
    """Laboratory results domain."""
    records: list[LBRecord] = Field(default_factory=list)


class CMRecord(BaseModel):
    """Single concomitant medication record."""
    CMTRT: str = Field(..., description="Medication name")
    CMINDC: str = Field(default="", description="Indication")
    CMDSTXT: str = Field(default="", description="Dose text")
    CMSTDAT: Optional[date] = Field(default=None, description="Start date")
    CMENDAT: Optional[date] = Field(default=None, description="End date")
    CMCAT: Optional[Literal["BASELINE", "AE_TREATMENT"]] = Field(
        default=None, description="Medication category"
    )


class CMDomain(BaseModel):
    """Concomitant medications domain."""
    records: list[CMRecord] = Field(default_factory=list)


class MHRecord(BaseModel):
    """Single medical history record."""
    MHTERM: str = Field(..., description="Medical history term")
    MHSTDAT: Optional[date] = Field(default=None, description="Start date")
    MHENDAT: Optional[date] = Field(default=None, description="End date")
    MHONGO: Literal["Y", "N"] = Field(default="N", description="Ongoing?")


class MHDomain(BaseModel):
    """Medical history domain."""
    records: list[MHRecord] = Field(default_factory=list)


class VSRecord(BaseModel):
    """Single vital sign record."""
    VSTESTCD: str = Field(..., description="Vital sign test code")
    VSORRES: str = Field(..., description="Result value")
    VSORRESU: str = Field(default="", description="Unit")
    VSDAT: Optional[date] = Field(default=None, description="Date")


class VSDomain(BaseModel):
    """Vital signs domain."""
    records: list[VSRecord] = Field(default_factory=list)
    WEIGHT: Optional[float] = Field(default=None, description="Body weight in kg")
    HEIGHT: Optional[float] = Field(default=None, description="Height in cm")


class ImagingRecord(BaseModel):
    """Single imaging study record."""
    IMG_MODALITY: str = Field(..., description="Imaging modality (CXR, HRCT, MRI, etc.)")
    IMG_REGION: str = Field(..., description="Anatomical region (Chest, Abdomen, etc.)")
    IMG_DAT: date = Field(..., description="Imaging date")
    IMG_FINDINGS: str = Field(..., description="Radiological findings summary")
    IMG_IMPRESSION: str = Field(..., description="Radiological impression/conclusion")
    IMG_PLEFF: Optional[Literal["Y", "N"]] = Field(default=None, description="Pleural effusion")
    IMG_CONSOL: Optional[Literal["Y", "N"]] = Field(default=None, description="Consolidation")
    IMG_READER: Optional[str] = Field(default=None, description="Radiologist name")


class ImagingDomain(BaseModel):
    """Imaging studies domain."""
    records: list[ImagingRecord] = Field(default_factory=list)


class PFTRecord(BaseModel):
    """Single pulmonary function test record."""
    PFT_TESTCD: str = Field(..., description="PFT test code (DLCO, FVC, FEV1)")
    PFT_TEST: str = Field(..., description="Full test name")
    PFT_RESULT: float = Field(..., description="Result value")
    PFT_UNIT: str = Field(..., description="Unit (%predicted, L, etc.)")
    PFT_REFLO: Optional[float] = Field(default=None, description="Reference range low")
    PFT_REFHI: Optional[float] = Field(default=None, description="Reference range high")
    PFT_DAT: date = Field(..., description="Test date")
    PFT_BLFL: Literal["Y", "N"] = Field(default="N", description="Baseline flag")


class PFTDomain(BaseModel):
    """Pulmonary function test domain."""
    records: list[PFTRecord] = Field(default_factory=list)


class MBRecord(BaseModel):
    """Single microbiology result record."""
    MB_SPECIMEN: str = Field(..., description="Specimen type (Blood, Sputum, Urine, CSF, etc.)")
    MB_TEST: str = Field(..., description="Test type (Culture, Gram stain, PCR, etc.)")
    MB_DAT: date = Field(..., description="Test date")
    MB_RESULT: str = Field(..., description="Result (No growth, Positive, Pending, etc.)")
    MB_ORGANISM: Optional[str] = Field(default=None, description="Identified organism (if positive)")
    MB_SENSITIVITY: Optional[str] = Field(default=None, description="Susceptibility summary (if positive)")


class MBDomain(BaseModel):
    """Microbiology results domain."""
    records: list[MBRecord] = Field(default_factory=list)


class ConsultRecord(BaseModel):
    """Single specialist consultation record."""
    CONSULT_SPECIALTY: str = Field(..., description="Specialty (Pulmonology, Infectious Disease, etc.)")
    CONSULT_DAT: date = Field(..., description="Consultation date")
    CONSULT_IMPRESSION: str = Field(..., description="Consultation impression and recommendations")
    CONSULT_PHYSICIAN: Optional[str] = Field(default=None, description="Consultant physician name")


class ConsultDomain(BaseModel):
    """Specialist consultation domain."""
    records: list[ConsultRecord] = Field(default_factory=list)


class DDDomain(BaseModel):
    """Death details domain."""
    DTHDAT: Optional[date] = Field(default=None, description="Date of death")
    PRCDTH: Optional[str] = Field(default=None, description="Primary cause of death")
    AUTOPIND: Optional[Literal["Y", "N"]] = Field(default=None, description="Autopsy performed?")


class DADomain(BaseModel):
    """Drug accountability domain."""
    LOT_NUMBER: Optional[str] = Field(default=None, description="Drug lot number")
    EXPIRY_DATE: Optional[date] = Field(default=None, description="Drug expiry date")


class InvestigatorInfo(BaseModel):
    """Investigator / Principal Investigator information."""
    name: str = Field(default="", description="Investigator name")
    institution: str = Field(default="", description="Institution/site name")
    address: str = Field(default="", description="Site address")
    phone: str = Field(default="", description="Phone number")
    email: str = Field(default="", description="Email address")


class CRFData(BaseModel):
    """Root model combining all CRF domains + investigator info."""
    dm: DMDomain
    ae: AEDomain
    ec: list[ECDomain] = Field(..., description="Exposure history (list for rechallenge)")
    lb: LBDomain = Field(default_factory=LBDomain)
    cm: CMDomain = Field(default_factory=CMDomain)
    mh: MHDomain = Field(default_factory=MHDomain)
    vs: VSDomain = Field(default_factory=VSDomain)
    dd: DDDomain = Field(default_factory=DDDomain)
    da: DADomain = Field(default_factory=DADomain)
    imaging: ImagingDomain = Field(default_factory=ImagingDomain)
    pft: PFTDomain = Field(default_factory=PFTDomain)
    microbiology: MBDomain = Field(default_factory=MBDomain)
    consultation: ConsultDomain = Field(default_factory=ConsultDomain)
    investigator: InvestigatorInfo = Field(default_factory=InvestigatorInfo)
