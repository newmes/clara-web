"""MedWatch FDA Form 3500A field models."""
from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator


def _empty_str_to_none(v):
    """Convert empty strings to None for Optional[date] fields."""
    if v == "" or v is None:
        return None
    return v


class SectionA(BaseModel):
    """Section A — Patient Information (5 fields)."""
    patient_id: str = Field(default="", description="A1: Patient identifier")
    age: Optional[int] = Field(default=None, description="A2: Age at event")
    sex: str = Field(default="", description="A3: Sex")
    weight: Optional[float] = Field(default=None, description="A4: Weight (kg)")
    ethnicity: str = Field(default="", description="A5a: Ethnicity (Hispanic/Not Hispanic)")
    race: str = Field(default="", description="A5b: Race (Asian, Black, White, etc.)")

    _fix_nums = field_validator("age", "weight", mode="before")(_empty_str_to_none)


class SectionB(BaseModel):
    """Section B — Adverse Event or Product Problem (7 fields)."""
    report_type: str = Field(default="Adverse Event", description="B1: Report type")
    # Seriousness criteria
    seriousness_death: bool = Field(default=False, description="B2: Death")
    seriousness_life_threatening: bool = Field(default=False, description="B2: Life-threatening")
    seriousness_hospitalization: bool = Field(default=False, description="B2: Hospitalization")
    hospitalization_start: Optional[date] = Field(default=None, description="B2: Hospitalization start date")
    hospitalization_end: Optional[date] = Field(default=None, description="B2: Hospitalization end date")
    seriousness_disability: bool = Field(default=False, description="B2: Disability")
    seriousness_congenital: bool = Field(default=False, description="B2: Congenital anomaly")
    seriousness_other: bool = Field(default=False, description="B2: Other medically important")
    death_date: Optional[date] = Field(default=None, description="B2: Date of death")
    onset_date: Optional[date] = Field(default=None, description="B3: Date of event onset")
    report_date: Optional[date] = Field(default=None, description="B4: Date of this report")
    outcome: str = Field(default="", description="B4a: Outcome (recovered, recovering, not recovered, fatal, unknown)")
    narrative: str = Field(default="", description="B5: Describe event (AI-generated)")
    lab_data: str = Field(default="", description="B6: Relevant tests/lab data")
    medical_history: str = Field(default="", description="B7: Other relevant medical history")

    _fix_dates = field_validator(
        "hospitalization_start", "hospitalization_end", "death_date",
        "onset_date", "report_date", mode="before",
    )(_empty_str_to_none)


class SectionC(BaseModel):
    """Section C — Suspect Product(s) (9 fields)."""
    drug_name: str = Field(default="", description="C1: Name, strength, manufacturer")
    dose_frequency_route: str = Field(default="", description="C2: Dose, frequency, route")
    therapy_start: Optional[date] = Field(default=None, description="C3: Therapy start date")
    therapy_end: Optional[date] = Field(default=None, description="C3: Therapy end date")
    indication: str = Field(default="", description="C4: Diagnosis/reason for use")
    product_type: str = Field(default="", description="C5: Type of product")
    lot_number: str = Field(default="", description="C1: Lot number (required for biologics)")
    expiry_date: Optional[date] = Field(default=None, description="C6: Expiration date")
    dechallenge: str = Field(default="", description="C7: Did reaction abate? (AI-generated)")
    rechallenge: str = Field(default="", description="C8: Did reaction reappear? (AI-generated)")
    concomitant_meds: str = Field(default="", description="C9: Concomitant medical products")

    _fix_dates = field_validator(
        "therapy_start", "therapy_end", "expiry_date", mode="before",
    )(_empty_str_to_none)


class SectionE(BaseModel):
    """Section E — Initial Reporter (6 fields)."""
    reporter_name: str = Field(default="", description="E1: Name")
    reporter_address: str = Field(default="", description="E2: Address")
    reporter_phone: str = Field(default="", description="E2: Phone number")
    reporter_email: str = Field(default="", description="E2: Email")
    reporter_qualification: str = Field(default="", description="E3: Health professional?")
    reported_to_fda: str = Field(default="No", description="E4: Also reported to FDA?")


class SectionG(BaseModel):
    """Section G — All Manufacturers (8 fields)."""
    sponsor_contact: str = Field(default="", description="G1: Contact office")
    source: str = Field(default="", description="G2: Source of report")
    awareness_date: Optional[date] = Field(default=None, description="G3: Date received")
    ind_type: str = Field(default="", description="G4: IND/NDA type")
    ind_number: str = Field(default="", description="G5: IND/NDA number")
    initial_followup: str = Field(default="Initial", description="G6: Initial or follow-up")
    ae_term: str = Field(default="", description="G7: MedWatch AE term (verbatim AETERM)")
    report_number: str = Field(default="", description="G8: Manufacturer report number")

    _fix_dates = field_validator("awareness_date", mode="before")(_empty_str_to_none)


class MedWatch3500A(BaseModel):
    """Complete MedWatch FDA Form 3500A."""
    section_a: SectionA = Field(default_factory=SectionA)
    section_b: SectionB = Field(default_factory=SectionB)
    section_c: SectionC = Field(default_factory=SectionC)
    section_e: SectionE = Field(default_factory=SectionE)
    section_g: SectionG = Field(default_factory=SectionG)
    ild_flag: bool = Field(default=False, description="ILD signal detected by Sentinel")
    non_serious_flag: bool = Field(default=False, description="Non-serious AE — internal tracking only")
