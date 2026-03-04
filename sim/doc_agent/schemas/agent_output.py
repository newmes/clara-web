"""Structured output schemas for Doc Agent AI components."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class DechallengeResult(BaseModel):
    """C7 Dechallenge structured output."""
    c7_answer: Literal["Yes", "No", "Does not apply", "Unknown"] = Field(
        ..., description="Dechallenge result"
    )
    c7_rationale: str = Field(..., description="One sentence explanation with dates")


class RechallengeResult(BaseModel):
    """C8 Rechallenge structured output."""
    c8_answer: Literal["Yes, recurred", "Yes, did not recur", "Does not apply"] = Field(
        ..., description="Rechallenge result"
    )
    c8_rationale: str = Field(..., description="One sentence explanation with dates")
    e2b_code: Literal[1, 2, 4] = Field(
        ..., description="E2B CL16 rechallenge code"
    )


class SentinelOutput(BaseModel):
    """Sentinel Agent ILD detection output."""
    ild_detected: bool = Field(default=False, description="ILD signal detected")
    ild_grade: Optional[int] = Field(default=None, description="ILD CTCAE grade (1-5)")
    ild_confidence: Optional[float] = Field(default=None, description="Detection confidence")
    cxr_findings: Optional[str] = Field(default=None, description="CXR findings description")
    differential_diagnosis: Optional[str] = Field(default=None, description="Differential diagnosis")
    kl6_value: Optional[float] = Field(default=None, description="KL-6 value (U/mL)")
    spo2_value: Optional[float] = Field(default=None, description="SpO2 percentage")


class MedDRACode(BaseModel):
    """MedDRA coding result."""
    pt: str = Field(..., description="Preferred Term")
    pt_code: Optional[str] = Field(default=None, description="PT code")
    soc: Optional[str] = Field(default=None, description="System Organ Class")
    llt: Optional[str] = Field(default=None, description="Lowest Level Term")
    confidence: Optional[float] = Field(default=None, description="Coding confidence (0.0-1.0)")
    source: Literal["lookup", "medgemma", "manual_review"] = Field(
        default="lookup", description="Coding source: lookup table, MedGemma inference, or flagged for manual review"
    )
