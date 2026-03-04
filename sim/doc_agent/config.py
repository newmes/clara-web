"""Doc Agent configuration — aligned with ClinicalTrialEngine env."""
from __future__ import annotations

import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Doc Agent configuration.

    Env vars loaded from project root .env with CTE_ prefix.
    Drug/protocol fields are settable per-simulation via constructor kwargs.
    """

    # ── LLM backend (MedGemma via vLLM) ──
    VLLM_BASE_URL: str = Field(
        default="http://localhost:38002/v1",
        description="vLLM OpenAI-compatible API base URL",
    )
    VLLM_MODEL_ID: str = Field(
        default="medgemma-4b-antihallu",
        description="Model ID served by vLLM",
    )
    VLLM_API_KEY: str = Field(
        default="EMPTY",
        description="vLLM API key (usually EMPTY for local)",
    )

    # Fallback: Gemini API via GOOGLE_API_KEY (for when vLLM is unavailable)
    GOOGLE_API_KEY: str = Field(
        default_factory=lambda: os.environ.get("GOOGLE_API_KEY", ""),
        description="Google API key (fallback if vLLM unavailable)",
    )

    # ── Generation parameters ──
    MAX_TOKENS_NARRATIVE: int = Field(default=1024)
    TEMPERATURE_NARRATIVE: float = Field(default=0.3)
    MAX_TOKENS_STRUCTURED: int = Field(default=512)
    TEMPERATURE_STRUCTURED: float = Field(default=0.1)

    # ── Protocol / Drug info (overridable per-simulation) ──
    DRUG_NAME: str = Field(default="Enfortumab vedotin (Padcev)")
    DRUG_MANUFACTURER: str = Field(default="Astellas / Seagen")
    INDICATION: str = Field(default="Locally advanced or metastatic urothelial carcinoma")
    PRODUCT_TYPE: str = Field(default="Biologic (ADC)")
    PROTOCOL_NUMBER: str = Field(default="CTE-SIM-2026-001")
    IND_NUMBER: str = Field(default="IND-999999")
    IND_TYPE: str = Field(default="IND")

    # ── Sponsor information ──
    SPONSOR_NAME: str = Field(default="ClinicalTrialEngine Research")
    SPONSOR_CONTACT: str = Field(default="Drug Safety Department, ClinicalTrialEngine Research")
    SPONSOR_ADDRESS: str = Field(default="100 Clinical Drive, Cambridge, MA 02142")
    SPONSOR_PHONE: str = Field(default="+1-617-555-0100")
    SPONSOR_EMAIL: str = Field(default="drugsafety@cte-research.example.com")

    # ── Reporter defaults ──
    REPORTER_QUALIFICATION: str = Field(default="Physician")
    REPORT_SOURCE: str = Field(default="Study")

    model_config = {
        "env_prefix": "CTE_",
        "env_file": ".env",
        "extra": "ignore",
    }

    @classmethod
    def from_simulation(
        cls,
        drug_name: str,
        indication: str,
        manufacturer: str = "",
        **overrides,
    ) -> "Settings":
        """Create Settings with simulation-specific drug info."""
        return cls(
            DRUG_NAME=drug_name,
            INDICATION=indication,
            DRUG_MANUFACTURER=manufacturer or drug_name.split("(")[0].strip(),
            **overrides,
        )
