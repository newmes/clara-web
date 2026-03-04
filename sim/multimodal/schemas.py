"""Input/output dataclasses for the multimodal API.

All schemas are plain dataclasses with dict() conversion support.
Field names and conventions follow the simulation engine
(see ``data/runs/*/patients/*.json`` and ``data/runs/*/simulations/*.jsonl``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Simulation-compatible patient profile adapter
# ---------------------------------------------------------------------------

@dataclass
class SimPatientProfile:
    """Flat patient profile extracted from the simulation's patients/*.json.

    Construct via :meth:`from_sim_patient` to handle path/format differences
    between the simulation JSON and the multimodal API.
    """
    patient_id: str
    age: int
    sex: str        # "M" | "F" (simulation convention)
    race: str       # e.g. "White", "Asian"
    bmi: float = 27.0
    persona_type: str = ""
    appearance: str = ""

    @classmethod
    def from_sim_patient(cls, patient_json: dict[str, Any]) -> "SimPatientProfile":
        """Build from a raw ``patients/PT-XXX.json`` dict."""
        emr = patient_json.get("emr", {})
        demo = emr.get("demographics", {})
        dm = patient_json.get("DM", {})
        persona = patient_json.get("persona", {})
        return cls(
            patient_id=patient_json.get("patient_id", ""),
            age=demo.get("age") or dm.get("AGE", 50),
            sex=(demo.get("sex") or dm.get("SEX", "M")).upper()[:1],
            race=demo.get("race") or dm.get("RACE", "White"),
            bmi=demo.get("bmi", 27.0),
            persona_type=persona.get("type", ""),
        )

    def to_voice_profile(self) -> dict[str, Any]:
        """Return the minimal dict that ``_select_voice`` expects."""
        return {"age": self.age, "sex": self.sex}

    def to_face_profile(self) -> dict[str, Any]:
        """Return the dict that ``_build_baseline_prompt`` expects."""
        sex_word = {"M": "male", "F": "female"}.get(self.sex, "male")
        return {
            "age": self.age,
            "sex": sex_word,
            "race": self.race.lower(),
            "bmi": self.bmi,
            "persona_type": self.persona_type,
            "appearance": self.appearance,
        }


# ---------------------------------------------------------------------------
# Simulation AE record adapter
# ---------------------------------------------------------------------------

@dataclass
class SimAE:
    """A single AE extracted from simulation day data.

    Works with both CDASH ``AE[]`` records (``AETERM`` / ``_grade``)
    and ``hospital_record.active_aes[]`` records (``ae`` / ``grade``).
    """
    ae_term: str        # e.g. "rash_maculopapular" — simulation convention
    grade: int
    onset_day: int = 0
    status: str = "active"
    days_active: int = 0
    visual: str | None = None

    @classmethod
    def from_cdash(cls, cdash_ae: dict[str, Any]) -> "SimAE":
        """Build from a CDASH ``AE[]`` record in the day JSONL."""
        return cls(
            ae_term=cdash_ae.get("AETERM", ""),
            grade=int(cdash_ae.get("_grade", 1)),
            onset_day=int(cdash_ae.get("AESTDAT", 0)),
            status=cdash_ae.get("_status", "active"),
            days_active=int(cdash_ae.get("_days_active", 0)),
            visual=cdash_ae.get("_visual"),
        )

    @classmethod
    def from_hr_active(cls, hr_ae: dict[str, Any]) -> "SimAE":
        """Build from ``hospital_record.objective.active_aes[]``."""
        return cls(
            ae_term=hr_ae.get("ae", ""),
            grade=int(hr_ae.get("grade", 1)),
            onset_day=int(hr_ae.get("onset_day", 0)),
            status=hr_ae.get("status", "active"),
        )

    def to_face_ae(self) -> dict[str, Any]:
        """Return the dict that ``_build_ae_edit_prompt`` expects."""
        return {"ae": self.ae_term, "grade": self.grade}


# ---------------------------------------------------------------------------
# Analysis result atoms
# ---------------------------------------------------------------------------

@dataclass
class DetectedAE:
    """A single adverse-event detection from face or voice analysis.

    ``ae_term`` uses the simulation convention (e.g. ``rash_maculopapular``).
    """
    ae_term: str                # e.g. "rash_maculopapular"
    grade: int                  # CTCAE grade 1-3+
    confidence: float           # 0.0-1.0
    reasoning: str              # model's reasoning text
    channel: str                # "face" | "voice"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CoughEvent:
    """A single cough event detected in audio."""
    timestamp_sec: float        # seconds from audio start
    cough_type: str             # "dry" | "productive" | "wheezing"
    severity: str               # "mild" | "moderate" | "severe"
    confidence: float           # 0.0-1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Face pipeline results
# ---------------------------------------------------------------------------

@dataclass
class FaceGenerationResult:
    """Result from generate_patient_face()."""
    image_bytes: bytes          # PNG image data
    prompt_used: str            # final prompt sent to the model
    ae_applied: list[dict]      # list of {ae_term, grade} applied
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # bytes not JSON-serializable — report length instead
        d["image_bytes"] = f"<{len(self.image_bytes)} bytes>"
        return d


@dataclass
class FaceAnalysisResult:
    """Result from analyze_face()."""
    detected_aes: list[DetectedAE]
    model_used: str             # "medgemma" | "medsiglip"
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected_aes": [ae.to_dict() for ae in self.detected_aes],
            "model_used": self.model_used,
            "latency_ms": self.latency_ms,
        }


# ---------------------------------------------------------------------------
# Voice pipeline results
# ---------------------------------------------------------------------------

@dataclass
class VoiceGenerationResult:
    """Result from generate_patient_voice()."""
    audio_bytes: bytes          # WAV PCM data (mono 24kHz 16-bit)
    duration_sec: float
    transcript: str             # text that was spoken
    cough_inserted: bool        # whether real cough clips were inserted
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["audio_bytes"] = f"<{len(self.audio_bytes)} bytes>"
        return d


@dataclass
class RespiratoryAssessment:
    """Structured respiratory assessment."""
    has_cough: bool
    has_wheeze: bool
    has_dyspnea: bool
    overall_severity: str          # "normal" | "mild" | "moderate" | "severe"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VoiceAnalysisResult:
    """Result from analyze_voice()."""
    transcript: str
    cough_events: list[CoughEvent]
    respiratory_assessment: RespiratoryAssessment
    model_used: str               # "gemini" | "hear"
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "transcript": self.transcript,
            "cough_events": [e.to_dict() for e in self.cough_events],
            "respiratory_assessment": self.respiratory_assessment.to_dict(),
            "model_used": self.model_used,
            "latency_ms": self.latency_ms,
        }
