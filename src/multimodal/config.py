"""Centralised configuration for the multimodal API.

All tuneable knobs live in :class:`MultimodalConfig`.  Call
:func:`get_config` for a sensible default instance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Voice-name mapping: (age_bucket, sex) → Gemini TTS voice
# ---------------------------------------------------------------------------

VOICE_MAP: dict[tuple[str, str], str] = {
    # young (18-39)
    ("young", "male"):   "Puck",
    ("young", "female"): "Kore",
    ("young", "m"):      "Puck",
    ("young", "f"):      "Kore",
    # middle (40-64)
    ("middle", "male"):   "Charon",
    ("middle", "female"): "Aoede",
    ("middle", "m"):      "Charon",
    ("middle", "f"):      "Aoede",
    # elderly (65+)
    ("elderly", "male"):   "Orus",
    ("elderly", "female"): "Leda",
    ("elderly", "m"):      "Orus",
    ("elderly", "f"):      "Leda",
}

# ---------------------------------------------------------------------------
# Cough-frequency mapping: label → (min_count, max_count)
# ---------------------------------------------------------------------------

COUGH_FREQUENCY: dict[str, tuple[int, int]] = {
    "none":       (0, 0),
    "occasional": (1, 2),
    "frequent":   (3, 5),
    "severe":     (6, 8),
}

# ---------------------------------------------------------------------------
# Cough-type mapping: semantic label → clip subdirectory
# ---------------------------------------------------------------------------

COUGH_TYPE_DIR: dict[str, str] = {
    "dry":        "dry",
    "productive": "wet",
    "wheezing":   "dry",  # closest available
}

# ---------------------------------------------------------------------------
# Cough severity profile: frequency → (overlay_volume, max_cough_sec)
#   overlay_volume: 0.0–1.0, controls how loud the cough sounds
#   max_cough_sec:  max clip duration — mild coughs are short, severe are long
# ---------------------------------------------------------------------------

COUGH_SEVERITY: dict[str, tuple[float, float]] = {
    "none":       (0.0, 0.0),
    "occasional": (0.9, 1.5),   # 약한 기침, 짧은 클립
    "frequent":   (1.0, 2.5),   # 보통 기침
    "severe":     (1.2, 3.5),   # 격렬한 기침, 긴 클립
}


# ---------------------------------------------------------------------------
# CTCAE grading criteria (single source of truth for generation & analysis)
# ---------------------------------------------------------------------------

CTCAE_CRITERIA: dict[str, dict[int, str]] = {
    # ---------------------------------------------------------------
    # Key naming follows the simulation engine's ae_term convention
    # (snake_case, noun-first: rash_maculopapular, not maculopapular_rash)
    # ---------------------------------------------------------------
    "rash_maculopapular": {
        1: (
            "Faint pink macules/papules scattered on cheeks (<10% BSA); "
            "mild erythema, no scaling, subtle and localized"
        ),
        2: (
            "Visible red macules/papules spreading across cheeks and forehead (10–30% BSA); "
            "moderate erythema with fine scaling at lesion edges"
        ),
        3: (
            "Severe confluent rash covering entire face including cheeks, forehead, chin, and nose (>30% BSA); "
            "intense erythema, coarse scaling, and facial edema"
        ),
    },
    "rash_acneiform": {
        1: (
            "Few small papules on forehead (<10% BSA); "
            "non-inflamed or mildly inflamed, skin-colored to pink"
        ),
        2: (
            "Multiple erythematous papules and pustules on cheeks and forehead (10–30% BSA); "
            "visible pus-filled lesions, surrounding redness"
        ),
        3: (
            "Dense pustules covering entire face — forehead, cheeks, nose, chin (>30% BSA); "
            "confluent inflammation, crusting, signs of secondary infection"
        ),
    },
    "periorbital_edema": {
        1: (
            "Slight puffiness of upper and lower eyelids, barely noticeable; "
            "periorbital skin appears mildly swollen"
        ),
        2: (
            "Obvious bilateral periorbital swelling; puffy, baggy eyelids with visible tissue distension; "
            "eyes appear partially narrowed"
        ),
        3: (
            "Severe periorbital edema causing near-closure of eyes; "
            "tense, shiny skin around orbital rims, eye-opening significantly impaired"
        ),
    },
    "sjs_prodrome": {
        1: (
            "Lip redness and dryness; vermilion border appears erythematous, "
            "slight chapping without blistering"
        ),
        2: (
            "Lip and oral mucosal blistering; fluid-filled vesicles on lip surface and inner mouth; "
            "erosions with crusting at lip margins"
        ),
        3: (
            "Beginning of epidermal detachment on lips and perioral skin; "
            "large erosions, bleeding mucosa, severe crusting extending beyond lip borders"
        ),
    },
    "stomatitis": {
        1: (
            "Mild redness or minor aphthous-like ulcer on lip mucosa; "
            "slight discomfort, no visible swelling from outside"
        ),
        2: (
            "Visible cracking and erythema at lip corners with shallow erosions; "
            "perioral redness, mild swelling of the lips"
        ),
        3: (
            "Severe lip swelling and deep erosions visible on external lip surface; "
            "crusting, bleeding, perioral inflammation"
        ),
    },
    "pruritus": {
        1: (
            "Mild localized skin excoriation marks on forehead or cheeks; "
            "faint scratch marks, minimal erythema"
        ),
        2: (
            "Moderate visible scratch marks and erythema across face; "
            "dry, irritated skin with diffuse redness"
        ),
        3: (
            "Severe widespread excoriations with lichenification; "
            "intense erythema, bleeding scratch marks, facial edema from chronic scratching"
        ),
    },
    "alopecia": {
        1: (
            "Mild hair thinning visible at temples and frontal hairline; "
            "slightly widened part line, subtle compared to baseline"
        ),
        2: (
            "Obvious diffuse hair thinning with clearly visible scalp through hair; "
            "temporal recession, noticeably sparse hair"
        ),
    },
}


# ---------------------------------------------------------------------------
# AE → face-renderable filter (only these AEs have visual CTCAE criteria)
# ---------------------------------------------------------------------------

FACE_RENDERABLE_AES: set[str] = set(CTCAE_CRITERIA.keys())

# ---------------------------------------------------------------------------
# AE → cough config mapping (for voice generation)
# These AEs produce cough symptoms when present.
# ---------------------------------------------------------------------------

AE_COUGH_MAP: dict[str, dict[str, str]] = {
    "pneumonitis": {"type": "dry", "frequency_g1": "occasional", "frequency_g2": "frequent", "frequency_g3": "severe"},
}

# ---------------------------------------------------------------------------
# Respiratory AE terms (triggers breathiness / fatigue in TTS style)
# ---------------------------------------------------------------------------

RESPIRATORY_AES: set[str] = {"pneumonitis", "fatigue", "anemia"}


def build_ctcae_table_text() -> str:
    """Render the full CTCAE criteria as human-readable text."""
    lines = []
    for ae_term, grades in CTCAE_CRITERIA.items():
        lines.append(f"### {ae_term}")
        for grade, desc in sorted(grades.items()):
            lines.append(f"- Grade {grade}: {desc}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MedSigLIP label set (13 classes)
# ---------------------------------------------------------------------------

SIGLIP_CLASSES: list[str] = [
    "normal",
    "rash_maculopapular_g1", "rash_maculopapular_g2", "rash_maculopapular_g3",
    "rash_acneiform_g1",     "rash_acneiform_g2",     "rash_acneiform_g3",
    "periorbital_edema_g1",  "periorbital_edema_g2",  "periorbital_edema_g3",
    "sjs_prodrome_g1",       "sjs_prodrome_g2",       "sjs_prodrome_g3",
    "stomatitis_g1",         "stomatitis_g2",         "stomatitis_g3",
    "pruritus_g1",           "pruritus_g2",           "pruritus_g3",
    "alopecia_g1",           "alopecia_g2",
]


@dataclass
class MultimodalConfig:
    """Single source of truth for every tunable in the multimodal API."""

    # --- API key (image generation + TTS) ---
    # Reads GEMINI_API_KEY first, falls back to GOOGLE_API_KEY (.env)
    gemini_api_key: str = field(
        default_factory=lambda: (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY", "")
        )
    )

    # --- Image generation ---
    image_gen_model: str = "gemini-3-pro-image-preview"
    image_rate_limit_rpm: int = 10
    image_max_retries: int = 5
    image_backoff_base: float = 2.0
    image_width: int = 768
    image_height: int = 768

    # --- TTS ---
    tts_model: str = "gemini-2.5-flash-preview-tts"

    # --- HuggingFace token (for gated models like MedGemma) ---
    hf_token: str = field(
        default_factory=lambda: os.environ.get(
            "HF_TOKEN",
            "",
        )
    )

    # --- MedGemma 27B (face analysis — local VLM) ---
    medgemma_model: str = "google/medgemma-27b-it"
    medgemma_device: str = "auto"  # device_map="auto" for 27B
    medgemma_gpus: list[int] = field(default_factory=lambda: [6, 7])  # GPUs to use

    # --- MedSigLIP (face analysis — local encoder + classifier) ---
    siglip_model: str = "google/medsiglip-448"
    siglip_device: str = "cuda:4"
    siglip_num_classes: int = 21
    siglip_image_size: int = 448

    # --- HeAR (voice analysis — local TF/Keras) ---
    hear_model: str = "google/hear"
    hear_sample_rate: int = 16000
    hear_clip_sec: float = 2.0
    hear_embed_dim: int = 512

    # --- Voice mapping ---
    voice_map: dict[tuple[str, str], str] = field(default_factory=lambda: dict(VOICE_MAP))
    nurse_voice: str = "Kore"

    # --- Cough ---
    cough_clips_dir: Path = field(
        default_factory=lambda: (
            Path(__file__).resolve().parent.parent.parent / "data" / "cough_clips_clean"
        )
    )
    cough_frequency: dict[str, tuple[int, int]] = field(
        default_factory=lambda: dict(COUGH_FREQUENCY)
    )
    cough_type_dir: dict[str, str] = field(
        default_factory=lambda: dict(COUGH_TYPE_DIR)
    )

    # --- Baseline cache ---
    cache_dir: Path = field(
        default_factory=lambda: (
            Path(__file__).resolve().parent.parent.parent / "cache" / "baseline"
        )
    )

    # --- Audio ---
    sample_rate: int = 24000
    silence_padding_sec: float = 0.15
    max_cough_sec: float = 3.0


def get_config(**overrides) -> MultimodalConfig:
    """Return a default :class:`MultimodalConfig`, with optional overrides."""
    return MultimodalConfig(**overrides)
