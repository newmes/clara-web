"""Patient face image generation with CTCAE-graded AE overlays.

Public API
----------
- ``generate_patient_face(patient_profile, active_aes, day, *, config)``
  → :class:`~schemas.FaceGenerationResult`

Re-uses the rate-limited Gemini image client pattern from
``image/face/src/api_client.py``.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .config import MultimodalConfig, CTCAE_CRITERIA, FACE_RENDERABLE_AES, build_ctcae_table_text, get_config
from .schemas import FaceGenerationResult, SimPatientProfile, SimAE

logger = logging.getLogger(__name__)


# =====================================================================
# Rate limiter & Gemini client (adapted from image/face/src/api_client.py)
# =====================================================================

class RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, requests_per_minute: int):
        self._interval = 60.0 / max(1, requests_per_minute)
        self._last_request = 0.0

    def wait(self) -> None:
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self._interval:
            sleep_time = self._interval - elapsed
            logger.debug("Rate limiting: sleeping %.1fs", sleep_time)
            time.sleep(sleep_time)
        self._last_request = time.time()


class APIError(Exception):
    pass


class SafetyFilterError(APIError):
    pass


class GeminiImageClient:
    """Gemini image generation: text→image and image+text→image."""

    def __init__(self, config: MultimodalConfig):
        from google import genai as _genai

        self._client = _genai.Client(api_key=config.gemini_api_key)
        self._model = config.image_gen_model
        self._rate_limiter = RateLimiter(config.image_rate_limit_rpm)
        self._max_retries = config.image_max_retries
        self._backoff_base = config.image_backoff_base

    def generate_from_text(self, prompt: str) -> Image.Image:
        from google.genai import types

        return self._call_with_retry(
            contents=[prompt],
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )

    def edit_image(self, image: Image.Image, prompt: str) -> Image.Image:
        from google.genai import types

        return self._call_with_retry(
            contents=[prompt, image],
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )

    def _call_with_retry(self, contents: list, config) -> Image.Image:
        last_error = None
        for attempt in range(self._max_retries):
            self._rate_limiter.wait()
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=config,
                )
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data is not None:
                            return Image.open(io.BytesIO(part.inline_data.data))
                raise APIError("No image in response")

            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if "safety" in error_str or "blocked" in error_str:
                    logger.warning("Safety filter triggered (attempt %d)", attempt + 1)
                    if isinstance(contents[0], str):
                        contents[0] = _soften_prompt(contents[0])
                    raise SafetyFilterError(f"Safety filter: {e}") from e

                wait_time = self._backoff_base ** attempt
                logger.warning(
                    "API error (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1, self._max_retries, e, wait_time,
                )
                time.sleep(wait_time)

        raise APIError(f"All {self._max_retries} retries failed. Last error: {last_error}")


def _soften_prompt(prompt: str) -> str:
    """Replace medical terms that trigger safety filters."""
    replacements = {
        "cyanosis": "bluish skin discoloration",
        "cyanotic": "slightly bluish",
        "diaphoresis": "perspiration",
        "vesicles": "small bumps",
        "blistering": "skin texture changes",
        "bleeding": "reddened",
        "labored breathing": "slightly winded",
        "distension": "slight prominence",
        "mottled": "uneven tone",
        "desquamating": "dry flaking",
    }
    result = prompt
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


# =====================================================================
# Prompt builders
# =====================================================================

def _normalize_profile(profile) -> dict[str, Any]:
    """Accept SimPatientProfile, raw sim patient JSON, or flat dict."""
    if isinstance(profile, SimPatientProfile):
        return profile.to_face_profile()
    if "emr" in profile:
        return SimPatientProfile.from_sim_patient(profile).to_face_profile()
    sex_raw = str(profile.get("sex", "male")).lower()
    if sex_raw in ("m", "male"):
        sex_word = "male"
    elif sex_raw in ("f", "female"):
        sex_word = "female"
    else:
        sex_word = sex_raw
    return {
        "age": profile.get("age", 50),
        "sex": sex_word,
        "race": str(profile.get("race", "asian")).lower(),
        "bmi": profile.get("bmi", 27.0),
        "persona_type": profile.get("persona_type", ""),
        "appearance": profile.get("appearance", ""),
    }


def _build_baseline_prompt(profile: dict[str, Any]) -> str:
    """Build text-to-image prompt for a healthy baseline face."""
    age = profile.get("age", 50)
    sex = profile.get("sex", "female")
    race = profile.get("race", "asian")
    appearance = profile.get("appearance", "")
    bmi = profile.get("bmi", 27.0)

    build_hint = ""
    if bmi and bmi > 30:
        build_hint = "Overweight build, fuller face."
    elif bmi and bmi < 20:
        build_hint = "Thin build, lean face."

    lines = [
        f"A photorealistic close-up face photograph of a {age}-year-old {race} {sex}",
        "cancer patient at home, taken from a webcam.",
        "This is a BASELINE photograph taken BEFORE starting treatment.",
        "The patient has normal full hair with no hair loss or thinning.",
        "Neutral, calm expression. Skin appears normal and healthy.",
        "Natural indoor lighting. Face fills most of the frame, looking directly at the camera.",
        "No hats, no head coverings.",
        "No screens, no monitors, no laptop frames, no UI elements, no text overlays.",
    ]
    if build_hint:
        lines.append(build_hint)
    if appearance:
        lines.append(appearance)
    return " ".join(lines)


def _normalize_aes(raw_aes: list) -> list[dict[str, Any]]:
    """Accept list of SimAE, CDASH AE dicts, or flat {ae, grade} dicts.

    Returns only face-renderable AEs (those with CTCAE visual criteria).
    """
    out = []
    for ae in raw_aes:
        if isinstance(ae, SimAE):
            term, grade = ae.ae_term, ae.grade
        elif "AETERM" in ae:
            term = ae["AETERM"]
            grade = int(ae.get("_grade", 1))
        else:
            term = ae.get("ae", ae.get("ae_term", ""))
            grade = int(ae.get("grade", 1))
        if term in FACE_RENDERABLE_AES:
            out.append({"ae": term, "grade": grade})
    return out


def _build_daily_state_prompt(day: int, mood_snapshot: dict[str, float] | None = None) -> str:
    """Build an edit prompt for daily appearance changes (no specific AE)."""
    fatigue_week = min(day // 7, 12)
    energy = (mood_snapshot or {}).get("energy", 0.6)
    depression = (mood_snapshot or {}).get("depression", 0.3)
    anxiety = (mood_snapshot or {}).get("anxiety", 0.3)

    lines = [f"Edit this patient's face to reflect day {day} of cancer chemotherapy treatment."]

    if fatigue_week <= 1:
        lines.append("Mild fatigue starting to show. Slightly tired eyes.")
    elif fatigue_week <= 4:
        lines.append("Moderate treatment fatigue. Slightly sunken eyes, mild pallor.")
    else:
        lines.append("Cumulative chemotherapy fatigue. Noticeable pallor, tired expression, slightly thinner face.")

    if energy < 0.3:
        lines.append("Very low energy — drooping eyelids, dull gaze.")
    elif energy < 0.5:
        lines.append("Low energy — slightly heavy eyelids.")

    if depression > 0.6:
        lines.append("Flat, sad expression. Downturned mouth corners.")
    if anxiety > 0.6:
        lines.append("Tense facial muscles, furrowed brow.")

    lines.append(
        "Maintain the patient's identity, age, ethnicity, and home setting. "
        "Changes should be subtle and realistic."
    )
    return " ".join(lines)


def _build_ae_edit_prompt(
    active_aes: list[dict[str, Any]],
    day: int,
    mood_snapshot: dict[str, float] | None = None,
) -> str:
    """Build the image-edit prompt with CTCAE criteria directly embedded."""
    ae_specs = []
    for ae in active_aes:
        ae_term = ae.get("ae", ae.get("ae_term", ""))
        grade = min(ae.get("grade", 1), 3)
        criteria = CTCAE_CRITERIA.get(ae_term, {})
        desc = criteria.get(grade)
        if desc:
            ae_specs.append(f"- {ae_term} Grade {grade}: {desc}")
        else:
            logger.warning("No CTCAE criteria for ae=%s grade=%d", ae_term, grade)

    ae_spec_block = "\n".join(ae_specs)

    energy = (mood_snapshot or {}).get("energy", 0.6)
    fatigue_hint = ""
    if energy < 0.3:
        fatigue_hint = "The patient appears very fatigued with drooping eyelids and dull gaze. "
    elif energy < 0.5:
        fatigue_hint = "The patient appears tired with slightly heavy eyelids. "

    return (
        f"Edit this patient's face image to show adverse events on day {day} of treatment.\n"
        f"\n"
        f"## CTCAE Grading Reference\n"
        f"{build_ctcae_table_text()}\n"
        f"## Requested AEs to render\n"
        f"{ae_spec_block}\n"
        f"\n"
        f"Visually render EXACTLY the clinical findings described by the CTCAE criteria above "
        f"on the patient's face. The severity must match the specified grade — "
        f"do not over- or under-represent. "
        f"{fatigue_hint}"
        f"The patient is on day {day} of chemotherapy, so general fatigue and slight pallor are expected. "
        f"Maintain the patient's identity, age, ethnicity, and home setting. "
        f"Only modify facial appearance to show the described clinical findings."
    )


# =====================================================================
# Baseline cache
# =====================================================================

def _profile_hash(profile: dict[str, Any]) -> str:
    canonical = json.dumps(profile, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _load_cached_baseline(cache_dir: Path, profile_hash: str) -> Image.Image | None:
    path = cache_dir / f"{profile_hash}.png"
    if path.exists():
        logger.info("Baseline cache hit: %s", path)
        return Image.open(path)
    return None


def _save_cached_baseline(cache_dir: Path, profile_hash: str, image: Image.Image) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{profile_hash}.png"
    image.save(path, format="PNG")
    logger.info("Baseline cached: %s", path)


# =====================================================================
# Module-level client cache
# =====================================================================

_client: GeminiImageClient | None = None
_client_key: str | None = None


def _get_client(config: MultimodalConfig) -> GeminiImageClient:
    global _client, _client_key
    if _client is None or _client_key != config.gemini_api_key:
        _client = GeminiImageClient(config)
        _client_key = config.gemini_api_key
    return _client


# =====================================================================
# Public API
# =====================================================================

def generate_patient_face(
    patient_profile,
    active_aes: list | None = None,
    day: int = 0,
    *,
    mood_snapshot: dict[str, float] | None = None,
    config: MultimodalConfig | None = None,
) -> FaceGenerationResult:
    """Generate a patient face image reflecting the patient's daily state.

    Always edits the baseline for day > 1 to show treatment fatigue,
    mood, and any CTCAE-graded AE overlays.
    """
    cfg = config or get_config()
    client = _get_client(cfg)

    profile = _normalize_profile(patient_profile)
    visual_aes = _normalize_aes(active_aes or [])

    # 1. Baseline generation / cache lookup
    p_hash = _profile_hash(profile)
    baseline = _load_cached_baseline(cfg.cache_dir, p_hash)

    baseline_prompt = _build_baseline_prompt(profile)

    if baseline is None:
        logger.info("Generating baseline for profile hash=%s", p_hash)
        baseline = client.generate_from_text(baseline_prompt)
        _save_cached_baseline(cfg.cache_dir, p_hash, baseline)

    # 2. Day 0/1 with no visual AEs → return baseline as-is
    if not visual_aes and day <= 1:
        buf = io.BytesIO()
        baseline.save(buf, format="PNG")
        return FaceGenerationResult(
            image_bytes=buf.getvalue(),
            prompt_used=baseline_prompt,
            ae_applied=[],
            metadata={"profile_hash": p_hash, "cached": True},
        )

    # 3. Day > 1, no visual AEs → edit baseline for daily state (fatigue/mood)
    if not visual_aes:
        daily_prompt = _build_daily_state_prompt(day, mood_snapshot)
        full_prompt = _soften_prompt(daily_prompt)
        logger.info("Editing baseline for daily state (day=%d, no visual AEs)", day)
        result_image = client.edit_image(baseline, full_prompt)
        buf = io.BytesIO()
        result_image.save(buf, format="PNG")
        return FaceGenerationResult(
            image_bytes=buf.getvalue(),
            prompt_used=full_prompt,
            ae_applied=[],
            metadata={"profile_hash": p_hash, "day": day, "daily_edit": True},
        )

    # 4. AE overlay via image editing
    ae_edit_prompt = _build_ae_edit_prompt(visual_aes, day, mood_snapshot)
    full_prompt = _soften_prompt(ae_edit_prompt)
    logger.info("Editing baseline with %d visual AE(s) (day=%d)", len(visual_aes), day)
    result_image = client.edit_image(baseline, full_prompt)

    buf = io.BytesIO()
    result_image.save(buf, format="PNG")

    return FaceGenerationResult(
        image_bytes=buf.getvalue(),
        prompt_used=full_prompt,
        ae_applied=[{"ae_term": a["ae"], "grade": a["grade"]} for a in visual_aes],
        metadata={"profile_hash": p_hash, "day": day},
    )
