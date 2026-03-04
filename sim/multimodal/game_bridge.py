"""Bridge between game session and multimodal generation pipeline.

Creates face images and voice audio on demand during game play,
with caching to minimise API calls.
"""

from __future__ import annotations

import base64
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .config import FACE_RENDERABLE_AES, MultimodalConfig, get_config
from .schemas import SimPatientProfile, SimAE

logger = logging.getLogger(__name__)


class MultimodalGameBridge:
    """Per-session bridge that generates face + voice for each patient turn.

    Typical lifecycle::

        bridge = MultimodalGameBridge(patient_json)
        # ... on each patient turn ...
        media = bridge.generate_turn_media(text, active_aes, day)
        # media = {"face_b64": "...", "audio_b64": "...", "mm_meta": {...}}
    """

    def __init__(
        self,
        patient_json: dict[str, Any],
        *,
        config: MultimodalConfig | None = None,
        enabled: bool = True,
    ):
        self.config = config or get_config()
        self.profile = SimPatientProfile.from_sim_patient(patient_json)
        self.enabled = enabled

        self._baseline_face_b64: str | None = None
        self._last_face_day: int = -1
        self._last_face_ae_key: str = ""
        self._last_face_b64: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_turn_media(
        self,
        text: str,
        active_aes: list[dict[str, Any]],
        day: int,
        mood_snapshot: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Generate face image + voice audio for one patient turn.

        Returns a dict with keys that get merged into the game response:
        - ``face_b64``: base64-encoded PNG (or *None* on error)
        - ``audio_b64``: base64-encoded WAV (or *None* on error)
        - ``mm_meta``: metadata dict (timings, cache hits, etc.)
        """
        if not self.enabled:
            return {"face_b64": None, "audio_b64": None, "mm_meta": {"enabled": False}}

        meta: dict[str, Any] = {"enabled": True}
        face_b64: str | None = None
        audio_b64: str | None = None

        with ThreadPoolExecutor(max_workers=2) as pool:
            face_future = pool.submit(self._generate_face, active_aes, day, meta, mood_snapshot)
            audio_future = pool.submit(self._generate_voice, text, active_aes, meta, mood_snapshot)

            for future in as_completed([face_future, audio_future]):
                try:
                    result = future.result()
                    if "face_b64" in result:
                        face_b64 = result["face_b64"]
                    if "audio_b64" in result:
                        audio_b64 = result["audio_b64"]
                except Exception as exc:
                    logger.error("Multimodal generation error: %s", exc, exc_info=True)

        return {"face_b64": face_b64, "audio_b64": audio_b64, "mm_meta": meta}

    # ------------------------------------------------------------------
    # Face generation (with cache)
    # ------------------------------------------------------------------

    def _visual_ae_key(self, active_aes: list[dict[str, Any]]) -> str:
        """Compute a cache key from the set of face-renderable AEs."""
        visual = []
        for ae in active_aes:
            term = ae.get("AETERM", ae.get("ae", ae.get("ae_term", "")))
            grade = ae.get("_grade", ae.get("grade", 1))
            if term in FACE_RENDERABLE_AES:
                visual.append(f"{term}:{grade}")
        visual.sort()
        return "|".join(visual)

    def _generate_face(
        self,
        active_aes: list[dict[str, Any]],
        day: int,
        meta: dict,
        mood_snapshot: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        from .face_generator import generate_patient_face

        current_ae_key = self._visual_ae_key(active_aes)

        # Cache hit only when same day AND same AE set (within a single day's
        # multiple chat turns). New day always regenerates.
        if (day == self._last_face_day
                and current_ae_key == self._last_face_ae_key
                and self._last_face_b64):
            meta["face_cache"] = "hit"
            return {"face_b64": self._last_face_b64}

        try:
            result = generate_patient_face(
                patient_profile=self.profile,
                active_aes=active_aes,
                day=day,
                mood_snapshot=mood_snapshot,
                config=self.config,
            )
            b64 = base64.b64encode(result.image_bytes).decode("ascii")

            if day <= 1 and not current_ae_key:
                self._baseline_face_b64 = b64
            meta["face_cache"] = "generated"
            meta["face_day"] = day

            self._last_face_day = day
            self._last_face_ae_key = current_ae_key
            self._last_face_b64 = b64
            return {"face_b64": b64}

        except Exception as exc:
            logger.warning("Face generation failed (day %d): %s", day, exc)
            meta["face_error"] = str(exc)
            fallback = self._last_face_b64 or self._baseline_face_b64
            return {"face_b64": fallback}

    # ------------------------------------------------------------------
    # Voice generation
    # ------------------------------------------------------------------

    def _generate_voice(
        self,
        text: str,
        active_aes: list[dict[str, Any]],
        meta: dict,
        mood_snapshot: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        from .voice_generator import generate_patient_voice

        if not text or len(text.strip()) < 10:
            meta["voice_skip"] = "text_too_short"
            return {"audio_b64": None}

        try:
            result = generate_patient_voice(
                text=text,
                patient_profile=self.profile,
                cough_config=None,
                active_aes=active_aes,
                mood_snapshot=mood_snapshot,
                config=self.config,
            )
            b64 = base64.b64encode(result.audio_bytes).decode("ascii")
            meta["voice_duration_sec"] = round(result.duration_sec, 1)
            meta["voice_cough"] = result.cough_inserted
            return {"audio_b64": b64}

        except Exception as exc:
            logger.warning("Voice generation failed: %s", exc)
            meta["voice_error"] = str(exc)
            return {"audio_b64": None}
