"""Gemini 2.5 Flash Preview TTS wrapper.

Converts nurse text responses into audio bytes (WAV, 24 kHz)
using the Gemini TTS model via google-genai SDK.
Falls back gracefully if TTS is unavailable.
"""

from __future__ import annotations

import base64
import logging
import os
import struct
import wave
import io

log = logging.getLogger("care-ai-api.tts")

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

TTS_MODEL = "gemini-2.5-flash-preview-tts"


class TTSService:
    """Gemini 2.5 Flash Preview TTS wrapper."""

    def __init__(
        self,
        voice_name: str = "Kore",
        speaking_rate: float = 0.95,
    ):
        self.voice_name = voice_name
        self.speaking_rate = speaking_rate
        self._client = None

        if not HAS_GENAI:
            log.warning("google-genai not installed — TTS unavailable.")
            return

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            log.warning("GOOGLE_API_KEY not set — TTS unavailable.")
            return

        try:
            self._client = genai.Client(api_key=api_key)
            # Quick validation: just check the client was created
            log.info("Gemini TTS client initialized (model=%s, voice=%s).", TTS_MODEL, voice_name)
        except Exception as exc:
            log.warning("Gemini TTS client failed to initialize: %s", exc)
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    @staticmethod
    def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
        """Wrap raw PCM data in a WAV container."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return buf.getvalue()

    def _get_client(self, api_key: str | None = None):
        """Return a genai client: temporary one if api_key given, else the default."""
        if api_key and HAS_GENAI:
            return genai.Client(api_key=api_key)
        return self._client

    def synthesize(self, text: str, *, api_key: str | None = None) -> bytes | None:
        """Convert text to WAV audio bytes (24 kHz). Returns None if TTS is unavailable."""
        client = self._get_client(api_key)
        if not client or not text.strip():
            return None

        try:
            response = client.models.generate_content(
                model=TTS_MODEL,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=self.voice_name,
                            )
                        )
                    ),
                ),
            )

            audio_part = response.candidates[0].content.parts[0].inline_data
            pcm_data = audio_part.data
            wav_bytes = self._pcm_to_wav(pcm_data)
            return wav_bytes

        except Exception as exc:
            log.warning("Gemini TTS synthesis failed: %s", exc)
            return None

    def synthesize_base64(self, text: str, *, api_key: str | None = None) -> str | None:
        """Convert text to base64-encoded WAV for JSON transport."""
        audio = self.synthesize(text, api_key=api_key)
        if audio:
            return base64.b64encode(audio).decode("utf-8")
        return None
