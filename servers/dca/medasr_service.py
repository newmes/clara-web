"""MedASR Service — Medical Automatic Speech Recognition.

Uses google/medasr (Conformer 105M) to transcribe medical audio.
Designed to run on CPU by default; GPU configurable via constructor.
"""

from __future__ import annotations

import base64
import io
import logging

import librosa
import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor

log = logging.getLogger("care-ai-api.medasr")


def _patch_lasr_fbank(processor):
    """Patch LasrFeatureExtractor to accept extra args from parent __call__."""
    fe = getattr(processor, "feature_extractor", None)
    if fe is None or type(fe).__name__ != "LasrFeatureExtractor":
        return
    orig = fe._torch_extract_fbank_features

    def _patched(waveform, device="cpu", center=True):
        return orig(waveform, device=device)

    fe._torch_extract_fbank_features = _patched
    log.info("Patched LasrFeatureExtractor._torch_extract_fbank_features for compat.")


class MedASRService:
    """Medical ASR using google/medasr via AutoModelForCTC + AutoProcessor."""

    def __init__(self, model_path: str = "google/medasr", device: str = "cpu"):
        log.info("Loading MedASR model (%s) on %s ...", model_path, device)
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path)
        _patch_lasr_fbank(self.processor)
        self.model = AutoModelForCTC.from_pretrained(model_path).to(device).eval()
        log.info("MedASR model loaded.")

    def transcribe(self, audio_b64: str) -> str:
        """Transcribe base64-encoded WAV audio to text."""
        raw = base64.b64decode(audio_b64)
        audio, _ = librosa.load(io.BytesIO(raw), sr=16000, mono=True)
        audio = audio.astype(np.float32)

        inputs = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        # Clean up CTC artifacts
        text = text.replace("<epsilon>", "").strip()
        return text
