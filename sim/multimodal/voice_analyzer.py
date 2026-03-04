"""Voice/audio analysis for cough detection via HeAR embeddings.

Public API
----------
- ``analyze_voice(audio_bytes, method, *, config)``
  → :class:`~schemas.VoiceAnalysisResult`
- ``train_cough_classifier(audio_dir, output_path, *, config)``
  → ``dict`` of training metrics

Two analysis methods:
  - ``"gemini"`` — Gemini Audio API (STT + cough event detection)
  - ``"hear"``   — google/hear local model (TF/Keras, 512-d embeddings)

Output ``VoiceAnalysisResult.cough_events`` can be fed directly into
the simulation engine's care_record or hospital_record AE pipeline.
"""

from __future__ import annotations

import array
import io
import json
import logging
import re
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .config import MultimodalConfig, get_config
from .schemas import CoughEvent, RespiratoryAssessment, VoiceAnalysisResult

logger = logging.getLogger(__name__)


# =====================================================================
# Gemini analysis prompt (kept for "gemini" method)
# =====================================================================

_AUDIO_ANALYSIS_PROMPT = """\
You are a clinical audio analysis expert. Analyze this audio recording from a
telemedicine video call with a clinical trial patient.

Perform the following tasks:

1. **Transcribe** the patient's speech as accurately as possible.
2. **Detect cough events**: For each distinct cough or coughing fit, report:
   - timestamp_sec: approximate seconds from the start of the audio
   - cough_type: "dry" (non-productive, barking) | "productive" (wet, with mucus) | "wheezing" (with wheeze sounds)
   - severity: "mild" | "moderate" | "severe"
   - confidence: 0.0-1.0
3. **Respiratory assessment**: Provide a brief clinical summary of the patient's
   respiratory status based on the audio (voice quality, breathing patterns,
   cough characteristics, etc.).

## Output Format (JSON)
{
  "transcript": "<full transcription>",
  "cough_events": [
    {
      "timestamp_sec": <float>,
      "cough_type": "<dry|productive|wheezing>",
      "severity": "<mild|moderate|severe>",
      "confidence": <0.0-1.0>
    }
  ],
  "respiratory_assessment": "<clinical summary>"
}
"""


# =====================================================================
# Method 1: Gemini Audio ("gemini")
# =====================================================================

def _analyze_gemini(audio_bytes: bytes, config: MultimodalConfig) -> VoiceAnalysisResult:
    """Gemini Audio analysis with STT + cough detection."""
    from google import genai as _genai
    from google.genai import types

    t0 = time.time()

    client = _genai.Client(api_key=config.gemini_api_key)
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[_AUDIO_ANALYSIS_PROMPT, audio_part],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    latency_ms = (time.time() - t0) * 1000
    raw_text = response.text.strip()

    return _parse_voice_result(raw_text, latency_ms, model_used="gemini")


def _parse_voice_result(raw: str, latency_ms: float, model_used: str = "gemini") -> VoiceAnalysisResult:
    """Parse model JSON output into VoiceAnalysisResult, with regex fallback."""
    parsed = _try_parse_json(raw)

    if parsed is None:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            parsed = _try_parse_json(match.group())

    if parsed is None:
        logger.warning("Failed to parse voice analysis JSON: %s", raw[:200])
        return VoiceAnalysisResult(
            transcript=raw,
            cough_events=[],
            respiratory_assessment=RespiratoryAssessment(
                has_cough=False, has_wheeze=False, has_dyspnea=False,
                overall_severity="normal",
            ),
            model_used=model_used,
            latency_ms=latency_ms,
        )

    cough_events = []
    for item in parsed.get("cough_events", []):
        try:
            cough_events.append(CoughEvent(
                timestamp_sec=float(item.get("timestamp_sec", 0)),
                cough_type=item.get("cough_type", "dry"),
                severity=item.get("severity", "mild"),
                confidence=float(item.get("confidence", 0.5)),
            ))
        except (TypeError, ValueError) as e:
            logger.warning("Skipping malformed cough event: %s (%s)", item, e)

    # Gemini가 반환한 respiratory_assessment를 구조화
    ra_raw = parsed.get("respiratory_assessment", "")
    n_coughs = len(cough_events)
    has_cough = n_coughs > 0
    has_wheeze = any(e.cough_type == "wheezing" for e in cough_events)
    has_dyspnea = any(e.severity == "severe" for e in cough_events) and n_coughs >= 4

    if n_coughs == 0:
        overall_severity = "normal"
    elif n_coughs <= 2:
        overall_severity = "mild"
    elif n_coughs <= 5:
        overall_severity = "moderate"
    else:
        overall_severity = "severe"

    # Gemini가 텍스트로 wheeze/dyspnea를 언급하면 반영
    ra_lower = ra_raw.lower()
    if "wheez" in ra_lower:
        has_wheeze = True
    if "dyspnea" in ra_lower or "shortness of breath" in ra_lower:
        has_dyspnea = True

    assessment = RespiratoryAssessment(
        has_cough=has_cough,
        has_wheeze=has_wheeze,
        has_dyspnea=has_dyspnea,
        overall_severity=overall_severity,
    )

    return VoiceAnalysisResult(
        transcript=parsed.get("transcript", ""),
        cough_events=cough_events,
        respiratory_assessment=assessment,
        model_used=model_used,
        latency_ms=latency_ms,
    )


def _try_parse_json(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return None


# =====================================================================
# Gemini STT (used by HeAR method for transcription)
# =====================================================================

def _gemini_stt(audio_bytes: bytes, config: MultimodalConfig) -> str:
    """Gemini API로 음성→텍스트 변환만 수행."""
    from google import genai as _genai
    from google.genai import types

    client = _genai.Client(api_key=config.gemini_api_key)
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[
            "Transcribe the following audio exactly as spoken. "
            "Return ONLY the transcription text, nothing else.",
            audio_part,
        ],
    )
    return response.text.strip()


# =====================================================================
# Method 2: HeAR local model ("hear")
# =====================================================================

_hear_model = None
_hear_serving = None
_cough_classifier = None


def _load_hear(config: MultimodalConfig):
    """Load HeAR SavedModel (cached at module level).

    Uses huggingface_hub.snapshot_download + tf.saved_model.load to avoid
    the from_pretrained_keras path which triggers a torch CUDA import conflict.
    """
    global _hear_model, _hear_serving

    if _hear_serving is not None:
        return _hear_serving

    import os
    import tensorflow as tf
    from huggingface_hub import snapshot_download

    # HeAR is small (~300M params); pin to a GPU with free memory
    hear_gpu = os.environ.get("HEAR_GPU", "5")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            gpu_idx = int(hear_gpu)
            if gpu_idx < len(gpus):
                tf.config.set_visible_devices(gpus[gpu_idx], "GPU")
                tf.config.experimental.set_memory_growth(gpus[gpu_idx], True)
                logger.info("HeAR pinned to GPU %d", gpu_idx)
        except (ValueError, RuntimeError) as e:
            logger.warning("GPU config failed, using default: %s", e)

    logger.info("Downloading HeAR model: %s", config.hear_model)
    model_dir = snapshot_download(config.hear_model)
    logger.info("Loading HeAR SavedModel from %s", model_dir)
    _hear_model = tf.saved_model.load(model_dir)
    _hear_serving = _hear_model.signatures["serving_default"]
    logger.info("HeAR loaded.")
    return _hear_serving


def _load_cough_classifier(path: Path, config: MultimodalConfig):
    """Load a trained cough classifier on top of HeAR embeddings."""
    global _cough_classifier

    if _cough_classifier is not None:
        return _cough_classifier

    import pickle

    with open(path, "rb") as f:
        _cough_classifier = pickle.load(f)
    logger.info("Cough classifier loaded from %s", path)
    return _cough_classifier


def _wav_bytes_to_pcm_float(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """WAV bytes → (float32 samples in [-1, 1], sample_rate)."""
    buf = io.BytesIO(audio_bytes)
    with wave.open(buf, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())

    samples = array.array("h")
    samples.frombytes(raw)

    # To mono
    if ch > 1:
        samples = array.array("h", [samples[i] for i in range(0, len(samples), ch)])

    pcm = np.array(samples, dtype=np.float32) / 32768.0
    return pcm, sr


def _resample_np(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Simple nearest-neighbor resample for numpy array."""
    if src_rate == dst_rate:
        return pcm
    ratio = src_rate / dst_rate
    new_len = int(len(pcm) / ratio)
    indices = np.clip((np.arange(new_len) * ratio).astype(int), 0, len(pcm) - 1)
    return pcm[indices]


def _chunk_audio(pcm: np.ndarray, clip_samples: int, hop_samples: int) -> list[tuple[float, np.ndarray]]:
    """Chunk audio into fixed-length windows. Returns (start_sec, chunk) pairs."""
    chunks = []
    for start in range(0, len(pcm) - clip_samples + 1, hop_samples):
        chunk = pcm[start : start + clip_samples]
        chunks.append((start, chunk))
    # Handle tail if audio is shorter than one clip
    if not chunks and len(pcm) > 0:
        padded = np.zeros(clip_samples, dtype=np.float32)
        padded[: len(pcm)] = pcm
        chunks.append((0, padded))
    return chunks


def _compute_energy(chunk: np.ndarray) -> float:
    """RMS energy of a chunk."""
    return float(np.sqrt(np.mean(chunk ** 2)))


def _classify_cough_type(energy: float) -> tuple[str, str]:
    """Heuristic cough type/severity from energy (without trained classifier)."""
    if energy > 0.15:
        return "dry", "severe"
    elif energy > 0.08:
        return "dry", "moderate"
    else:
        return "dry", "mild"


def _analyze_hear(
    audio_bytes: bytes,
    config: MultimodalConfig,
    classifier_path: Path | None = None,
) -> VoiceAnalysisResult:
    """HeAR-based cough detection from audio.

    Pipeline:
      1. Resample to 16kHz mono
      2. Chunk into 2s windows (1s hop)
      3. Extract HeAR 512-d embeddings per chunk
      4. Detect cough events via trained classifier or energy heuristic
    """
    import tensorflow as tf

    t0 = time.time()

    serving = _load_hear(config)

    # Prepare audio
    pcm, src_rate = _wav_bytes_to_pcm_float(audio_bytes)
    pcm = _resample_np(pcm, src_rate, config.hear_sample_rate)

    clip_samples = int(config.hear_clip_sec * config.hear_sample_rate)  # 32000
    hop_samples = clip_samples // 2  # 1s hop for overlap

    chunks = _chunk_audio(pcm, clip_samples, hop_samples)
    if not chunks:
        latency_ms = (time.time() - t0) * 1000
        return VoiceAnalysisResult(
            transcript="",
            cough_events=[],
            respiratory_assessment=RespiratoryAssessment(
                has_cough=False, has_wheeze=False, has_dyspnea=False,
                overall_severity="normal",
            ),
            model_used="hear",
            latency_ms=latency_ms,
        )

    # Batch inference
    starts = [s for s, _ in chunks]
    batch = np.stack([c for _, c in chunks], axis=0)  # (N, 32000)
    embeddings = serving(x=tf.constant(batch, dtype=tf.float32))["output_0"].numpy()  # (N, 512)

    # Classify each chunk
    has_classifier = False
    clf = None
    if classifier_path is not None and classifier_path.exists():
        clf = _load_cough_classifier(classifier_path, config)
        has_classifier = True

    cough_events: list[CoughEvent] = []
    energies = []

    for i, (sample_start, chunk) in enumerate(chunks):
        timestamp = sample_start / config.hear_sample_rate
        energy = _compute_energy(chunk)
        energies.append(energy)

        if has_classifier:
            # Trained classifier: expects (1, 512) → class probabilities
            emb = embeddings[i : i + 1]
            pred = clf.predict(emb)[0]
            proba = clf.predict_proba(emb)[0]
            # Classes: 0=silence/speech, 1=cough_dry, 2=cough_productive, 3=cough_wheezing
            class_names = getattr(clf, "classes_", [0, 1, 2, 3])
            if pred != 0:  # not silence/speech
                type_map = {1: "dry", 2: "productive", 3: "wheezing"}
                cough_type = type_map.get(pred, "dry")
                conf = float(proba[list(class_names).index(pred)])
                severity = "severe" if energy > 0.15 else "moderate" if energy > 0.08 else "mild"
                cough_events.append(CoughEvent(
                    timestamp_sec=round(timestamp, 1),
                    cough_type=cough_type,
                    severity=severity,
                    confidence=round(conf, 2),
                ))
        else:
            # Energy heuristic fallback: high-energy bursts → likely cough
            if energy > 0.05:
                cough_type, severity = _classify_cough_type(energy)
                conf = min(1.0, energy / 0.2)
                cough_events.append(CoughEvent(
                    timestamp_sec=round(timestamp, 1),
                    cough_type=cough_type,
                    severity=severity,
                    confidence=round(conf, 2),
                ))

    # Deduplicate overlapping detections (keep highest confidence per 1s window)
    cough_events = _deduplicate_events(cough_events, window_sec=1.5)

    # Build structured respiratory assessment
    n_coughs = len(cough_events)
    has_cough = n_coughs > 0
    has_wheeze = any(e.cough_type == "wheezing" for e in cough_events)
    has_dyspnea = any(e.severity == "severe" for e in cough_events) and n_coughs >= 4

    if n_coughs == 0:
        overall_severity = "normal"
    elif n_coughs <= 2:
        overall_severity = "mild"
    elif n_coughs <= 5:
        overall_severity = "moderate"
    else:
        overall_severity = "severe"

    assessment = RespiratoryAssessment(
        has_cough=has_cough,
        has_wheeze=has_wheeze,
        has_dyspnea=has_dyspnea,
        overall_severity=overall_severity,
    )

    # STT via Gemini
    try:
        transcript = _gemini_stt(audio_bytes, config)
    except Exception as e:
        logger.warning("Gemini STT failed, skipping transcript: %s", e)
        transcript = ""

    latency_ms = (time.time() - t0) * 1000

    return VoiceAnalysisResult(
        transcript=transcript,
        cough_events=cough_events,
        respiratory_assessment=assessment,
        model_used="hear",
        latency_ms=latency_ms,
    )


def _deduplicate_events(events: list[CoughEvent], window_sec: float) -> list[CoughEvent]:
    """Keep only the highest-confidence event per time window."""
    if not events:
        return []
    events = sorted(events, key=lambda e: e.timestamp_sec)
    deduped = [events[0]]
    for ev in events[1:]:
        if ev.timestamp_sec - deduped[-1].timestamp_sec < window_sec:
            if ev.confidence > deduped[-1].confidence:
                deduped[-1] = ev
        else:
            deduped.append(ev)
    return deduped


# =====================================================================
# Public API: train_cough_classifier
# =====================================================================

def train_cough_classifier(
    audio_dir: str | Path,
    output_path: str | Path,
    *,
    config: MultimodalConfig | None = None,
) -> dict[str, Any]:
    """Train a cough classifier on top of HeAR embeddings.

    Expected directory structure::

        audio_dir/
            silence/       ← normal speech or silence clips (.wav)
            cough_dry/     ← dry cough clips
            cough_wet/     ← productive/wet cough clips
            cough_wheeze/  ← wheezing cough clips

    Each clip should be ≤ 2 seconds, 16kHz mono WAV.

    Returns
    -------
    dict with ``{"accuracy": float, "f1": float, "n_samples": int}``
    """
    import tensorflow as tf
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    import pickle

    cfg = config or get_config()
    audio_dir = Path(audio_dir)
    output_path = Path(output_path)

    serving = _load_hear(cfg)
    clip_samples = int(cfg.hear_clip_sec * cfg.hear_sample_rate)

    class_dirs = {
        0: "silence",
        1: "cough_dry",
        2: "cough_wet",
        3: "cough_wheeze",
    }

    all_embeddings = []
    all_labels = []

    for label, dirname in class_dirs.items():
        d = audio_dir / dirname
        if not d.is_dir():
            logger.warning("Missing class dir: %s", d)
            continue
        for wav_path in sorted(d.glob("*.wav")):
            pcm, sr = _wav_bytes_to_pcm_float(wav_path.read_bytes())
            pcm = _resample_np(pcm, sr, cfg.hear_sample_rate)
            # Pad or trim to clip_samples
            if len(pcm) < clip_samples:
                padded = np.zeros(clip_samples, dtype=np.float32)
                padded[: len(pcm)] = pcm
                pcm = padded
            else:
                pcm = pcm[:clip_samples]

            emb = serving(x=tf.constant(pcm[np.newaxis], dtype=tf.float32))["output_0"].numpy()
            all_embeddings.append(emb[0])
            all_labels.append(label)

    if len(all_embeddings) == 0:
        raise ValueError(f"No audio clips found in {audio_dir}")

    X = np.stack(all_embeddings)
    y = np.array(all_labels)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = LogisticRegression(max_iter=1000, multi_class="multinomial", random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(clf, f)

    logger.info("Cough classifier trained: accuracy=%.3f, f1=%.3f, saved to %s", acc, f1, output_path)
    return {
        "accuracy": acc,
        "f1": f1,
        "n_samples": len(all_embeddings),
        "output_path": str(output_path),
    }


# =====================================================================
# Public API: analyze_voice
# =====================================================================

def analyze_voice(
    audio_bytes: bytes,
    method: str = "hear",
    *,
    config: MultimodalConfig | None = None,
    classifier_path: Path | str | None = None,
) -> VoiceAnalysisResult:
    """Analyze a voice audio recording for cough events.

    Parameters
    ----------
    audio_bytes:
        WAV audio bytes.
    method:
        ``"hear"`` (HeAR local model, default) or ``"gemini"`` (Gemini API).
    config:
        Override the default :class:`MultimodalConfig`.
    classifier_path:
        Path to trained cough classifier pickle (optional for ``"hear"``).
        Without it, energy-based heuristic is used.

    Returns
    -------
    VoiceAnalysisResult
    """
    cfg = config or get_config()

    if method == "hear":
        cp = Path(classifier_path) if classifier_path else None
        return _analyze_hear(audio_bytes, cfg, classifier_path=cp)
    elif method == "gemini":
        return _analyze_gemini(audio_bytes, cfg)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'hear' or 'gemini'.")
