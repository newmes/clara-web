"""Patient voice generation with real cough-clip insertion.

Public API
----------
- ``generate_patient_voice(text, patient_profile, cough_config, *, config)``
  → :class:`~schemas.VoiceGenerationResult`

Audio utilities are adapted from ``voice/cough_mix_test.py``.
TTS call patterns are adapted from ``voice/tts_test.py``.
"""

from __future__ import annotations

import array
import logging
import random
import re
import wave
from io import BytesIO
from pathlib import Path
from typing import Any

from .config import MultimodalConfig, COUGH_SEVERITY, AE_COUGH_MAP, RESPIRATORY_AES, get_config
from .schemas import VoiceGenerationResult, SimPatientProfile, SimAE

logger = logging.getLogger(__name__)


# =====================================================================
# Speech style per cough severity (TTS acting direction + fillers)
# =====================================================================

SPEECH_STYLE: dict[str, dict] = {
    "none": {
        "direction": "",
        "interjections": [],
        "filler_prob": 0.0,
        "fragment": False,
    },
    "occasional": {
        "direction": (
            "Say the following in a slightly tired, uneasy voice. "
            "You are a patient feeling mildly unwell, speaking a bit slower than normal."
        ),
        "interjections": ["um", "uh"],
        "filler_prob": 0.3,
        "fragment": False,
    },
    "frequent": {
        "direction": (
            "Say the following in a strained, breathy voice with noticeable fatigue. "
            "You are a patient clearly unwell, having difficulty breathing. "
            "Speak slowly with audible effort and discomfort."
        ),
        "interjections": ["uh", "*sigh*", "sorry"],
        "filler_prob": 0.5,
        "fragment": True,
    },
    "severe": {
        "direction": (
            "Say the following in a very weak, exhausted, labored voice. "
            "You are seriously ill and struggling to get each word out. "
            "Every phrase requires enormous effort. "
            "Speak very slowly, pausing to catch your breath, sounding pained."
        ),
        "interjections": ["*heavy breath*", "*sigh*", "excuse me", "sorry"],
        "filler_prob": 0.7,
        "fragment": True,
    },
}


# =====================================================================
# Audio utilities (from voice/cough_mix_test.py)
# =====================================================================

def read_wav(path: Path) -> tuple[bytes, int, int, int]:
    """Read WAV → (pcm_data, sample_rate, channels, sample_width)."""
    with wave.open(str(path), "rb") as w:
        return (
            w.readframes(w.getnframes()),
            w.getframerate(),
            w.getnchannels(),
            w.getsampwidth(),
        )


def save_wav(path: Path, pcm: bytes, rate: int = 24000, channels: int = 1, sw: int = 2):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sw)
        w.setframerate(rate)
        w.writeframes(pcm)


def resample_simple(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Simple nearest-neighbor resample (mono, 16-bit)."""
    if src_rate == dst_rate:
        return pcm
    samples = array.array("h")
    samples.frombytes(pcm)
    ratio = src_rate / dst_rate
    new_len = int(len(samples) / ratio)
    resampled = array.array("h", [0] * new_len)
    for i in range(new_len):
        src_idx = min(int(i * ratio), len(samples) - 1)
        resampled[i] = samples[src_idx]
    return resampled.tobytes()


def to_mono(pcm: bytes, channels: int) -> bytes:
    """Convert multi-channel to mono by taking first channel."""
    if channels == 1:
        return pcm
    samples = array.array("h")
    samples.frombytes(pcm)
    mono = array.array("h", [samples[i] for i in range(0, len(samples), channels)])
    return mono.tobytes()


def normalize_audio(pcm: bytes, src_rate: int, src_channels: int, target_rate: int = 24000) -> bytes:
    """Convert any WAV PCM to mono target_rate 16-bit."""
    data = to_mono(pcm, src_channels)
    data = resample_simple(data, src_rate, target_rate)
    return data


def mix_audio(base: bytes, overlay: bytes, offset_samples: int, overlay_volume: float = 1.0) -> bytes:
    """Mix overlay audio into base audio at sample offset."""
    base_arr = array.array("h")
    base_arr.frombytes(base)
    over_arr = array.array("h")
    over_arr.frombytes(overlay)

    needed = offset_samples + len(over_arr)
    if needed > len(base_arr):
        base_arr.extend(array.array("h", [0] * (needed - len(base_arr))))

    for i, s in enumerate(over_arr):
        idx = offset_samples + i
        mixed = base_arr[idx] + int(s * overlay_volume)
        base_arr[idx] = max(-32768, min(32767, mixed))

    return base_arr.tobytes()


def generate_silence(duration_sec: float, rate: int = 24000) -> bytes:
    n = int(duration_sec * rate)
    return b"\x00\x00" * n


def concatenate_audio(*parts: bytes) -> bytes:
    return b"".join(parts)


def get_random_cough(cough_clips_dir: Path, cough_type: str) -> Path:
    d = cough_clips_dir / cough_type
    clips = sorted(d.glob("*.wav"))
    if not clips:
        raise FileNotFoundError(f"No {cough_type} cough clips in {d}")
    return random.choice(clips)


def load_cough_clip(cough_clips_dir: Path, cough_type: str, target_rate: int = 24000) -> bytes:
    path = get_random_cough(cough_clips_dir, cough_type)
    pcm, rate, ch, sw = read_wav(path)
    return normalize_audio(pcm, rate, ch, target_rate)


def trim_cough(pcm: bytes, max_sec: float = 3.0, rate: int = 24000) -> bytes:
    """Trim cough clip to max duration, picking the loudest segment."""
    samples = array.array("h")
    samples.frombytes(pcm)
    max_samples = int(max_sec * rate)

    if len(samples) <= max_samples:
        return pcm

    best_start = 0
    best_energy = 0
    window = max_samples
    step = rate // 4
    for start in range(0, len(samples) - window, step):
        energy = sum(abs(s) for s in samples[start : start + window])
        if energy > best_energy:
            best_energy = energy
            best_start = start

    trimmed = samples[best_start : best_start + window]
    return trimmed.tobytes()


def pcm_to_wav_bytes(pcm: bytes, rate: int = 24000, channels: int = 1, sw: int = 2) -> bytes:
    """Wrap raw PCM into a WAV byte buffer."""
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sw)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


# =====================================================================
# Sick-speech text transforms
# =====================================================================

def _apply_speech_style(text: str, frequency: str) -> str:
    """Prepend a TTS acting direction so the voice sounds sick."""
    style = SPEECH_STYLE.get(frequency, SPEECH_STYLE["none"])
    direction = style["direction"]
    if not direction:
        return text
    return f"{direction}\n\n{text}"


# Baseline direction applied even without cough / active AEs.
# A cancer patient on chemo should never sound bright or cheerful.
_BASELINE_DIRECTIONS: dict[str, str] = {
    "stoic": (
        "Speak in a calm, subdued, low-energy voice. "
        "You are a reserved elderly cancer patient, tired from chemotherapy. "
        "You speak slowly and quietly, not showing much emotion. "
        "Your tone is flat and restrained."
    ),
    "minimizer": (
        "Speak in a gentle, somewhat tired voice. "
        "You are a cancer patient who downplays symptoms. "
        "You try to sound okay but your voice betrays fatigue. "
        "Your tone is soft and slightly strained."
    ),
    "anxious": (
        "Speak in a slightly nervous, worried voice. "
        "You are a cancer patient feeling uneasy about your treatment. "
        "Your voice has a tremor of anxiety and uncertainty. "
        "You speak somewhat quickly and hesitantly."
    ),
    "default": (
        "Speak in a tired, low-energy voice. "
        "You are a cancer patient undergoing chemotherapy. "
        "You are not feeling well and your voice reflects weariness. "
        "Speak slowly, with a somber and subdued tone."
    ),
}


def _apply_baseline_speech_direction(
    text: str,
    patient_profile,
    frequency_label: str,
    mood_snapshot: dict[str, float] | None = None,
) -> str:
    """Always prepend an appropriate TTS direction for a cancer patient.

    If the AE-based frequency_label already has a direction (occasional+),
    use that. Otherwise apply a baseline 'sick patient' direction based
    on persona type, enriched with mood context when available.
    """
    if frequency_label in ("occasional", "frequent", "severe"):
        return _apply_speech_style(text, frequency_label)

    persona_type = "default"
    if isinstance(patient_profile, SimPatientProfile):
        persona_type = (patient_profile.persona_type or "default").lower()
    elif isinstance(patient_profile, dict):
        persona_type = (
            patient_profile.get("persona", {}).get("type", "default")
            if "persona" in patient_profile
            else "default"
        ).lower()

    for key in _BASELINE_DIRECTIONS:
        if key in persona_type:
            direction = _BASELINE_DIRECTIONS[key]
            break
    else:
        direction = _BASELINE_DIRECTIONS["default"]

    if mood_snapshot:
        mood_hints = []
        energy = mood_snapshot.get("energy", 0.5)
        anxiety = mood_snapshot.get("anxiety", 0.3)
        depression = mood_snapshot.get("depression", 0.3)
        if energy < 0.3:
            mood_hints.append("Very low energy — speak slowly and weakly.")
        elif energy < 0.5:
            mood_hints.append("Low energy — slightly sluggish speech.")
        if anxiety > 0.7:
            mood_hints.append("High anxiety — slightly trembling, faster pace.")
        if depression > 0.7:
            mood_hints.append("Depressed — flat affect, monotone delivery.")
        if mood_hints:
            direction = direction + " " + " ".join(mood_hints)

    return f"{direction}\n\n{text}"


def _insert_filler(sentence: str, frequency: str) -> str:
    """Probabilistically prepend a hesitation / sigh filler."""
    style = SPEECH_STYLE.get(frequency, SPEECH_STYLE["none"])
    interjections = style.get("interjections", [])
    prob = style.get("filler_prob", 0)
    if interjections and random.random() < prob:
        return f"{random.choice(interjections)}... {sentence}"
    return sentence


def _fragment_for_sick_speech(sentence: str, frequency: str) -> str:
    """Break a long sentence into '...'-separated fragments.

    Example (severe):
        "My symptoms have gotten worse and the rash is spreading"
      → "My symptoms have gotten worse... and the rash is... spreading"
    """
    style = SPEECH_STYLE.get(frequency, SPEECH_STYLE["none"])
    if not style.get("fragment"):
        return sentence

    words = sentence.split()
    if len(words) <= 5:
        return sentence

    chunk_size = 5 if frequency == "severe" else 7
    fragments = []
    for i in range(0, len(words), chunk_size):
        fragment = " ".join(words[i : i + chunk_size])
        fragments.append(fragment)

    return "... ".join(fragments)


# =====================================================================
# TTS helpers (adapted from voice/tts_test.py + cough_mix_test.py)
# =====================================================================

def _create_tts_client(config: MultimodalConfig):
    """Lazy-create a Gemini client for TTS."""
    from google import genai as _genai
    return _genai.Client(api_key=config.gemini_api_key)


def _generate_tts_segment(text: str, voice: str, config: MultimodalConfig) -> bytes:
    """Single-speaker TTS segment → raw PCM bytes (24kHz mono 16-bit)."""
    from google.genai import types

    client = _create_tts_client(config)
    response = client.models.generate_content(
        model=config.tts_model,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice,
                    )
                )
            ),
        ),
    )
    return response.candidates[0].content.parts[0].inline_data.data


# =====================================================================
# Voice selection
# =====================================================================

def _age_bucket(age: int) -> str:
    if age < 40:
        return "young"
    elif age < 65:
        return "middle"
    return "elderly"


def _select_voice(patient_profile, config: MultimodalConfig) -> str:
    if isinstance(patient_profile, SimPatientProfile):
        age = patient_profile.age
        sex = patient_profile.sex.lower()
    elif "emr" in patient_profile:
        p = SimPatientProfile.from_sim_patient(patient_profile)
        age, sex = p.age, p.sex.lower()
    else:
        age = patient_profile.get("age", 50)
        sex = patient_profile.get("sex", "male").lower()
    bucket = _age_bucket(age)
    return config.voice_map.get((bucket, sex), "Puck")


# =====================================================================
# Sentence splitting
# =====================================================================

_SENTENCE_RE = re.compile(r'(?<=[.!?…])\s+')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for segment-by-segment TTS."""
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# =====================================================================
# Public API
# =====================================================================

def extract_patient_speech(care_record: list[dict[str, Any]]) -> str:
    """Extract patient speech text from a simulation care_record.

    Concatenates all patient-turn text from the care_record's conversation
    turns into a single string suitable for TTS.
    """
    parts: list[str] = []
    for record in care_record:
        for turn in record.get("turns", []):
            if turn.get("role") != "patient":
                continue
            content = turn.get("content", {})
            if isinstance(content, str):
                parts.append(content)
                continue
            if g := content.get("greeting"):
                parts.append(g)
            if wb := content.get("general_wellbeing"):
                parts.append(wb)
            for resp in content.get("responses", []):
                if a := resp.get("answer"):
                    parts.append(a)
    return " ".join(parts)


def derive_cough_config(active_aes: list) -> dict[str, str]:
    """Derive cough_config from the simulation's active AE list.

    Checks AE terms against ``AE_COUGH_MAP`` and ``RESPIRATORY_AES``.
    Returns ``{"type": ..., "frequency": ...}`` or ``{"frequency": "none"}``.
    """
    best_freq_rank = 0
    freq_labels = ["none", "occasional", "frequent", "severe"]
    result: dict[str, str] = {"type": "dry", "frequency": "none"}

    for ae in active_aes:
        if isinstance(ae, SimAE):
            term, grade = ae.ae_term, ae.grade
        elif "AETERM" in ae:
            term, grade = ae["AETERM"], int(ae.get("_grade", 1))
        else:
            term = ae.get("ae", ae.get("ae_term", ""))
            grade = int(ae.get("grade", 1))

        mapping = AE_COUGH_MAP.get(term)
        if mapping:
            grade_key = f"frequency_g{min(grade, 3)}"
            freq = mapping.get(grade_key, "occasional")
            rank = freq_labels.index(freq) if freq in freq_labels else 0
            if rank > best_freq_rank:
                best_freq_rank = rank
                result = {"type": mapping.get("type", "dry"), "frequency": freq}
        elif term in RESPIRATORY_AES and grade >= 2:
            if best_freq_rank < 1:
                best_freq_rank = 1
                result = {"type": "dry", "frequency": "occasional"}

    return result


def generate_patient_voice(
    text: str,
    patient_profile,
    cough_config: dict[str, Any] | None = None,
    *,
    active_aes: list | None = None,
    mood_snapshot: dict[str, float] | None = None,
    config: MultimodalConfig | None = None,
) -> VoiceGenerationResult:
    """Generate patient speech audio, optionally with real cough clips inserted.

    Parameters
    ----------
    text:
        The dialogue text the patient should speak.
    patient_profile:
        One of:
        - :class:`SimPatientProfile`
        - Raw ``patients/PT-XXX.json`` dict
        - Flat ``{"age": int, "sex": str}`` dict
    cough_config:
        ``{"type": "dry"|"productive"|"wheezing", "frequency": "none"|"occasional"|"frequent"|"severe"}``.
        If *None*, auto-derived from ``active_aes`` via :func:`derive_cough_config`.
    active_aes:
        Optional list of AEs (CDASH dicts, SimAE, or flat dicts).
    mood_snapshot:
        Optional mood dimensions (anxiety, depression, energy, etc.) from
        the game session, used to enrich TTS speech style direction.
    config:
        Override the default :class:`MultimodalConfig`.
    """
    cfg = config or get_config()
    voice = _select_voice(patient_profile, cfg)
    if cough_config is None and active_aes:
        cough_config = derive_cough_config(active_aes)
    cough_config = cough_config or {}
    frequency_label = cough_config.get("frequency", "none")
    cough_type_label = cough_config.get("type", "dry")
    cough_dir_name = cfg.cough_type_dir.get(cough_type_label, "dry")

    min_coughs, max_coughs = cfg.cough_frequency.get(frequency_label, (0, 0))
    num_coughs = random.randint(min_coughs, max_coughs) if max_coughs > 0 else 0
    cough_inserted = num_coughs > 0

    # Severity에 따른 볼륨 / 클립 길이 조절
    overlay_volume, max_cough_sec = COUGH_SEVERITY.get(frequency_label, (0.8, 2.5))

    # --- Simple TTS (no cough) ---
    if not cough_inserted:
        logger.info("Generating pure TTS (voice=%s, style=%s)", voice, frequency_label)
        styled_text = _apply_baseline_speech_direction(text, patient_profile, frequency_label, mood_snapshot)
        pcm = _generate_tts_segment(styled_text, voice, cfg)
        wav_bytes = pcm_to_wav_bytes(pcm, rate=cfg.sample_rate)
        duration = len(pcm) / 2 / cfg.sample_rate
        return VoiceGenerationResult(
            audio_bytes=wav_bytes,
            duration_sec=duration,
            transcript=text,
            cough_inserted=False,
            metadata={"voice": voice, "speech_style": frequency_label},
        )

    # --- Segment-by-segment TTS with cough clip insertion ---
    sentences = _split_sentences(text)
    if not sentences:
        sentences = [text]

    # Decide where to insert coughs (between sentences)
    num_gaps = max(len(sentences) - 1, 1)
    cough_positions: set[int] = set()
    if num_coughs > 0 and num_gaps > 0:
        available = list(range(num_gaps))
        random.shuffle(available)
        for pos in available[: min(num_coughs, num_gaps)]:
            cough_positions.add(pos)

    logger.info(
        "Generating segmented TTS: %d sentences, %d cough insertions (voice=%s)",
        len(sentences), len(cough_positions), voice,
    )

    result_pcm = b""
    cough_timestamps = []
    for i, sentence in enumerate(sentences):
        # Apply sick-speech transforms then generate TTS
        styled = _fragment_for_sick_speech(sentence, frequency_label)
        styled = _insert_filler(styled, frequency_label)
        styled = _apply_speech_style(styled, frequency_label)
        seg_pcm = _generate_tts_segment(styled, voice, cfg)
        result_pcm = concatenate_audio(result_pcm, seg_pcm)

        # Insert cough between sentences if scheduled
        if i in cough_positions:
            cough_pcm = load_cough_clip(cfg.cough_clips_dir, cough_dir_name, cfg.sample_rate)
            cough_pcm = trim_cough(cough_pcm, max_sec=max_cough_sec, rate=cfg.sample_rate)
            silence = generate_silence(cfg.silence_padding_sec, cfg.sample_rate)
            # mix_audio로 볼륨 조절하여 삽입
            cough_offset = len(result_pcm) // 2 + int(cfg.silence_padding_sec * cfg.sample_rate)
            cough_dur_sec = len(cough_pcm) / 2 / cfg.sample_rate
            cough_start_sec = cough_offset / cfg.sample_rate
            cough_timestamps.append({
                "start_sec": round(cough_start_sec, 3),
                "end_sec": round(cough_start_sec + cough_dur_sec, 3),
                "duration_sec": round(cough_dur_sec, 3),
                "cough_type": cough_type_label,
            })
            padded_cough = concatenate_audio(silence, cough_pcm, silence)
            result_pcm = concatenate_audio(result_pcm, generate_silence(
                cfg.silence_padding_sec + len(cough_pcm) / 2 / cfg.sample_rate + cfg.silence_padding_sec,
                cfg.sample_rate,
            ))
            result_pcm = mix_audio(result_pcm, cough_pcm, cough_offset, overlay_volume=overlay_volume)

    wav_bytes = pcm_to_wav_bytes(result_pcm, rate=cfg.sample_rate)
    duration = len(result_pcm) / 2 / cfg.sample_rate

    return VoiceGenerationResult(
        audio_bytes=wav_bytes,
        duration_sec=duration,
        transcript=text,
        cough_inserted=True,
        metadata={
            "voice": voice,
            "cough_type": cough_type_label,
            "cough_dir": cough_dir_name,
            "num_coughs_inserted": len(cough_positions),
            "cough_timestamps": cough_timestamps,
            "frequency": frequency_label,
            "overlay_volume": overlay_volume,
            "max_cough_sec": max_cough_sec,
            "speech_style": frequency_label,
        },
    )
