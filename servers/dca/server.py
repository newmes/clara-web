"""Care AI API Server — FastAPI

Endpoints:
    POST /v1/consult
        Input:  SigLIP 1152-dim vector + patient STT text + (optional) audio WAV + drug/patient info
        Output: Nurse AI text response + TTS audio (base64) + cough analysis + medical transcript + session_id

    POST /v1/chat
        Input:  session_id + follow-up message
        Output: Nurse AI text response + TTS audio (base64)

    GET /v1/health
        Health check

Startup:
    uvicorn dca_server.server:app --host 0.0.0.0 --port 8300
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

from dca_server.siglip_classifier import SigLIPClassifier
from dca_server.nurse_engine import NurseEngine
from dca_server.tts_service import TTSService
from dca_server.cough_classifier import CoughClassifier
from dca_server.medasr_service import MedASRService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("care-ai-api")

# ===================================================================
# Config (environment variables with defaults)
# ===================================================================

CARE_AI_DIR = Path(__file__).resolve().parent

GPU_ID = int(os.getenv("CARE_AI_GPU", "0"))
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://clara-medgemma4b-base:8000/v1")
VLLM_MODEL_ID = os.getenv("VLLM_MODEL_ID", "medgemma-1.5-4b-it")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
SIGLIP_HEAD = os.getenv("SIGLIP_HEAD", str(CARE_AI_DIR / "models" / "siglip_head.pt"))

DEFAULT_RULE_SET = os.getenv(
    "DEFAULT_RULE_SET",
    "/data/rule_set_calibrated_ev302.json",
)

HEAR_GPU = os.getenv("HEAR_GPU", None)  # None = CPU (default, ~400-750ms)
MEDASR_GPU = os.getenv("MEDASR_GPU", None)  # None = CPU (default, Conformer 105M is fast on CPU)
MEDASR_MODEL_PATH = os.getenv("MEDASR_MODEL_PATH", "google/medasr")
SEGMENTATION_DIR = os.getenv(
    "SEGMENTATION_DIR",
    str(CARE_AI_DIR / "models" / "segmentation"),
)
COUGH_MODEL_DIR_3C = os.getenv(
    "COUGH_MODEL_DIR_3C",
    str(CARE_AI_DIR / "models" / "hear_3c"),
)
COUGH_MODEL_DIR_2C = os.getenv(
    "COUGH_MODEL_DIR_2C",
    str(CARE_AI_DIR / "models" / "hear_2c"),
)

# ===================================================================
# Request / Response Schemas
# ===================================================================

class ConsultRequest(BaseModel):
    siglip_vector: Optional[list[float]] = Field(
        None,
        description="1152-dim SigLIP embedding (required if image_b64 not provided)",
        min_length=1152,
        max_length=1152,
    )
    image_b64: Optional[str] = Field(
        None,
        description="Base64-encoded image (PNG/JPEG). Server encodes via MedSigLIP-448. Use this OR siglip_vector.",
    )
    patient_text: str = Field(
        ...,
        description="STT-transcribed patient speech",
    )
    drug_name: Optional[str] = Field(
        None,
        description="Drug name (e.g., 'Padcev + Pembrolizumab'). Falls back to default if not provided.",
    )
    indication: Optional[str] = Field(
        None,
        description="Indication (e.g., 'metastatic urothelial carcinoma'). Falls back to default.",
    )
    audio_b64: Optional[str] = Field(
        None,
        description="Base64-encoded WAV audio from the patient (for cough detection).",
    )
    skip_tts: bool = Field(
        False,
        description="If true, skip TTS and return text only (faster).",
    )
    api_key: str | None = Field(
        None,
        description="Gemini API key for TTS. Falls back to server env var if not provided.",
    )


class VisualFinding(BaseModel):
    ae_term: str | None
    estimated_grade: int | None
    confidence: float
    description: str


class ConsultResponse(BaseModel):
    session_id: str = Field(
        ...,
        description="Session ID for follow-up chat via /v1/chat",
    )
    nurse_text: str = Field(
        ...,
        description="Natural language nurse response (ready for TTS or display)",
    )
    nurse_structured: dict = Field(
        ...,
        description="Structured JSON nurse response (acknowledgment, questions, concerns)",
    )
    visual_assessment: dict = Field(
        ...,
        description="SigLIP visual analysis results",
    )
    audio_assessment: dict | None = Field(
        None,
        description="Cough detection results from HeAR pipeline (null if no audio provided or classifier unavailable)",
    )
    medical_transcript: str | None = Field(
        None,
        description="Medical ASR transcript from MedASR (null if no audio provided or MedASR unavailable)",
    )
    audio_base64: str | None = Field(
        None,
        description="Base64-encoded audio of the nurse response (null if TTS unavailable or skipped)",
    )
    latency_ms: dict = Field(
        ...,
        description="Latency breakdown in milliseconds",
    )


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Session ID from /v1/consult response")
    message: str = Field(..., description="Patient follow-up message")
    skip_tts: bool = Field(False, description="If true, skip TTS and return text only")
    api_key: str | None = Field(None, description="Gemini API key for TTS")


class ChatResponse(BaseModel):
    nurse_text: str = Field(..., description="Natural language nurse response")
    nurse_structured: dict = Field(..., description="Structured nurse response")
    audio_base64: str | None = Field(None, description="Base64-encoded TTS audio (null if skipped)")
    latency_ms: dict = Field(..., description="Latency breakdown in milliseconds")


class ClassifyRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded image (PNG/JPEG)")

class ClassifyResponse(BaseModel):
    visual_assessment: dict = Field(..., description="SigLIP visual analysis results")
    latency_ms: int = Field(..., description="Latency in milliseconds")

class CoughRequest(BaseModel):
    audio_b64: str = Field(..., description="Base64-encoded WAV audio")

class CoughResponse(BaseModel):
    audio_assessment: dict = Field(..., description="Cough detection results")
    latency_ms: int = Field(..., description="Latency in milliseconds")

class TranscribeRequest(BaseModel):
    audio_b64: str = Field(..., description="Base64-encoded WAV audio")

class TranscribeResponse(BaseModel):
    medical_transcript: str = Field(..., description="Medical ASR transcript")
    latency_ms: int = Field(..., description="Latency in milliseconds")

class NurseRequest(BaseModel):
    patient_text: str = Field(..., description="STT-transcribed patient speech")
    visual_assessment: dict = Field(..., description="SigLIP visual analysis results")
    audio_assessment: dict | None = Field(None, description="Cough detection results")
    medical_transcript: str | None = Field(None, description="Medical ASR transcript")
    drug_name: str | None = Field(None, description="Drug name")
    indication: str | None = Field(None, description="Indication")
    skip_tts: bool = Field(False, description="If true, skip TTS")
    api_key: str | None = Field(None, description="Gemini API key for TTS")

class NurseResponse(BaseModel):
    session_id: str = Field(..., description="Session ID for follow-up chat")
    nurse_text: str = Field(..., description="Natural language nurse response")
    nurse_structured: dict = Field(..., description="Structured JSON nurse response")
    audio_base64: str | None = Field(None, description="Base64-encoded TTS audio")
    latency_ms: dict = Field(..., description="Latency breakdown in milliseconds")


# ===================================================================
# App
# ===================================================================

app = FastAPI(
    title="Care AI — Nurse Agent API",
    version="0.2.0",
    description="MedGemma Nurse Agent with MedSigLIP visual assessment + HeAR cough detection + MedASR + Google TTS",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session store (in-memory, TTL 30 min) ──
SESSION_TTL_SEC = 30 * 60
sessions: dict[str, dict] = {}


def _cleanup_sessions() -> None:
    """Remove expired sessions."""
    now = time.time()
    expired = [sid for sid, s in sessions.items() if now - s["last_active"] > SESSION_TTL_SEC]
    for sid in expired:
        del sessions[sid]


_validated_keys: dict[str, float] = {}  # key -> timestamp
_KEY_CACHE_TTL = 3600  # 1 hour

def _validate_gemini_key(api_key: str) -> bool:
    """Validate a Gemini API key by making a lightweight models.list call."""
    now = time.time()
    if api_key in _validated_keys and (now - _validated_keys[api_key]) < _KEY_CACHE_TTL:
        return True
    try:
        import google.genai as genai
        client = genai.Client(api_key=api_key)
        # Lightweight call to validate the key
        list(client.models.list(config={"page_size": 1}))
        _validated_keys[api_key] = now
        return True
    except Exception as exc:
        log.warning("Gemini API key validation failed: %s", exc)
        _validated_keys.pop(api_key, None)
        return False

def _require_api_key(req_api_key: str | None) -> None:
    """Raise HTTPException if api_key is missing or invalid."""
    if not req_api_key:
        raise HTTPException(status_code=403, detail="Gemini API key required. Provide api_key in request body.")
    if not _validate_gemini_key(req_api_key):
        raise HTTPException(status_code=403, detail="Invalid Gemini API key. Please check your key and try again.")


classifier: SigLIPClassifier | None = None
cough_clf: CoughClassifier | None = None
medasr: MedASRService | None = None
nurse: NurseEngine | None = None
tts: TTSService | None = None


@app.on_event("startup")
async def startup():
    global classifier, cough_clf, medasr, nurse, tts

    log.info("Loading SigLIP classification head from %s", SIGLIP_HEAD)
    classifier = SigLIPClassifier(SIGLIP_HEAD, device="cpu")
    log.info("SigLIP head loaded.")

    try:
        siglip_encoder_model = os.getenv("SIGLIP_ENCODER_MODEL", "google/medsiglip-448")
        siglip_encoder_device = os.getenv("SIGLIP_ENCODER_DEVICE", "cpu")
        classifier.load_encoder(siglip_encoder_model, siglip_encoder_device)
    except Exception as exc:
        log.warning("SigLIP encoder failed to load — image mode disabled: %s", exc)

    try:
        log.info("Loading CoughClassifier (3c=%s, 2c=%s, gpu=%s)", COUGH_MODEL_DIR_3C, COUGH_MODEL_DIR_2C, HEAR_GPU)
        cough_clf = CoughClassifier(
            segmentation_dir=SEGMENTATION_DIR,
            model_dir_3c=COUGH_MODEL_DIR_3C,
            model_dir_2c=COUGH_MODEL_DIR_2C,
            hear_gpu=HEAR_GPU,
        )
        log.info("CoughClassifier ready.")
    except Exception as exc:
        log.warning("CoughClassifier failed to load — audio analysis disabled: %s", exc)
        cough_clf = None

    try:
        medasr_device = f"cuda:{MEDASR_GPU}" if MEDASR_GPU is not None else "cpu"
        log.info("Loading MedASR (model=%s, device=%s)", MEDASR_MODEL_PATH, medasr_device)
        medasr = MedASRService(model_path=MEDASR_MODEL_PATH, device=medasr_device)
        log.info("MedASR ready.")
    except Exception as exc:
        log.warning("MedASR failed to load — medical transcription disabled: %s", exc)
        medasr = None

    log.info("Initializing NurseEngine via vLLM: %s (model=%s)", VLLM_BASE_URL, VLLM_MODEL_ID)
    nurse = NurseEngine(
        vllm_base_url=VLLM_BASE_URL,
        vllm_model_id=VLLM_MODEL_ID,
        vllm_api_key=VLLM_API_KEY,
    )

    if Path(DEFAULT_RULE_SET).exists():
        ctx = nurse.load_drug_context(DEFAULT_RULE_SET)
        log.info("Default drug context loaded: %s", ctx["drug_name"])

    rule_sets_dir = Path("/data") / "new_drugs"
    if rule_sets_dir.exists():
        for rs_path in sorted(rule_sets_dir.glob("*/base.json")):
            try:
                ctx = nurse.load_drug_context(rs_path)
                log.info("  Loaded drug: %s", ctx["drug_name"])
            except Exception as e:
                log.warning("  Failed to load %s: %s", rs_path, e)

    log.info("MedGemma nurse ready.")

    log.info("Initializing TTS service...")
    tts = TTSService()
    if tts.available:
        log.info("TTS ready (Google Cloud).")
    else:
        log.warning("TTS unavailable — will return text-only responses.")

    log.info("=== Care AI API ready ===")


@app.get("/v1/health")
async def health():
    return {
        "status": "ok",
        "classifier_loaded": classifier is not None,
        "encoder_loaded": classifier.encoder_loaded if classifier else False,
        "cough_classifier_loaded": cough_clf is not None,
        "medasr_loaded": medasr is not None,
        "nurse_loaded": nurse is not None,
        "tts_available": tts.available if tts else False,
        "loaded_drugs": list(nurse._drug_profiles.keys()) if nurse else [],
    }


@app.post("/v1/consult", response_model=ConsultResponse)
async def consult(req: ConsultRequest):
    # Gemini key is only needed for TTS; vLLM-based generation works without it
    if not classifier or not nurse:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    if not req.siglip_vector and not req.image_b64:
        raise HTTPException(status_code=422, detail="Either siglip_vector or image_b64 is required")

    log.info("=== /v1/consult ===")
    log.info("patient_text: %s", req.patient_text[:200])
    log.info("drug_name: %s | image: %s | audio: %s",
             req.drug_name, bool(req.image_b64), bool(req.audio_b64))

    timings = {}

    t0 = time.time()
    if req.image_b64:
        if not classifier.encoder_loaded:
            raise HTTPException(status_code=503, detail="SigLIP encoder not loaded — image mode unavailable")
        visual_assessment = classifier.build_visual_assessment_from_image(req.image_b64)
    else:
        visual_assessment = classifier.build_visual_assessment(req.siglip_vector)
    timings["siglip_ms"] = round((time.time() - t0) * 1000)

    # Cough analysis (optional — only if audio provided and classifier loaded)
    audio_assessment = None
    if req.audio_b64 and cough_clf:
        try:
            t0 = time.time()
            audio_assessment = cough_clf.build_audio_assessment(req.audio_b64)
            timings["cough_ms"] = round((time.time() - t0) * 1000)
        except Exception as exc:
            log.warning("Cough analysis failed: %s", exc)
            audio_assessment = None

    # MedASR transcription (optional — only if audio provided and MedASR loaded)
    medical_transcript = None
    if req.audio_b64 and medasr:
        try:
            t0 = time.time()
            medical_transcript = medasr.transcribe(req.audio_b64)
            timings["medasr_ms"] = round((time.time() - t0) * 1000)
        except Exception as exc:
            log.warning("MedASR transcription failed: %s", exc)
            medical_transcript = None

    t0 = time.time()
    nurse_response = nurse.generate_response(
        patient_text=req.patient_text,
        visual_assessment=visual_assessment,
        drug_name=req.drug_name,
        indication=req.indication,
        audio_assessment=audio_assessment,
        medical_transcript=medical_transcript,
    )
    timings["nurse_ms"] = round((time.time() - t0) * 1000)

    speech_text = nurse.response_to_speech_text(nurse_response)

    tts_audio_b64 = None
    if not req.skip_tts and tts and (tts.available or req.api_key):
        t0 = time.time()
        tts_audio_b64 = tts.synthesize_base64(speech_text, api_key=req.api_key)
        timings["tts_ms"] = round((time.time() - t0) * 1000)

    timings["total_ms"] = sum(timings.values())

    # ── Create session for follow-up chat ──
    session_id = uuid.uuid4().hex
    drug_name = req.drug_name
    indication = req.indication
    drug_ae_profile = None
    if drug_name and drug_name in nurse._drug_profiles:
        ctx = nurse._drug_profiles[drug_name]
        drug_name = ctx["drug_name"]
        indication = indication or ctx["indication"]
        drug_ae_profile = ctx["ae_profile"]
    elif not drug_name and nurse._drug_profiles:
        ctx = next(iter(nurse._drug_profiles.values()))
        drug_name = ctx["drug_name"]
        indication = indication or ctx["indication"]
        drug_ae_profile = ctx["ae_profile"]
    else:
        drug_name = drug_name or "Unknown"
        indication = indication or ""
        drug_ae_profile = []

    system_prompt = nurse.build_system_prompt(
        drug_name, indication, visual_assessment, drug_ae_profile,
        audio_assessment=audio_assessment,
        medical_transcript=medical_transcript,
    )
    user_prompt = nurse.build_user_prompt(
        req.patient_text,
        has_audio=audio_assessment is not None,
        medical_transcript=medical_transcript,
    )

    now = time.time()
    sessions[session_id] = {
        "system_prompt": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(nurse_response, ensure_ascii=False)},
        ],
        "created_at": now,
        "last_active": now,
        "drug_name": drug_name,
        "api_key": req.api_key,
    }

    log.info("=== consult response ===")
    log.info("visual: %s", [f.get("ae_term") for f in visual_assessment.get("findings", [])][:5])
    log.info("audio: cough=%s type=%s",
             audio_assessment.get("cough_detected") if audio_assessment else None,
             audio_assessment.get("majority_type") if audio_assessment else None)
    log.info("medasr: %s", (medical_transcript or "")[:100])
    log.info("nurse_text (%d chars): %s", len(speech_text), speech_text[:200])
    log.info("latency: %s", timings)

    return ConsultResponse(
        session_id=session_id,
        nurse_text=speech_text,
        nurse_structured=nurse_response,
        visual_assessment=visual_assessment,
        audio_assessment=audio_assessment,
        medical_transcript=medical_transcript,
        audio_base64=tts_audio_b64,
        latency_ms=timings,
    )


@app.post("/v1/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    if not classifier:
        raise HTTPException(status_code=503, detail="SigLIP classifier not loaded")
    if not classifier.encoder_loaded:
        raise HTTPException(status_code=503, detail="SigLIP encoder not loaded")

    log.info("=== /v1/classify ===")
    t0 = time.time()
    visual_assessment = classifier.build_visual_assessment_from_image(req.image_b64)
    latency = round((time.time() - t0) * 1000)
    log.info("classify done: %d ms", latency)

    return ClassifyResponse(visual_assessment=visual_assessment, latency_ms=latency)


@app.post("/v1/cough", response_model=CoughResponse)
async def cough(req: CoughRequest):
    if not cough_clf:
        raise HTTPException(status_code=503, detail="CoughClassifier not loaded")

    log.info("=== /v1/cough ===")
    t0 = time.time()
    audio_assessment = cough_clf.build_audio_assessment(req.audio_b64)
    latency = round((time.time() - t0) * 1000)
    log.info("cough done: %d ms", latency)

    return CoughResponse(audio_assessment=audio_assessment, latency_ms=latency)


@app.post("/v1/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest):
    if not medasr:
        raise HTTPException(status_code=503, detail="MedASR not loaded")

    log.info("=== /v1/transcribe ===")
    t0 = time.time()
    medical_transcript = medasr.transcribe(req.audio_b64)
    latency = round((time.time() - t0) * 1000)
    log.info("transcribe done: %d ms, %d chars", latency, len(medical_transcript or ""))

    return TranscribeResponse(medical_transcript=medical_transcript, latency_ms=latency)


@app.post("/v1/nurse", response_model=NurseResponse)
async def nurse_endpoint(req: NurseRequest):
    # Gemini key is only needed for TTS; vLLM nurse generation works without it
    if not nurse:
        raise HTTPException(status_code=503, detail="NurseEngine not loaded")

    log.info("=== /v1/nurse ===")
    timings = {}

    t0 = time.time()
    nurse_response = nurse.generate_response(
        patient_text=req.patient_text,
        visual_assessment=req.visual_assessment,
        drug_name=req.drug_name,
        indication=req.indication,
        audio_assessment=req.audio_assessment,
        medical_transcript=req.medical_transcript,
    )
    timings["nurse_ms"] = round((time.time() - t0) * 1000)

    speech_text = nurse.response_to_speech_text(nurse_response)

    tts_audio_b64 = None
    if not req.skip_tts and tts and (tts.available or req.api_key):
        t0 = time.time()
        tts_audio_b64 = tts.synthesize_base64(speech_text, api_key=req.api_key)
        timings["tts_ms"] = round((time.time() - t0) * 1000)

    # Create session for follow-up chat
    session_id = uuid.uuid4().hex
    drug_name = req.drug_name
    indication = req.indication
    drug_ae_profile = None
    if drug_name and drug_name in nurse._drug_profiles:
        ctx = nurse._drug_profiles[drug_name]
        drug_name = ctx["drug_name"]
        indication = indication or ctx["indication"]
        drug_ae_profile = ctx["ae_profile"]
    elif not drug_name and nurse._drug_profiles:
        ctx = next(iter(nurse._drug_profiles.values()))
        drug_name = ctx["drug_name"]
        indication = indication or ctx["indication"]
        drug_ae_profile = ctx["ae_profile"]
    else:
        drug_name = drug_name or "Unknown"
        indication = indication or ""
        drug_ae_profile = []

    system_prompt = nurse.build_system_prompt(
        drug_name, indication, req.visual_assessment, drug_ae_profile,
        audio_assessment=req.audio_assessment,
        medical_transcript=req.medical_transcript,
    )
    user_prompt = nurse.build_user_prompt(
        req.patient_text,
        has_audio=req.audio_assessment is not None,
        medical_transcript=req.medical_transcript,
    )

    now = time.time()
    sessions[session_id] = {
        "system_prompt": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(nurse_response, ensure_ascii=False)},
        ],
        "created_at": now,
        "last_active": now,
        "drug_name": drug_name,
        "api_key": req.api_key,
    }

    timings["total_ms"] = sum(timings.values())

    log.info("nurse_text (%d chars): %s", len(speech_text), speech_text[:200])
    log.info("latency: %s", timings)

    return NurseResponse(
        session_id=session_id,
        nurse_text=speech_text,
        nurse_structured=nurse_response,
        audio_base64=tts_audio_b64,
        latency_ms=timings,
    )


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Follow-up chat within an existing consult session."""
    # Gemini key is only needed for TTS; vLLM-based generation works without it
    if not nurse:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    _cleanup_sessions()

    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    timings = {}

    session["messages"].append({"role": "user", "content": req.message})
    session["last_active"] = time.time()

    t0 = time.time()
    nurse_response = nurse.generate_chat_response(
        system_prompt=session["system_prompt"],
        messages=session["messages"],
    )
    timings["nurse_ms"] = round((time.time() - t0) * 1000)

    session["messages"].append(
        {"role": "assistant", "content": json.dumps(nurse_response, ensure_ascii=False)}
    )

    speech_text = nurse.chat_response_to_speech_text(nurse_response)

    # Use api_key from request, or fall back to session-stored key
    chat_api_key = req.api_key or session.get("api_key")

    tts_audio_b64 = None
    if not req.skip_tts and tts and (tts.available or chat_api_key):
        t0 = time.time()
        tts_audio_b64 = tts.synthesize_base64(speech_text, api_key=chat_api_key)
        timings["tts_ms"] = round((time.time() - t0) * 1000)

    timings["total_ms"] = sum(timings.values())

    return ChatResponse(
        nurse_text=speech_text,
        nurse_structured=nurse_response,
        audio_base64=tts_audio_b64,
        latency_ms=timings,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "dca_server.server:app",
        host="0.0.0.0",
        port=int(os.getenv("CARE_AI_PORT", "8300")),
        reload=False,
        workers=1,
    )
