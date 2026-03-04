"""Multimodal API for clinical trial patient simulation.

Public functions::

    # Face
    generate_patient_face(patient_profile, active_aes, day)
    analyze_face(image_bytes, method)
    train_classifier(image_dir, output_path, epochs)

    # Voice
    generate_patient_voice(text, patient_profile, cough_config, active_aes)
    analyze_voice(audio_bytes, method)
    train_cough_classifier(audio_dir, output_path)

    # Helpers (simulation data adapters)
    extract_patient_speech(care_record)
    derive_cough_config(active_aes)

All functions accept raw simulation data formats directly
(patients/*.json, day JSONL AE records, care_record dicts).
"""

# --- Public functions ---
from .face_generator import generate_patient_face
from .face_analyzer import analyze_face, train_classifier
from .voice_generator import generate_patient_voice, extract_patient_speech, derive_cough_config
from .voice_analyzer import analyze_voice, train_cough_classifier

# --- Schemas (simulation-compatible adapters) ---
from .schemas import (
    SimPatientProfile,
    SimAE,
    DetectedAE,
    CoughEvent,
    RespiratoryAssessment,
    FaceGenerationResult,
    FaceAnalysisResult,
    VoiceGenerationResult,
    VoiceAnalysisResult,
)

# --- Config ---
from .config import (
    MultimodalConfig,
    get_config,
    CTCAE_CRITERIA,
    FACE_RENDERABLE_AES,
    AE_COUGH_MAP,
    RESPIRATORY_AES,
    build_ctcae_table_text,
)

__all__ = [
    # Functions
    "generate_patient_face",
    "analyze_face",
    "train_classifier",
    "generate_patient_voice",
    "extract_patient_speech",
    "derive_cough_config",
    "analyze_voice",
    "train_cough_classifier",
    # Simulation adapters
    "SimPatientProfile",
    "SimAE",
    # Result schemas
    "DetectedAE",
    "CoughEvent",
    "RespiratoryAssessment",
    "FaceGenerationResult",
    "FaceAnalysisResult",
    "VoiceGenerationResult",
    "VoiceAnalysisResult",
    # Config
    "MultimodalConfig",
    "get_config",
    "CTCAE_CRITERIA",
    "FACE_RENDERABLE_AES",
    "AE_COUGH_MAP",
    "RESPIRATORY_AES",
    "build_ctcae_table_text",
]
