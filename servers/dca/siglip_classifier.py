"""SigLIP AE Classification — vector or image inference.

Supports two modes:
  1. Vector mode: mobile app sends pre-computed 1152-dim SigLIP embedding
  2. Image mode: server encodes image via MedSigLIP-448 vision encoder → 1152-dim → head
"""

from __future__ import annotations

import base64
import io
import logging

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

log = logging.getLogger("care-ai-api.siglip")

EMBED_DIM = 1152
SIGLIP_MODEL_NAME = "google/medsiglip-448"
SIGLIP_IMAGE_SIZE = 448

CLASSES = [
    "normal",
    "rash_maculopapular_g1", "rash_maculopapular_g2", "rash_maculopapular_g3",
    "rash_acneiform_g1", "rash_acneiform_g2", "rash_acneiform_g3",
    "periorbital_edema_g1", "periorbital_edema_g2", "periorbital_edema_g3",
    "sjs_prodrome_g1", "sjs_prodrome_g2", "sjs_prodrome_g3",
    "stomatitis_g1", "stomatitis_g2", "stomatitis_g3",
    "pruritus_g1", "pruritus_g2", "pruritus_g3",
    "alopecia_g1", "alopecia_g2",
]
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}


def build_head(num_classes: int = len(CLASSES)) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(EMBED_DIM, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )


class SigLIPClassifier:
    """AE classifier — supports both pre-computed vectors and raw images."""

    def __init__(self, head_path: str | Path, device: str = "cpu"):
        self.device = device
        self.head = build_head()
        self.head.load_state_dict(torch.load(head_path, map_location=device))
        self.head = self.head.to(device).eval()

        # Encoder (lazy-loaded on first image request)
        self._encoder = None
        self._processor = None

    def load_encoder(self, model_name: str = SIGLIP_MODEL_NAME, encoder_device: str | None = None):
        """Load MedSigLIP-448 vision encoder for image → embedding."""
        from transformers import AutoModel, AutoImageProcessor

        dev = encoder_device or self.device
        log.info("Loading MedSigLIP encoder: %s on %s", model_name, dev)
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._encoder = AutoModel.from_pretrained(model_name).to(dev).eval()
        self._encoder_device = dev
        log.info("MedSigLIP encoder loaded.")

    @property
    def encoder_loaded(self) -> bool:
        return self._encoder is not None

    def encode_image(self, image_b64: str) -> list[float]:
        """Encode a base64 image → 1152-dim embedding vector."""
        from PIL import Image

        if not self.encoder_loaded:
            raise RuntimeError("SigLIP encoder not loaded. Call load_encoder() first.")

        raw = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = img.resize((SIGLIP_IMAGE_SIZE, SIGLIP_IMAGE_SIZE), Image.BILINEAR)

        inputs = self._processor(images=img, return_tensors="pt").to(self._encoder_device)
        with torch.inference_mode():
            outputs = self._encoder.vision_model(**inputs)
            embedding = outputs.pooler_output[0]  # (1152,)

        return embedding.cpu().tolist()

    def predict(self, embedding: list[float], top_k: int = 3) -> dict:
        vec = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.head(vec)
            probs = torch.softmax(logits, dim=-1)[0]

        pred_idx = probs.argmax().item()
        pred_class = IDX_TO_CLASS[pred_idx]

        topk_vals, topk_idxs = probs.topk(min(top_k, len(CLASSES)))
        top_probs = {
            IDX_TO_CLASS[idx.item()]: round(val.item(), 4)
            for val, idx in zip(topk_vals, topk_idxs)
        }

        ae_term, grade = None, None
        if pred_class != "normal":
            parts = pred_class.rsplit("_g", 1)
            ae_term = parts[0]
            grade = int(parts[1]) if len(parts) == 2 else None

        return {
            "prediction": pred_class,
            "probability": round(probs[pred_idx].item(), 4),
            "top_k": top_probs,
            "ae_term": ae_term,
            "grade": grade,
        }

    def build_visual_assessment(self, embedding: list[float]) -> dict:
        """Convert SigLIP prediction into the visual_assessment format used by the nurse."""
        result = self.predict(embedding)
        findings = []
        if result["ae_term"]:
            findings.append({
                "ae_term": result["ae_term"],
                "estimated_grade": result["grade"],
                "confidence": result["probability"],
                "description": f"Visual analysis detected {result['ae_term'].replace('_', ' ')} (grade {result['grade']})",
            })
        general_obs = []
        for cls_name, prob in result["top_k"].items():
            if cls_name != "normal" and prob > 0.1:
                general_obs.append(f"{cls_name.replace('_', ' ')} ({prob:.1%})")

        return {
            "findings": findings,
            "general_observations": general_obs if general_obs else ["No significant visual abnormalities detected"],
            "raw_prediction": result,
        }

    def build_visual_assessment_from_image(self, image_b64: str) -> dict:
        """Full pipeline: image → encode → classify → assessment."""
        embedding = self.encode_image(image_b64)
        return self.build_visual_assessment(embedding)
