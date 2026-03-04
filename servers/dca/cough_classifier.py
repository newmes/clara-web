"""Cough Detection Classifier — 2-stage HeAR pipeline.

Pipeline:
    base64 WAV → librosa(16kHz)
    → segment_cough() — energy-based candidate extraction
    → HeAR 512-D embedding
    → Stage 1: 3-class sklearn (none/dry/wet) — filter non-cough
    → Stage 2: 2-class sklearn (dry/wet) — re-classify confirmed cough
    → majority vote → assessment dict

Dependencies:
    - Segmentation: detect-segment-cough/src/segmentation.py (energy hysteresis)
    - HeAR: google/hear base model (TF SavedModel, embedding extraction only)
    - Stage 1: 3-class sklearn (hear_mixed_model/{classifier,label_encoder}.joblib)
    - Stage 2: 2-class sklearn (hear_cough_only_model/{classifier,label_encoder}.joblib)
"""

from __future__ import annotations

import base64
import logging
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

log = logging.getLogger("care-ai-api.cough")

SR = 16_000
TARGET_LEN = 2 * SR  # 32 000 samples (2 sec for HeAR)


class CoughClassifier:
    """2-stage HeAR cough detector + dry/wet classifier.

    Stage 1: segment_cough() → energy-based candidate segments
    Stage 2: HeAR + 3-class sklearn → none/dry/wet (cough detection)
    Stage 3: HeAR + 2-class sklearn → dry/wet re-classification
    """

    def __init__(
        self,
        segmentation_dir: str | Path,
        model_dir_3c: str | Path,
        model_dir_2c: str | Path,
        hear_gpu: str | None = None,
    ):
        self._load_segmentation(Path(segmentation_dir))
        self._load_hear(hear_gpu)
        self._load_classifiers(Path(model_dir_3c), Path(model_dir_2c))
        log.info("CoughClassifier ready (2-stage HeAR).")

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_segmentation(self, seg_dir: Path) -> None:
        """Import segment_cough from segmentation directory."""
        seg_path = str(seg_dir)
        if seg_path not in sys.path:
            sys.path.insert(0, seg_path)
        from segmentation import segment_cough
        self._segment_cough = segment_cough
        log.info("Segmentation loaded from %s", seg_path)

    def _load_hear(self, hear_gpu: str | None) -> None:
        """Load HeAR TF SavedModel via huggingface_hub snapshot."""
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        warnings.filterwarnings("ignore", module="tensorflow")

        import tensorflow as tf
        from huggingface_hub import snapshot_download

        if hear_gpu is not None:
            gpus = tf.config.list_physical_devices("GPU")
            try:
                idx = int(hear_gpu)
                if gpus and idx < len(gpus):
                    tf.config.set_visible_devices(gpus[idx], "GPU")
                    tf.config.experimental.set_memory_growth(gpus[idx], True)
                    log.info("HeAR pinned to GPU %d", idx)
            except (ValueError, RuntimeError) as exc:
                log.warning("HeAR GPU config failed (%s), falling back to default", exc)
        else:
            # Force CPU-only: hide all GPUs from TF to avoid CUDA conflicts with PyTorch
            tf.config.set_visible_devices([], "GPU")
            log.info("HeAR forced to CPU (TF GPU disabled).")

        self._tf = tf
        hear_dir = snapshot_download("google/hear")
        model = tf.saved_model.load(hear_dir)
        self._serving = model.signatures["serving_default"]
        log.info("HeAR model loaded.")

    @staticmethod
    def _patch_sklearn_compat(estimator):
        """Patch sklearn version mismatch: models saved in 1.8+ lack deprecated attrs."""
        # LogisticRegression.multi_class was removed in sklearn 1.8.0 but
        # sklearn 1.7.x still expects it via get_params(). Add it back if missing.
        if hasattr(estimator, "predict") and not hasattr(estimator, "multi_class"):
            estimator.multi_class = "deprecated"

    def _load_classifiers(self, model_dir_3c: Path, model_dir_2c: Path) -> None:
        """Load 3-class (stage 1) and 2-class (stage 2) sklearn classifiers."""
        import joblib

        # Stage 1: 3-class (none / dry / wet) — cough detection
        self._clf_3c = joblib.load(model_dir_3c / "classifier.joblib")
        self._patch_sklearn_compat(self._clf_3c)
        self._le_3c = joblib.load(model_dir_3c / "label_encoder.joblib")
        log.info("Stage 1 (3-class) loaded: classes=%s", list(self._le_3c.classes_))

        # Stage 2: 2-class (dry / wet) — cough type classification
        self._clf_2c = joblib.load(model_dir_2c / "classifier.joblib")
        self._patch_sklearn_compat(self._clf_2c)
        self._le_2c = joblib.load(model_dir_2c / "label_encoder.joblib")
        log.info("Stage 2 (2-class) loaded: classes=%s", list(self._le_2c.classes_))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode_audio(self, audio_b64: str) -> np.ndarray:
        """Decode base64 WAV → numpy float32 array at 16 kHz."""
        import librosa

        raw_bytes = base64.b64decode(audio_b64)
        audio, _ = librosa.load(
            __import__("io").BytesIO(raw_bytes),
            sr=SR,
            mono=True,
        )
        return audio.astype(np.float32)

    def get_embedding(self, segment: np.ndarray) -> np.ndarray:
        """Pad/trim a segment to 2 s and extract HeAR 512-D embedding."""
        tf = self._tf
        if len(segment) < TARGET_LEN:
            padded = np.pad(segment, (0, TARGET_LEN - len(segment)))
        else:
            padded = segment[:TARGET_LEN]
        emb = self._serving(
            x=tf.constant(padded.reshape(1, -1), dtype=tf.float32),
        )["output_0"].numpy()
        return emb  # (1, 512)

    def classify_3c(self, emb: np.ndarray) -> dict:
        """Stage 1: 3-class (none/dry/wet) — is this a cough?"""
        pred = self._clf_3c.predict(emb)[0]
        proba = self._clf_3c.predict_proba(emb)[0]
        label = self._le_3c.inverse_transform([pred])[0]
        prob_dict = {
            c: round(float(p), 4)
            for c, p in zip(self._le_3c.classes_, proba)
        }
        return {"label": label, "probabilities": prob_dict}

    def classify_2c(self, emb: np.ndarray) -> dict:
        """Stage 2: 2-class (dry/wet) — re-classify confirmed cough."""
        pred = self._clf_2c.predict(emb)[0]
        proba = self._clf_2c.predict_proba(emb)[0]
        label = self._le_2c.inverse_transform([pred])[0]
        prob_dict = {
            c: round(float(p), 4)
            for c, p in zip(self._le_2c.classes_, proba)
        }
        return {"label": label, "probabilities": prob_dict}

    def classify_2stage(self, emb: np.ndarray) -> dict:
        """2-stage classification: 3-class filter → 2-class dry/wet."""
        s1 = self.classify_3c(emb)

        if s1["label"] == "none":
            return {
                "final_label": "none",
                "stage": "1-none",
                "probabilities": s1["probabilities"],
            }

        # Cough confirmed — re-classify with 2-class
        s2 = self.classify_2c(emb)
        none_prob = s1["probabilities"].get("none", 0.0)
        cough_prob = 1.0 - none_prob
        merged_probs = {
            "none": round(none_prob, 4),
            "dry": round(cough_prob * s2["probabilities"].get("dry", 0.0), 4),
            "wet": round(cough_prob * s2["probabilities"].get("wet", 0.0), 4),
        }
        return {
            "final_label": s2["label"],
            "stage": "2-cough",
            "probabilities": merged_probs,
            "stage1": s1,
            "stage2": s2,
        }

    def build_audio_assessment(self, audio_b64: str) -> dict:
        """Full pipeline: decode → segment → HeAR 2-stage classify → vote.

        Returns a structured assessment dict suitable for NurseEngine.
        """
        t0 = time.time()

        try:
            audio = self.decode_audio(audio_b64)
        except Exception as exc:
            log.warning("Audio decode failed: %s", exc)
            return self._empty_assessment(error=str(exc))

        duration_sec = len(audio) / SR

        # Step 1: Energy-based segmentation → candidate segments
        segments, mask = self._segment_cough(audio, SR, cough_padding=0)
        if not segments:
            return {
                "cough_detected": False,
                "num_cough_segments": 0,
                "num_energy_segments": 0,
                "duration_sec": round(duration_sec, 2),
                "segments": [],
                "majority_type": None,
                "latency_ms": round((time.time() - t0) * 1000),
            }

        # Compute segment boundaries for timestamps
        changes = np.diff(mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        # Step 2 & 3: HeAR embedding → 2-stage classification
        seg_results = []
        for i, seg in enumerate(segments):
            start_sec = round(starts[i] / SR, 3) if i < len(starts) else 0.0
            end_sec = round(ends[i] / SR, 3) if i < len(ends) else round(len(seg) / SR, 3)

            emb = self.get_embedding(seg)
            cls = self.classify_2stage(emb)

            seg_results.append({
                "index": i,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": round(end_sec - start_sec, 3),
                **cls,
            })

        # Filter to cough-only segments for majority vote
        cough_segs = [r for r in seg_results if r["final_label"] != "none"]

        if not cough_segs:
            return {
                "cough_detected": False,
                "num_cough_segments": 0,
                "num_energy_segments": len(seg_results),
                "duration_sec": round(duration_sec, 2),
                "segments": seg_results,
                "majority_type": None,
                "latency_ms": round((time.time() - t0) * 1000),
            }

        # Majority vote over cough segments
        cough_labels = [r["final_label"] for r in cough_segs]
        vote_counts = dict(Counter(cough_labels))
        majority_type = Counter(cough_labels).most_common(1)[0][0]

        return {
            "cough_detected": True,
            "num_cough_segments": len(cough_segs),
            "num_energy_segments": len(seg_results),
            "duration_sec": round(duration_sec, 2),
            "majority_type": majority_type,
            "vote_counts": vote_counts,
            "segments": seg_results,
            "latency_ms": round((time.time() - t0) * 1000),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_assessment(error: str | None = None) -> dict:
        d = {
            "cough_detected": False,
            "num_cough_segments": 0,
            "duration_sec": 0.0,
            "segments": [],
            "majority_type": None,
            "latency_ms": 0,
        }
        if error:
            d["error"] = error
        return d
