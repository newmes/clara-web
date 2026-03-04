"""Face image analysis for CTCAE adverse-event detection.

Public API
----------
- ``analyze_face(image_bytes, method, *, config)``
  → :class:`~schemas.FaceAnalysisResult`
- ``train_classifier(image_dir, output_path, epochs, *, config)``
  → ``dict`` of training metrics

Two analysis methods:
  - ``"medgemma"``  — MedGemma 27B local VLM with CTCAE prompt
  - ``"medsiglip"`` — MedSigLIP-448 embeddings + linear classifier head
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from .config import MultimodalConfig, SIGLIP_CLASSES, build_ctcae_table_text, get_config
from .schemas import DetectedAE, FaceAnalysisResult

logger = logging.getLogger(__name__)


# =====================================================================
# CTCAE reference table embedded in the analysis prompt
# =====================================================================

_CTCAE_PROMPT = f"""\
You are a clinical dermatology expert. Analyze this patient's face image and detect
any adverse events (AEs) from the following CTCAE categories. Report ONLY what you
can visually confirm. If the face looks normal, return an empty list.

## AE Categories & Grading

{build_ctcae_table_text()}
## Output Format
Return a JSON array of detected AEs. Each element:
{{
  "ae_term": "<category>",
  "grade": <1|2|3>,
  "confidence": <0.0-1.0>,
  "reasoning": "<brief clinical reasoning>"
}}

If no AE is detected, return: []
"""


# =====================================================================
# Method 1: MedGemma 27B local VLM ("medgemma")
# =====================================================================

_medgemma_model = None
_medgemma_processor = None


def _load_medgemma(config: MultimodalConfig):
    """Load MedGemma 27B model and processor (cached at module level)."""
    global _medgemma_model, _medgemma_processor

    if _medgemma_model is not None:
        return _medgemma_model, _medgemma_processor

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    # Pick GPUs with enough free memory for 27B model (~54GB in bf16)
    max_memory = None
    if config.medgemma_device == "auto":
        max_memory = {}
        for i in config.medgemma_gpus:
            max_memory[i] = "46GiB"
        max_memory["cpu"] = "0GiB"

    logger.info("Loading MedGemma: %s (gpus=%s)", config.medgemma_model, config.medgemma_gpus)
    _medgemma_processor = AutoProcessor.from_pretrained(
        config.medgemma_model,
        token=config.hf_token,
    )
    _medgemma_model = AutoModelForImageTextToText.from_pretrained(
        config.medgemma_model,
        torch_dtype=torch.bfloat16,
        device_map=config.medgemma_device,
        max_memory=max_memory,
        token=config.hf_token,
    )
    _medgemma_model.eval()
    logger.info("MedGemma loaded.")
    return _medgemma_model, _medgemma_processor


def _analyze_medgemma(image_bytes: bytes, config: MultimodalConfig) -> FaceAnalysisResult:
    """MedGemma 27B VLM analysis with CTCAE prompt."""
    import torch
    from PIL import Image

    t0 = time.time()

    model, processor = _load_medgemma(config)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a clinical dermatology expert."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": _CTCAE_PROMPT},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generation = generation[0][input_len:]

    raw_text = processor.decode(generation, skip_special_tokens=True).strip()
    latency_ms = (time.time() - t0) * 1000

    detected = _parse_ae_json(raw_text)

    return FaceAnalysisResult(
        detected_aes=detected,
        model_used="medgemma",
        latency_ms=latency_ms,
    )


# =====================================================================
# JSON parsing (shared)
# =====================================================================

def _parse_ae_json(raw: str) -> list[DetectedAE]:
    """Parse model JSON output into DetectedAE list, with regex fallback."""
    # Try direct JSON parse
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [
                DetectedAE(
                    ae_term=item["ae_term"],
                    grade=int(item["grade"]),
                    confidence=float(item.get("confidence", 0.9)),
                    reasoning=item.get("reasoning", ""),
                    channel="face",
                )
                for item in data
            ]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Regex fallback: find JSON array in the text
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return [
                    DetectedAE(
                        ae_term=item.get("ae_term", "unknown"),
                        grade=int(item.get("grade", 1)),
                        confidence=float(item.get("confidence", 0.5)),
                        reasoning=item.get("reasoning", ""),
                        channel="face",
                    )
                    for item in data
                ]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    logger.warning("Failed to parse AE JSON from model output: %s", raw[:200])
    return []


# =====================================================================
# Method 2: MedSigLIP-448 + Linear Head ("medsiglip")
# =====================================================================

_siglip_model = None
_siglip_processor = None
_classifier_head = None


def _load_siglip(config: MultimodalConfig):
    """Load MedSigLIP-448 model and processor (cached at module level)."""
    global _siglip_model, _siglip_processor

    if _siglip_model is not None:
        return _siglip_model, _siglip_processor

    import torch
    from transformers import AutoModel, AutoImageProcessor

    logger.info("Loading MedSigLIP: %s on %s", config.siglip_model, config.siglip_device)
    _siglip_processor = AutoImageProcessor.from_pretrained(config.siglip_model)
    _siglip_model = AutoModel.from_pretrained(config.siglip_model).to(config.siglip_device)
    _siglip_model.eval()
    logger.info("MedSigLIP loaded.")
    return _siglip_model, _siglip_processor


def _load_classifier_head(path: Path, config: MultimodalConfig):
    """Load a trained linear classifier head."""
    global _classifier_head

    if _classifier_head is not None:
        return _classifier_head

    import torch
    import torch.nn as nn

    head = nn.Sequential(
        nn.Linear(1152, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, config.siglip_num_classes),
    )
    head.load_state_dict(torch.load(path, map_location=config.siglip_device, weights_only=True))
    head = head.to(config.siglip_device)
    head.eval()
    _classifier_head = head
    return _classifier_head


def _resize_for_medsiglip(img, size: int = 448):
    """Resize image to match MedSigLIP's expected input (448x448)."""
    from PIL import Image as PILImage
    return img.resize((size, size), PILImage.BILINEAR)


def _analyze_siglip(image_bytes: bytes, config: MultimodalConfig, head_path: Path | None = None) -> FaceAnalysisResult:
    """MedSigLIP-448 embedding + linear head classification."""
    import torch
    from PIL import Image

    t0 = time.time()

    if head_path is None:
        raise ValueError(
            "medsiglip method requires a trained classifier head. "
            "Pass head_path or use train_classifier() first."
        )

    model, processor = _load_siglip(config)
    head = _load_classifier_head(head_path, config)

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = _resize_for_medsiglip(img, config.siglip_image_size)
    inputs = processor(images=img, return_tensors="pt").to(config.siglip_device)

    with torch.no_grad():
        outputs = model.vision_model(**inputs)
        embedding = outputs.pooler_output  # (1, 1152)
        logits = head(embedding)           # (1, 13)
        probs = torch.softmax(logits, dim=-1).squeeze()

    latency_ms = (time.time() - t0) * 1000

    detected: list[DetectedAE] = []
    for idx, prob in enumerate(probs.tolist()):
        class_name = SIGLIP_CLASSES[idx]
        if class_name == "normal":
            continue
        if prob < 0.5:
            continue
        parts = class_name.rsplit("_g", 1)
        if len(parts) == 2:
            ae_term = parts[0]
            grade = int(parts[1])
        else:
            ae_term = class_name
            grade = 1

        detected.append(DetectedAE(
            ae_term=ae_term,
            grade=grade,
            confidence=prob,
            reasoning=f"MedSigLIP classifier: P({class_name})={prob:.3f}",
            channel="face",
        ))

    detected.sort(key=lambda x: x.confidence, reverse=True)

    return FaceAnalysisResult(
        detected_aes=detected,
        model_used="medsiglip",
        latency_ms=latency_ms,
    )


# =====================================================================
# Public API
# =====================================================================

def analyze_face(
    image_bytes: bytes,
    method: str = "medgemma",
    *,
    config: MultimodalConfig | None = None,
    head_path: Path | str | None = None,
) -> FaceAnalysisResult:
    """Analyze a face image for CTCAE adverse events.

    Parameters
    ----------
    image_bytes:
        PNG/JPEG image bytes.
    method:
        ``"medgemma"`` (MedGemma 27B local VLM) or
        ``"medsiglip"`` (MedSigLIP-448 + linear head).
    config:
        Override the default :class:`MultimodalConfig`.
    head_path:
        Path to trained classifier weights (required for ``"medsiglip"``).

    Returns
    -------
    FaceAnalysisResult
    """
    cfg = config or get_config()

    if method == "medgemma":
        return _analyze_medgemma(image_bytes, cfg)
    elif method == "medsiglip":
        hp = Path(head_path) if head_path else None
        return _analyze_siglip(image_bytes, cfg, head_path=hp)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'medgemma' or 'medsiglip'.")


def train_classifier(
    image_dir: str | Path,
    output_path: str | Path,
    epochs: int = 50,
    *,
    config: MultimodalConfig | None = None,
) -> dict[str, Any]:
    """Train a MedSigLIP linear classifier head on labelled AE images.

    Expected directory structure::

        image_dir/
            normal/
            maculopapular_rash_g1/
            maculopapular_rash_g2/
            ...

    Parameters
    ----------
    image_dir:
        Directory containing subdirectories named by class.
    output_path:
        Where to save the trained head weights (``.pt``).
    epochs:
        Training epochs (default 20).
    config:
        Override the default :class:`MultimodalConfig`.

    Returns
    -------
    dict with ``{"accuracy": float, "f1": float, "epochs": int, "classes": int}``
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    from sklearn.metrics import accuracy_score, f1_score

    cfg = config or get_config()
    image_dir = Path(image_dir)
    output_path = Path(output_path)

    model, processor = _load_siglip(cfg)

    class AEImageDataset(Dataset):
        def __init__(self, root: Path, class_names: list[str]):
            self.samples: list[tuple[Path, int]] = []
            for label_idx, name in enumerate(class_names):
                class_dir = root / name
                if not class_dir.is_dir():
                    logger.warning("Missing class dir: %s", class_dir)
                    continue
                for img_path in sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg")):
                    self.samples.append((img_path, label_idx))
            logger.info("Dataset: %d images, %d classes", len(self.samples), len(class_names))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            img = Image.open(path).convert("RGB")
            img = _resize_for_medsiglip(img, cfg.siglip_image_size)
            inputs = processor(images=img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
            return pixel_values, label

    dataset = AEImageDataset(image_dir, SIGLIP_CLASSES)
    if len(dataset) == 0:
        raise ValueError(f"No images found in {image_dir}")

    n_val = max(1, len(dataset) // 5)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    for param in model.parameters():
        param.requires_grad = False

    head = nn.Sequential(
        nn.Linear(1152, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, cfg.siglip_num_classes),
    ).to(cfg.siglip_device)

    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        head.train()
        for pixel_values, labels in train_loader:
            pixel_values = pixel_values.to(cfg.siglip_device)
            labels = labels.to(cfg.siglip_device)

            with torch.no_grad():
                vision_out = model.vision_model(pixel_values=pixel_values)
                embeddings = vision_out.pooler_output

            logits = head(embeddings)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        head.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for pixel_values, labels in val_loader:
                pixel_values = pixel_values.to(cfg.siglip_device)
                vision_out = model.vision_model(pixel_values=pixel_values)
                embeddings = vision_out.pooler_output
                logits = head(embeddings)
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        acc = accuracy_score(all_labels, all_preds)
        logger.info("Epoch %d/%d — val accuracy: %.3f", epoch + 1, epochs, acc)

        if acc > best_acc:
            best_acc = acc
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(head.state_dict(), output_path)

    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    logger.info("Training complete. Best accuracy: %.3f, F1: %.3f", best_acc, f1)
    return {
        "accuracy": best_acc,
        "f1": f1,
        "epochs": epochs,
        "classes": cfg.siglip_num_classes,
        "output_path": str(output_path),
    }
