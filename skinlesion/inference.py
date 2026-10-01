"""CNN inference for the locked fine-tuned MobileNetV2.

Prediction uses malignant_probability >= DECISION_THRESHOLD (0.29), not argmax.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from skinlesion.config import CLASS_INDEX_TO_NAME, DECISION_THRESHOLD, DEFAULT_WEIGHTS_PATH
from skinlesion.device import select_device
from skinlesion.model import load_model
from skinlesion.transforms import eval_transforms


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Shared eval preprocess: resize 224, ImageNet normalize."""
    return eval_transforms()(image.convert("RGB")).unsqueeze(0)


def predict_from_pil(
    image: Image.Image,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    weights_path: str | Path | None = None,
    threshold: float = DECISION_THRESHOLD,
) -> dict:
    if device is None:
        device = select_device()
    if model is None:
        model, _ = load_model(weights_path=weights_path or DEFAULT_WEIGHTS_PATH, device=device)

    batch = preprocess_image(image).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)[0]

    probs = probabilities.detach().cpu().tolist()
    p_benign, p_malignant = float(probs[0]), float(probs[1])
    pred_index = 1 if p_malignant >= threshold else 0
    return {
        "logits": logits.detach().cpu().squeeze(0).tolist(),
        "probabilities": {CLASS_INDEX_TO_NAME[0]: p_benign, CLASS_INDEX_TO_NAME[1]: p_malignant},
        "probability_vector": [p_benign, p_malignant],
        "benign_probability": p_benign,
        "malignant_probability": p_malignant,
        "pred_index": pred_index,
        "pred_label": CLASS_INDEX_TO_NAME[pred_index],
        "decision_threshold": threshold,
        "checkpoint": str(weights_path or DEFAULT_WEIGHTS_PATH),
    }


def predict(
    image_path: str | Path,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    weights_path: str | Path | None = None,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    return predict_from_pil(image, model=model, device=device, weights_path=weights_path)
