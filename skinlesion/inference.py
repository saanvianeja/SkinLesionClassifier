"""CNN-only inference using the shared model and eval transforms."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from skinlesion.config import CLASS_INDEX_TO_NAME, DEFAULT_WEIGHTS_PATH
from skinlesion.model import load_model
from skinlesion.transforms import eval_transforms


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Apply the original eval Resize/ToTensor/ImageNet Normalize pipeline."""
    tensor = eval_transforms()(image.convert("RGB"))
    return tensor.unsqueeze(0)


def predict_from_pil(
    image: Image.Image,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    weights_path: str | Path | None = None,
) -> dict:
    """Return raw logits, softmax probabilities, and predicted class index."""
    if device is None:
        device = torch.device("cpu")
    if model is None:
        model, _ = load_model(weights_path=weights_path or DEFAULT_WEIGHTS_PATH, device=device)

    batch = preprocess_image(image).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probabilities).item())

    probs = probabilities.detach().cpu().tolist()
    return {
        "logits": logits.detach().cpu().squeeze(0).tolist(),
        "probabilities": {CLASS_INDEX_TO_NAME[i]: float(p) for i, p in enumerate(probs)},
        "probability_vector": probs,
        "pred_index": pred_idx,
        "pred_label": CLASS_INDEX_TO_NAME[pred_idx],
        "predicted_class_probability": float(probs[pred_idx]),
        "label_mapping_unverified": True,
    }


def predict(
    image_path: str | Path,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    weights_path: str | Path | None = None,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    return predict_from_pil(image, model=model, device=device, weights_path=weights_path)
