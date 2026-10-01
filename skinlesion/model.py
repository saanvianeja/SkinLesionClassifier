"""MobileNetV2 binary classifier matching the saved checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

from skinlesion.config import CLASSIFIER_DROPOUT, DEFAULT_WEIGHTS_PATH, NUM_CLASSES


def create_model(num_classes: int = NUM_CLASSES, pretrained: bool = False) -> nn.Module:
    """Build MobileNetV2 with Dropout + Linear(last_channel, num_classes).

    The classifier layout must stay identical to the checkpoint:
    Sequential(Dropout(0.2), Linear(last_channel, 2)).
    """
    if pretrained:
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(CLASSIFIER_DROPOUT),
        nn.Linear(model.last_channel, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module) -> nn.Module:
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model


def unfreeze_backbone(model: nn.Module) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = True
    return model


def trainable_parameter_counts(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def load_model(
    weights_path: str | Path | None = None,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict]:
    """Load checkpoint state_dict. Returns (model, load_state_dict result)."""
    if device is None:
        device = torch.device("cpu")
    path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model = create_model(num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    load_result = model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()
    return model, load_result
