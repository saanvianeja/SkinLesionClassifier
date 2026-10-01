"""Grad-CAM for MobileNetV2 using the shared model and eval transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from skinlesion.config import DEFAULT_WEIGHTS_PATH, IMAGE_SIZE
from skinlesion.inference import preprocess_image
from skinlesion.model import load_model


def _last_conv_module(model: torch.nn.Module) -> torch.nn.Module:
    """MobileNetV2 stores conv blocks in model.features; last layer is Conv2dNormActivation."""
    return model.features[-1]


def generate_gradcam(
    image: Image.Image,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    weights_path: str | Path | None = None,
    target_index: int | None = None,
) -> tuple[Image.Image, int]:
    """Return a heatmap overlay and the class index used for CAM.

    Requires a gradient through the last conv activations, so this is not run
    under torch.no_grad().
    """
    if device is None:
        device = torch.device("cpu")
    if model is None:
        model, _ = load_model(weights_path=weights_path or DEFAULT_WEIGHTS_PATH, device=device)
    model.eval()

    activations = []
    gradients = []

    def forward_hook(_module, _input, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = _last_conv_module(model)
    handles = [
        target_layer.register_forward_hook(forward_hook),
        target_layer.register_full_backward_hook(backward_hook),
    ]

    rgb = image.convert("RGB")
    batch = preprocess_image(rgb).to(device)
    batch.requires_grad_(True)
    logits = model(batch)
    if target_index is None:
        target_index = int(torch.argmax(logits, dim=1).item())

    model.zero_grad(set_to_none=True)
    logits[0, target_index].backward()

    for handle in handles:
        handle.remove()

    act = activations[0].detach()[0]
    grad = gradients[0].detach()[0]
    weights = grad.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * act).sum(dim=0))
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = F.interpolate(
        cam[None, None, :, :],
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()

    heatmap = np.uint8(255 * cam)
    heatmap_img = Image.fromarray(heatmap).resize(rgb.size).convert("RGB")
    # Simple red overlay without adding OpenCV as a dependency.
    overlay = Image.blend(rgb.resize(rgb.size), heatmap_img, alpha=0.45)
    return overlay, target_index
