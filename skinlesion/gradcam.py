"""Grad-CAM for the locked MobileNetV2 classifier.

Explains the *displayed prediction class* (threshold-based Benign=0 or Malignant=1)
by attributing the corresponding logit to the last convolutional feature map in
`model.features`. This is an explanatory visualization, not clinical evidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from skinlesion.config import DEFAULT_WEIGHTS_PATH
from skinlesion.device import select_device
from skinlesion.inference import predict_from_pil, preprocess_image
from skinlesion.model import load_model


def _last_conv_module(model: nn.Module) -> nn.Conv2d:
    last = None
    for module in model.features.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    if last is None:
        raise RuntimeError("No Conv2d found in MobileNetV2 features.")
    return last


def _colorize(cam: np.ndarray) -> np.ndarray:
    import matplotlib.cm as cm

    colored = cm.jet(np.clip(cam, 0.0, 1.0))[..., :3]
    return colored.astype(np.float32)


def generate_gradcam(
    image: Image.Image,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    weights_path: str | Path | None = None,
    target_index: int | None = None,
) -> tuple[Image.Image, int, np.ndarray]:
    """Overlay Grad-CAM for `target_index` (the displayed predicted class).

    Returns (overlay RGB matching original size, class index explained, cam 0-1).
    Does not update model parameters.
    """
    if device is None:
        device = select_device()
    if model is None:
        model, _ = load_model(weights_path=weights_path or DEFAULT_WEIGHTS_PATH, device=device)
    model.eval()

    rgb = image.convert("RGB")
    orig_w, orig_h = rgb.size
    if target_index is None:
        target_index = predict_from_pil(rgb, model=model, device=device)["pred_index"]

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _input, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = _last_conv_module(model)
    handles = [
        target_layer.register_forward_hook(forward_hook),
        target_layer.register_full_backward_hook(backward_hook),
    ]
    try:
        batch = preprocess_image(rgb).to(device)
        batch.requires_grad_(True)
        logits = model(batch)
        model.zero_grad(set_to_none=True)
        logits[0, int(target_index)].backward()
        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        act = activations[0]
        grad = gradients[0]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
        cam = cam[0, 0].detach().float().cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)

    cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
    cam_min, cam_max = float(cam.min()), float(cam.max())
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    cam_u8 = np.clip(cam * 255.0, 0, 255).astype(np.uint8)
    cam_img = Image.fromarray(cam_u8, mode="L").resize((orig_w, orig_h), resample=Image.BILINEAR)
    cam_resized = np.asarray(cam_img, dtype=np.float32) / 255.0
    orig = np.asarray(rgb).astype(np.float32) / 255.0
    overlay = (0.55 * orig) + (0.45 * _colorize(cam_resized))
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay), int(target_index), cam_resized
