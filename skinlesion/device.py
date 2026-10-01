"""Shared device selection for training and evaluation."""

from __future__ import annotations

import torch


def select_device(preferred: str | None = None) -> torch.device:
    if preferred:
        device = torch.device(preferred)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device
