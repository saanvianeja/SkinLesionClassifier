"""CNN-only skin lesion classifier package."""

from skinlesion.inference import predict, predict_from_pil
from skinlesion.model import create_model, load_model

__all__ = ["create_model", "load_model", "predict", "predict_from_pil"]
