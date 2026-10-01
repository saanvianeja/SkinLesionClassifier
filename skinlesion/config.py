"""Project paths and model constants."""

from __future__ import annotations

import os
from pathlib import Path

from skinlesion.labels import CLASS_INDEX_TO_NAME, DX_TO_LABEL  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch"))
ARTIFACTS_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "data"
HAM10000_DIR = DATA_DIR / "ham10000"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUTS_DIR = ROOT / "outputs"

LEGACY_WEIGHTS_PATH = ARTIFACTS_DIR / "best_isic_model.pth"
FROZEN_BASELINE_PATH = ARTIFACTS_DIR / "mobilenetv2_frozen_baseline.pth"
FINETUNED_PATH = ARTIFACTS_DIR / "mobilenetv2_finetuned.pth"
DEFAULT_WEIGHTS_PATH = FINETUNED_PATH
DECISION_THRESHOLD = 0.29

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 2
CLASSIFIER_DROPOUT = 0.2
SEED = 42
