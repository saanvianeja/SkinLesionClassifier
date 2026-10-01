"""Project paths and model constants.

Class names follow the existing inference code (argmax 0/1). That mapping has
not been verified against the original training CSV or label-generation process.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
DEFAULT_WEIGHTS_PATH = ARTIFACTS_DIR / "best_isic_model.pth"
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 2
CLASSIFIER_DROPOUT = 0.2

# UNVERIFIED compatibility mapping used by the original predict_with_cnn().
# Do not treat this as confirmed ground truth until the training labels are inspected.
CLASS_INDEX_TO_NAME = {
    0: "Benign",
    1: "Malignant",
}
