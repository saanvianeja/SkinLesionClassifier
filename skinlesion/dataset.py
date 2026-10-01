"""HAM10000 image dataset. Loads from fixed split manifests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from skinlesion.config import HAM10000_DIR

IMAGE_DIRS = ("HAM10000_images_part_1", "HAM10000_images_part_2")


def build_image_index(root: Path | None = None) -> dict[str, Path]:
    root = Path(root) if root else HAM10000_DIR
    index: dict[str, Path] = {}
    for folder in IMAGE_DIRS:
        d = root / folder
        if not d.is_dir():
            continue
        for path in d.iterdir():
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                index[path.stem] = path
    return index


def resolve_image_path(row, index: dict[str, Path], root: Path) -> Path:
    if "image_path" in row and pd.notna(row["image_path"]):
        candidate = Path(row["image_path"])
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.exists():
            return candidate
    image_id = str(row["image_id"])
    if image_id in index:
        return index[image_id]
    raise FileNotFoundError(f"No image file for image_id={image_id}")


class HAM10000Dataset(Dataset):
    def __init__(self, csv_file: str | Path, transform=None, ham_root: Path | None = None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.ham_root = Path(ham_root) if ham_root else HAM10000_DIR
        self.index = build_image_index(self.ham_root)
        required = {"image_id", "label"}
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(f"Split CSV missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        path = resolve_image_path(row, self.index, self.ham_root)
        try:
            image = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Unreadable image: {path}") from exc
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label


# Back-compat alias used nowhere in Streamlit.
ISICDataset = HAM10000Dataset
