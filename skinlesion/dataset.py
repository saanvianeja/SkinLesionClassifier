"""ISIC image dataset. Expects CSV columns image_name, label, diagnosis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ISICDataset(Dataset):
    def __init__(self, csv_file: str | Path, img_dir: str | Path, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        img_name = row["image_name"]
        label = int(row["label"])
        diagnosis = row["diagnosis"]
        img_path = self.img_dir / diagnosis / img_name
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
