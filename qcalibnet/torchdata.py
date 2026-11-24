"""PyTorch datasets for CXR images and cached backbone features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CXRImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.paths = [str(p) for p in frame["image_path"].tolist()]
        self.labels = frame["label"].astype(np.float32).to_numpy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index])
        tensor = self.transform(image)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return tensor, label, self.paths[index]


class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(np.asarray(features, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


def lookup_features(index: dict[str, int], matrix: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    rows = []
    missing = []
    for path in frame["image_path"].astype(str):
        key = str(Path(path))
        if key not in index:
            alt = path.replace("\\", "/")
            if alt not in index:
                missing.append(path)
                continue
            key = alt
        rows.append(index[key])
    if missing:
        raise KeyError(f"{len(missing)} image paths have no cached features, e.g. {missing[0]}")
    return matrix[np.asarray(rows, dtype=np.int64)]
