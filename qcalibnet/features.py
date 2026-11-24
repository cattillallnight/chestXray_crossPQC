"""Frozen-backbone feature cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from qcalibnet.models import build_backbone
from qcalibnet.paths import FEATURES_DIR, ensure_dirs
from qcalibnet.torchdata import CXRImageDataset, lookup_features
from qcalibnet.transforms import eval_transform


def cache_path(backbone: str) -> Path:
    ensure_dirs()
    return FEATURES_DIR / f"{backbone}_features.npz"


def load_feature_cache(backbone: str) -> tuple[dict[str, int], np.ndarray] | tuple[None, None]:
    path = cache_path(backbone)
    if not path.is_file():
        return None, None
    blob = np.load(path, allow_pickle=True)
    paths = [str(p) for p in blob["paths"].tolist()]
    index = {p: i for i, p in enumerate(paths)}
    return index, blob["features"]


def save_feature_cache(backbone: str, paths: list[str], features: np.ndarray) -> Path:
    path = cache_path(backbone)
    np.savez_compressed(path, paths=np.asarray(paths, dtype=object), features=features.astype(np.float32))
    return path


def collect_split_frames(split_dir) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(Path(split_dir).rglob("*.csv")):
        if csv_path.name.endswith("_unused.csv"):
            continue
        frame = pd.read_csv(csv_path)
        if "image_path" not in frame.columns:
            continue
        frames.append(frame[["image_path"]].assign(source=csv_path.name))
    if not frames:
        raise FileNotFoundError(f"No split CSVs with image_path in {split_dir}")
    out = pd.concat(frames, ignore_index=True).drop_duplicates("image_path")
    return out.reset_index(drop=True)


@torch.no_grad()
def extract_missing(
    backbone: str,
    frame: pd.DataFrame,
    image_size: int = 224,
    batch_size: int = 32,
    device: str | None = None,
) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    index, matrix = load_feature_cache(backbone)
    known = set(index) if index else set()
    needed = [str(p) for p in frame["image_path"].astype(str) if str(p) not in known]
    if not needed and index is not None and matrix is not None:
        return cache_path(backbone)

    net, feat_dim = build_backbone(backbone)
    net = net.to(device)
    net.eval()
    transform = eval_transform(image_size)
    pending = pd.DataFrame({"image_path": needed, "label": [0] * len(needed)})
    loader = DataLoader(
        CXRImageDataset(pending, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )
    chunks = []
    paths = []
    for images, _, batch_paths in tqdm(loader, desc=f"extract {backbone}", leave=False):
        images = images.to(device, non_blocking=True)
        feats = net(images).detach().cpu().numpy().astype(np.float32)
        chunks.append(feats)
        paths.extend(list(batch_paths))

    new_feats = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, feat_dim), dtype=np.float32)
    if index is None or matrix is None:
        all_paths = paths
        all_feats = new_feats
    else:
        old_paths = [None] * len(index)
        for path, row in index.items():
            old_paths[row] = path
        all_paths = list(old_paths) + paths
        all_feats = np.concatenate([matrix, new_feats], axis=0) if len(new_feats) else matrix
    return save_feature_cache(backbone, all_paths, all_feats)


def features_for_frame(backbone: str, frame: pd.DataFrame) -> np.ndarray:
    index, matrix = load_feature_cache(backbone)
    if index is None or matrix is None:
        raise FileNotFoundError(f"Feature cache missing: {cache_path(backbone)}")
    return lookup_features(index, matrix, frame)
