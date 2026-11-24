"""Data access. Read-only over v1 artefacts.

Feature cache is a single npz keyed by image path; v1 splits carry image_path.
Logits for the 45 v1 runs live under results/figures/logits/*.npz (only
chexpert_test and vindr_bal were dumped in v1; vindr_nat logits do not exist
and are recomputed in Stage 1 from the checkpoints if needed).
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from .paths import FEATURES, LOGITS_V1, SPLITS, V1_HEADS, V1_SEEDS, PRIMARY_TASKS


@lru_cache(maxsize=1)
def _feature_cache():
    blob = np.load(FEATURES, allow_pickle=True)
    paths = [str(p) for p in blob["paths"].tolist()]
    index = {p: i for i, p in enumerate(paths)}
    # also index forward-slash variants
    for i, p in enumerate(paths):
        index.setdefault(p.replace("\\", "/"), i)
    return index, blob["features"]


def features_for(frame_rows: list[dict]) -> np.ndarray:
    index, mat = _feature_cache()
    idx = []
    for r in frame_rows:
        p = str(r["image_path"])
        key = p if p in index else p.replace("\\", "/")
        if key not in index:
            raise KeyError(f"no cached feature for {p}")
        idx.append(index[key])
    return mat[np.asarray(idx, dtype=np.int64)]


def read_split(task: str, name: str) -> list[dict]:
    p = SPLITS / task / name
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_xy(task: str, name: str):
    rows = read_split(task, name)
    y = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
    return rows, y


def load_v1_logits(task: str, head: str, seed: int, split: str):
    """Return (logits, labels, temperature) or None if absent.

    Checks the v1 dump dir first, then the extended dump written by
    dump_v1_extra.py (chexpert_val, vindr_nat).
    """
    from .paths import V2
    for base in (LOGITS_V1, V2 / "logits_v1_ext"):
        p = base / f"{task}_{head}_seed{seed}_{split}.npz"
        if p.is_file():
            b = np.load(p, allow_pickle=True)
            return (b["logits"].astype(np.float64), b["labels"].astype(np.int64),
                    float(b["temperature"][0]))
    return None


def available_v1_logits() -> list[dict]:
    """Summary of which (task, head, seed, split) logit files exist."""
    out = []
    for task in PRIMARY_TASKS + ["pneumonia_proxy"]:
        for head in V1_HEADS:
            for seed in V1_SEEDS:
                for split in ("chexpert_test", "vindr_bal"):
                    p = LOGITS_V1 / f"{task}_{head}_seed{seed}_{split}.npz"
                    out.append({"task": task, "head": head, "seed": seed,
                                "split": split, "exists": p.is_file()})
    return out