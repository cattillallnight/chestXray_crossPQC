"""Filesystem layout for Q-CalibNet.

Set ``QCALIBNET_ROOT``, ``QCALIBNET_DATA_DIR``, ``QCALIBNET_CHEXPERT_DIR``,
and ``QCALIBNET_VINDR_DIR`` to locate project and private data.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("QCALIBNET_ROOT", Path(__file__).resolve().parent.parent))
PRIVATE_ROOT = Path(os.environ.get("QCALIBNET_PRIVATE_ROOT", ROOT.parent / "MIDLquantumpersional"))
PRIVATE_ROOT = PRIVATE_ROOT if PRIVATE_ROOT.is_dir() else ROOT

CHEXPERT_DIR = Path(os.environ.get("QCALIBNET_CHEXPERT_DIR", PRIVATE_ROOT / "CheXpert"))
CHEXPERT_TRAIN_CSV = CHEXPERT_DIR / "train.csv"
CHEXPERT_VALID_CSV = CHEXPERT_DIR / "valid.csv"
CHEXPERT_TRAIN_DIR = CHEXPERT_DIR / "train"
CHEXPERT_VALID_DIR = CHEXPERT_DIR / "valid"

VINDR_DIR = Path(os.environ.get("QCALIBNET_VINDR_DIR", PRIVATE_ROOT / "VinBigData"))
VINDR_TRAIN_DIR = VINDR_DIR / "train"
VINDR_TEST_DIR = VINDR_DIR / "test"
VINDR_META_CSV = VINDR_DIR / "train_meta.csv"
DATA_DIR = Path(os.environ.get("QCALIBNET_DATA_DIR", PRIVATE_ROOT / "data"))
VINDR_LABEL_CANDIDATES = (
    VINDR_DIR / "train.csv",
    DATA_DIR / "external" / "vindr_train.csv",
)

CONFIG_DIR = ROOT / "configs"
SPLITS_DIR = DATA_DIR / "splits"
FEATURES_DIR = DATA_DIR / "features"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


def ensure_dirs() -> None:
    for path in (SPLITS_DIR, FEATURES_DIR, RUNS_DIR, TABLES_DIR, FIGURES_DIR, DATA_DIR / "external"):
        path.mkdir(parents=True, exist_ok=True)


def resolve_vindr_label_csv() -> Path | None:
    for path in VINDR_LABEL_CANDIDATES:
        if path.is_file():
            return path
    return None


def chexpert_image_path(csv_path: str) -> Path:
    """Map CheXpert CSV Path to the local CheXpert-v1.0-small tree."""
    normalized = csv_path.replace("\\", "/")
    marker = "CheXpert-v1.0-small/"
    if marker in normalized:
        rel = normalized.split(marker, 1)[1]
    elif normalized.startswith("train/") or normalized.startswith("valid/"):
        rel = normalized
    else:
        rel = "/".join(Path(normalized).parts[-4:])
    return CHEXPERT_DIR / Path(rel)


def vindr_image_path(image_id: str) -> Path:
    return VINDR_TRAIN_DIR / f"{image_id}.png"
