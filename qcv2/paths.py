"""Shared paths for the v2 pipeline. Read-only over v1 artefacts."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results" / "v2"
STATE = V2 / "state"
CSV = V2 / "csv"
FIG = V2 / "figures"
TEX = V2 / "tables"
LOGS = V2 / "logs"

SPLITS = ROOT / "data" / "splits"
FEATURES = ROOT / "data" / "features" / "densenet121_features.npz"
RUNS = ROOT / "results" / "runs"
TABLES = ROOT / "results" / "tables"
LOGITS_V1 = ROOT / "results" / "figures" / "logits"

PRIMARY_TASKS = ["no_finding", "cardiomegaly", "pleural_effusion", "consolidation"]
ALL_TASKS = PRIMARY_TASKS + ["pneumonia_proxy"]
V1_HEADS = ["mlp", "product_pqc", "entangling_pqc"]
V1_SEEDS = [42, 43, 44]

SPLIT_FILES = ["chexpert_train.csv", "chexpert_val.csv", "chexpert_test.csv",
               "vindr_external_test.csv", "vindr_external_test_natural.csv"]

N_BINS = 15


def ensure_dirs() -> None:
    for d in (V2, STATE, CSV, FIG, TEX, LOGS):
        d.mkdir(parents=True, exist_ok=True)