"""Fail fast if the train pipeline is not runnable."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from qcalibnet.datasets import vindr_labels_available
from qcalibnet.features import cache_path, load_feature_cache
from qcalibnet.models import QCalibNet
from qcalibnet.paths import CHEXPERT_TRAIN_CSV, SPLITS_DIR, chexpert_image_path


def main() -> int:
    errors = []
    warnings = []
    if not CHEXPERT_TRAIN_CSV.is_file():
        errors.append(f"missing {CHEXPERT_TRAIN_CSV}")
    sample = chexpert_image_path("CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg")
    if not sample.is_file():
        errors.append(f"missing CheXpert image {sample}")
    if not vindr_labels_available():
        warnings.append("VinDr train.csv missing: CheXpert-only train is possible, external test is not.")

    model = QCalibNet(feat_dim=32, head="product_pqc", n_qubits=6, n_layers=2)
    x = torch.randn(2, 32, requires_grad=True)
    y = model(x)
    y.sum().backward()
    if model.bottleneck.proj.weight.grad is None:
        errors.append("PQC bottleneck gradient is None")
    if x.grad is None:
        errors.append("PQC input gradient is None")

    splits = list(SPLITS_DIR.glob("*/chexpert_train.csv"))
    if not splits:
        warnings.append("splits not built yet: python scripts/build_splits.py")
    index, matrix = load_feature_cache("densenet121")
    if index is None:
        warnings.append(f"features not extracted yet: python scripts/extract_features.py ({cache_path('densenet121')})")

    print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    for w in warnings:
        print("WARN", w)
    for e in errors:
        print("ERR ", e)
    if errors:
        return 1
    print("READY: library + PQC grads OK")
    if splits and index is not None:
        print("READY: splits + features present — you can run scripts/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
