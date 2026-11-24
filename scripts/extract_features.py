"""Extract frozen DenseNet/ResNet features for every image in the splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qcalibnet.config import load_config
from qcalibnet.features import collect_split_frames, extract_missing
from qcalibnet.paths import SPLITS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--backbone", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    backbone = args.backbone or cfg["backbone"]
    frame = collect_split_frames(SPLITS_DIR)
    print(f"unique images: {len(frame)}")
    path = extract_missing(
        backbone=backbone,
        frame=frame,
        image_size=int(cfg["image_size"]),
        batch_size=int(cfg["feature_batch_size"]),
    )
    print(f"feature cache: {path}")


if __name__ == "__main__":
    main()
