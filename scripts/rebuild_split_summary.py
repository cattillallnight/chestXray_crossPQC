"""Rebuild data/splits/summary.json from existing CSVs. Does not resample."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qcalibnet.config import load_config
from qcalibnet.datasets import cohort_summary
from qcalibnet.paths import SPLITS_DIR
import pandas as pd


def main() -> int:
    cfg = load_config()
    summary = {"seed": int(cfg["seed"]), "tasks": {}, "note": "Rebuilt from existing split CSVs; splits were not resampled."}
    for task in cfg["tasks"]:
        task_dir = SPLITS_DIR / task
        if not task_dir.is_dir():
            continue
        task_sum: dict = {}
        for name in ("train", "val", "test"):
            path = task_dir / f"chexpert_{name}.csv"
            if path.is_file():
                task_sum.setdefault("chexpert", {})[name] = cohort_summary(pd.read_csv(path))
        bal = task_dir / "vindr_external_test.csv"
        nat = task_dir / "vindr_external_test_natural.csv"
        if bal.is_file():
            task_sum["vindr_external_test"] = cohort_summary(pd.read_csv(bal))
        if nat.is_file():
            task_sum["vindr_external_test_natural"] = cohort_summary(pd.read_csv(nat))
        summary["tasks"][task] = task_sum
        print(task, json.dumps(task_sum, indent=2))
    out = SPLITS_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
