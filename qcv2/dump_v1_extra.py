"""Dump the v1 logits that were never written: chexpert_val and vindr_nat.

Read-only over checkpoints. Writes results/v2/logits_v1_ext/*.npz.
Needed because Stage 1.2 must fit recalibration on CheXpert val, and Stage 1.5
needs natural-prevalence VinDr metrics; v1 dumped neither.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qcalibnet.config import load_config
from qcalibnet.features import features_for_frame
from qcalibnet.models import QCalibNet
from qcv2.paths import LOGITS_V1, RUNS, SPLITS, V1_HEADS, V1_SEEDS, V2, PRIMARY_TASKS, ALL_TASKS
from qcv2.state import log, warn

OUT = V2 / "logits_v1_ext"

SPLITS_MAP = {
    "chexpert_val": "chexpert_val.csv",
    "vindr_nat": "vindr_external_test_natural.csv",
}


def _frame_rows(csv_path: Path):
    import csv
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _df(rows):
    import pandas as pd
    return pd.DataFrame(rows)


def dump_one(task: str, head: str, seed: int, cfg: dict, device: torch.device):
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = f"{task}_{head}_seed{seed}"
    ck = RUNS / f"{run_id}.pt"
    if not ck.is_file():
        return []
    payload = torch.load(ck, map_location="cpu", weights_only=False)
    train_rows = _frame_rows(SPLITS / task / "chexpert_train.csv")
    feat_dim = int(features_for_frame(cfg["backbone"], _df(train_rows[:1])).shape[1])
    model = QCalibNet(feat_dim=feat_dim, head=head, n_qubits=int(cfg["n_qubits"]),
                      n_layers=int(cfg["n_layers"]), mlp_hidden=int(cfg["mlp_hidden"]),
                      mlp_dropout=float(cfg["mlp_dropout"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    written = []
    for split, csv_name in SPLITS_MAP.items():
        rows = _frame_rows(SPLITS / task / csv_name)
        x = features_for_frame(cfg["backbone"], _df(rows))
        y = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
        logits = []
        with torch.no_grad():
            for i in range(0, len(x), 64):
                xb = torch.from_numpy(x[i:i + 64].astype(np.float32)).to(device)
                logits.append(model(xb).detach().cpu().numpy())
        z = np.concatenate(logits).astype(np.float64)
        out = OUT / f"{run_id}_{split}.npz"
        np.savez_compressed(out, logits=z.astype(np.float32), labels=y.astype(np.int8),
                            temperature=np.array([float(payload["temperature"])], dtype=np.float32),
                            task=np.array(task), head=np.array(head), seed=np.array(seed),
                            split=np.array(split))
        written.append(out.name)
    del model
    return written


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = 0
    skipped = 0
    for task in ALL_TASKS:
        for head in V1_HEADS:
            for seed in V1_SEEDS:
                run_id = f"{task}_{head}_seed{seed}"
                if all((OUT / f"{run_id}_{s}.npz").is_file() for s in SPLITS_MAP):
                    skipped += 12
                    continue
                try:
                    w = dump_one(task, head, seed, cfg, device)
                    n += len(w)
                    log(f"  dumped {run_id}: {len(w)} splits")
                except Exception as exc:  # noqa: BLE001
                    warn(f"dump failed for {run_id}: {exc}", "dump_failure")
    log(f"dump_v1_extra: wrote {n} files, skipped-existing batches {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())