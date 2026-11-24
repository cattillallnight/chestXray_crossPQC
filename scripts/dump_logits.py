"""Dump logits/labels from trained checkpoints for reliability diagrams.

Does not retrain. Resumes by skipping existing .npz files.
Default: primary paper tasks, CheXpert test + VinDr balanced, all seeds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qcalibnet.config import load_config
from qcalibnet.engine import evaluate_split, select_device
from qcalibnet.features import features_for_frame
from qcalibnet.models import QCalibNet
from qcalibnet.paths import FIGURES_DIR, RUNS_DIR, SPLITS_DIR, TABLES_DIR, ensure_dirs

PRIMARY_TASKS = ["no_finding", "cardiomegaly", "pleural_effusion", "consolidation"]
SPLITS = {
    "chexpert_test": "chexpert_test.csv",
    "vindr_bal": "vindr_external_test.csv",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=PRIMARY_TASKS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    parser.add_argument("--heads", nargs="*", default=["mlp", "product_pqc", "entangling_pqc"])
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs()
    out_dir = FIGURES_DIR / "logits"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_skip = 0
    for task in args.tasks:
        for head in args.heads:
            for seed in args.seeds:
                run_id = f"{task}_{head}_seed{seed}"
                ckpt_path = RUNS_DIR / f"{run_id}.pt"
                table_path = TABLES_DIR / f"{run_id}.json"
                if not ckpt_path.is_file():
                    print(f"missing {ckpt_path.name}")
                    continue
                payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                temperature = float(payload["temperature"])
                train_df = pd.read_csv(SPLITS_DIR / task / "chexpert_train.csv")
                feat_dim = int(features_for_frame(cfg["backbone"], train_df.iloc[:1]).shape[1])
                device = select_device(head)
                model = QCalibNet(
                    feat_dim=feat_dim,
                    head=head,
                    n_qubits=int(cfg["n_qubits"]),
                    n_layers=int(cfg["n_layers"]),
                    mlp_hidden=int(cfg["mlp_hidden"]),
                    mlp_dropout=float(cfg["mlp_dropout"]),
                )
                model.load_state_dict(payload["state_dict"])
                model.to(device)
                model.eval()
                for split_name, csv_name in SPLITS.items():
                    out_path = out_dir / f"{run_id}_{split_name}.npz"
                    if out_path.is_file():
                        n_skip += 1
                        continue
                    frame = pd.read_csv(SPLITS_DIR / task / csv_name)
                    x = features_for_frame(cfg["backbone"], frame)
                    y = frame["label"].to_numpy()
                    ev = evaluate_split(model, x, y, cfg, device, temperature=temperature)
                    np.savez_compressed(
                        out_path,
                        logits=ev["logits"].astype(np.float32),
                        labels=ev["labels"].astype(np.int8),
                        temperature=np.array([temperature], dtype=np.float32),
                        task=np.array(task),
                        head=np.array(head),
                        seed=np.array(seed),
                        split=np.array(split_name),
                    )
                    print(f"wrote {out_path.name} n={len(y)} T={temperature:.4g}")
                    n_ok += 1
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    print(f"dumped {n_ok}  skipped {n_skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
