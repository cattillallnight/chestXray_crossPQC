"""Attach VinDr external metrics to already-trained CheXpert checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qcalibnet.config import load_config
from qcalibnet.engine import evaluate_split
from qcalibnet.features import features_for_frame
from qcalibnet.models import QCalibNet
from qcalibnet.paths import RUNS_DIR, SPLITS_DIR, TABLES_DIR


def flatten(prefix: str, metrics: dict | None) -> dict:
    if metrics is None:
        return {}
    return {f"{prefix}_{k}": v for k, v in metrics.items() if k not in {"logits", "labels"}}


def eval_csv(model, backbone, csv_path: Path, cfg, device, temperature):
    if not csv_path.is_file():
        return None
    frame = pd.read_csv(csv_path)
    x = features_for_frame(backbone, frame)
    y = frame["label"].to_numpy()
    return evaluate_split(model, x, y, cfg, device, temperature=temperature)


def main() -> int:
    cfg = load_config()
    n_ok = 0
    for table_path in sorted(TABLES_DIR.glob("*_seed*.json")):
        row = json.loads(table_path.read_text(encoding="utf-8"))
        if "vindr_bal_raw_auroc" in row:
            print(f"skip {table_path.name} (already has VinDr)")
            continue
        ckpt_path = RUNS_DIR / f"{table_path.stem}.pt"
        if not ckpt_path.is_file():
            print(f"missing checkpoint {ckpt_path.name}")
            continue
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        task = payload["task"]
        head = payload["head"]
        backbone = payload.get("backbone", cfg["backbone"])
        train_df = pd.read_csv(SPLITS_DIR / task / "chexpert_train.csv")
        feat_dim = int(features_for_frame(backbone, train_df.iloc[:1]).shape[1])
        model = QCalibNet(
            feat_dim=feat_dim,
            head=head,
            n_qubits=int(cfg["n_qubits"]),
            n_layers=int(cfg["n_layers"]),
            mlp_hidden=int(cfg["mlp_hidden"]),
            mlp_dropout=float(cfg["mlp_dropout"]),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        device = torch.device("cpu")
        temperature = float(payload["temperature"])
        bal = eval_csv(model, backbone, SPLITS_DIR / task / "vindr_external_test.csv", cfg, device, temperature)
        nat = eval_csv(
            model, backbone, SPLITS_DIR / task / "vindr_external_test_natural.csv", cfg, device, temperature
        )
        if bal:
            row.update(flatten("vindr_bal_raw", bal["raw"]))
            row.update(flatten("vindr_bal_cal", bal["calibrated"]))
        if nat:
            row.update(flatten("vindr_nat_raw", nat["raw"]))
            row.update(flatten("vindr_nat_cal", nat["calibrated"]))
        table_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(f"updated {table_path.name}  vindr_bal_auroc={row.get('vindr_bal_raw_auroc')}")
        n_ok += 1
    print(f"backfilled {n_ok} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
