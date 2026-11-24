"""Train one matched-capacity head on cached CheXpert features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qcalibnet.config import load_config
from qcalibnet.engine import evaluate_split, fit_val_temperature, save_run, train_head
from qcalibnet.features import features_for_frame
from qcalibnet.paths import RUNS_DIR, SPLITS_DIR, TABLES_DIR, ensure_dirs
from qcalibnet.seed import seed_everything


def load_split(task: str, name: str) -> pd.DataFrame:
    path = SPLITS_DIR / task / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def maybe_eval(model, backbone, frame_path: Path, cfg, device, temperature):
    if not frame_path.is_file():
        return None
    frame = pd.read_csv(frame_path)
    x = features_for_frame(backbone, frame)
    y = frame["label"].to_numpy()
    return evaluate_split(model, x, y, cfg, device, temperature=temperature)


def flatten(prefix: str, metrics: dict | None) -> dict:
    if metrics is None:
        return {}
    return {f"{prefix}_{k}": v for k, v in metrics.items() if k not in {"logits", "labels"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--task", default="no_finding")
    parser.add_argument("--head", default="mlp", choices=["mlp", "product_pqc", "entangling_pqc"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs (use 1 for smoke).")
    parser.add_argument("--smoke", action="store_true", help="1 epoch, no early-stop wait.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg["epochs"] = 1
        cfg["patience"] = 1
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    seed_everything(seed)
    ensure_dirs()
    backbone = cfg["backbone"]
    task = args.task
    head = args.head

    train_df = load_split(task, "chexpert_train.csv")
    val_df = load_split(task, "chexpert_val.csv")
    test_df = load_split(task, "chexpert_test.csv")
    train_x = features_for_frame(backbone, train_df)
    val_x = features_for_frame(backbone, val_df)
    test_x = features_for_frame(backbone, test_df)
    feat_dim = int(train_x.shape[1])

    result = train_head(
        feat_dim=feat_dim,
        head=head,
        train_x=train_x,
        train_y=train_df["label"].to_numpy(),
        val_x=val_x,
        val_y=val_df["label"].to_numpy(),
        cfg=cfg,
        seed=seed,
    )
    model = result["model"]
    device = result["device"]
    temperature = fit_val_temperature(model, val_x, val_df["label"].to_numpy(), cfg, device)

    val_eval = evaluate_split(model, val_x, val_df["label"].to_numpy(), cfg, device, temperature)
    test_eval = evaluate_split(model, test_x, test_df["label"].to_numpy(), cfg, device, temperature)

    run_id = f"{task}_{head}_seed{seed}"
    ckpt = RUNS_DIR / f"{run_id}.pt"
    save_run(
        ckpt,
        {
            "task": task,
            "head": head,
            "seed": seed,
            "backbone": backbone,
            "n_trainable": result["n_trainable"],
            "temperature": temperature,
            "history": result["history"],
            "state_dict": model.state_dict(),
        },
    )

    row = {
        "task": task,
        "head": head,
        "seed": seed,
        "backbone": backbone,
        "n_trainable": result["n_trainable"],
        "best_val_auroc": result["best_val_auroc"],
        "temperature": temperature,
        "epochs_run": len(result["history"]),
    }
    row.update(flatten("chexpert_val_raw", val_eval["raw"]))
    row.update(flatten("chexpert_val_cal", val_eval["calibrated"]))
    row.update(flatten("chexpert_test_raw", test_eval["raw"]))
    row.update(flatten("chexpert_test_cal", test_eval["calibrated"]))

    table_path = TABLES_DIR / f"{run_id}.json"
    table_path.write_text(json.dumps(row, indent=2), encoding="utf-8")

    vindr_bal = None
    vindr_nat = None
    try:
        vindr_bal = maybe_eval(model, backbone, SPLITS_DIR / task / "vindr_external_test.csv", cfg, device, temperature)
        vindr_nat = maybe_eval(
            model, backbone, SPLITS_DIR / task / "vindr_external_test_natural.csv", cfg, device, temperature
        )
    except Exception as exc:
        print(f"VinDr eval failed after CheXpert metrics were saved: {exc}")
    if vindr_bal:
        row.update(flatten("vindr_bal_raw", vindr_bal["raw"]))
        row.update(flatten("vindr_bal_cal", vindr_bal["calibrated"]))
    if vindr_nat:
        row.update(flatten("vindr_nat_raw", vindr_nat["raw"]))
        row.update(flatten("vindr_nat_cal", vindr_nat["calibrated"]))
    table_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps(row, indent=2))
    print(f"checkpoint: {ckpt}")
    print(f"metrics: {table_path}")
    if vindr_bal is None:
        print("VinDr external test skipped or failed.")


if __name__ == "__main__":
    main()
