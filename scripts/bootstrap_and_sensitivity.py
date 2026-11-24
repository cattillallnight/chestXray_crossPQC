"""Bootstrap AUROC/ECE and ECE bin-width sensitivity from dumped logits.

Does not retrain. Degenerate-T ablation is computed from all_runs.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qcalibnet.metrics import expected_calibration_error, sigmoid
from qcalibnet.paths import FIGURES_DIR, TABLES_DIR, ensure_dirs

PRIMARY = ["no_finding", "cardiomegaly", "pleural_effusion", "consolidation"]
RNG = np.random.default_rng(42)


def degenerate_T(t: float) -> bool:
    return (t <= 1.001e-3) or (t >= 99.0)


def bootstrap_metrics(logits: np.ndarray, labels: np.ndarray, temperature: float, n_boot: int = 300, n_bins: int = 15):
    n = len(labels)
    auroc = np.empty(n_boot)
    ece_raw = np.empty(n_boot)
    ece_cal = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        y = labels[idx]
        z = logits[idx]
        if np.unique(y).size < 2:
            auroc[b] = np.nan
        else:
            auroc[b] = roc_auc_score(y, sigmoid(z))
        ece_raw[b], _ = expected_calibration_error(sigmoid(z), y, n_bins=n_bins)
        ece_cal[b], _ = expected_calibration_error(sigmoid(z / temperature), y, n_bins=n_bins)
    def ci(arr):
        arr = arr[~np.isnan(arr)]
        return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    return {
        "auroc_mean": ci(auroc)[0],
        "auroc_lo": ci(auroc)[1],
        "auroc_hi": ci(auroc)[2],
        "ece_raw_mean": ci(ece_raw)[0],
        "ece_raw_lo": ci(ece_raw)[1],
        "ece_raw_hi": ci(ece_raw)[2],
        "ece_cal_mean": ci(ece_cal)[0],
        "ece_cal_lo": ci(ece_cal)[1],
        "ece_cal_hi": ci(ece_cal)[2],
        "n_boot": n_boot,
        "n": int(n),
    }


def ece_by_bins(logits, labels, temperature, bins=(10, 15, 20)):
    row = {}
    for n_bins in bins:
        raw, _ = expected_calibration_error(sigmoid(logits), labels, n_bins=n_bins)
        cal, _ = expected_calibration_error(sigmoid(logits / temperature), labels, n_bins=n_bins)
        row[f"ece_raw_{n_bins}"] = raw
        row[f"ece_cal_{n_bins}"] = cal
    return row


def ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["T_degenerate"] = df["temperature"].map(degenerate_T)
    rows = []
    for task in PRIMARY:
        for head in ("mlp", "product_pqc", "entangling_pqc"):
            sub = df[(df["task"] == task) & (df["head"] == head)]
            keep = sub[~sub["T_degenerate"]]
            rows.append(
                {
                    "task": task,
                    "head": head,
                    "n_all": int(len(sub)),
                    "n_kept": int(len(keep)),
                    "n_dropped": int(len(sub) - len(keep)),
                    "chexpert_auroc_all": sub["chexpert_test_raw_auroc"].mean(),
                    "chexpert_auroc_kept": keep["chexpert_test_raw_auroc"].mean() if len(keep) else np.nan,
                    "chexpert_ece_all": sub["chexpert_test_raw_ece"].mean(),
                    "chexpert_ece_kept": keep["chexpert_test_raw_ece"].mean() if len(keep) else np.nan,
                    "vindr_auroc_all": sub["vindr_bal_raw_auroc"].mean(),
                    "vindr_auroc_kept": keep["vindr_bal_raw_auroc"].mean() if len(keep) else np.nan,
                    "vindr_ece_all": sub["vindr_bal_raw_ece"].mean(),
                    "vindr_ece_kept": keep["vindr_bal_raw_ece"].mean() if len(keep) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def paired_summary(df: pd.DataFrame) -> dict:
    paired = []
    for task in PRIMARY:
        for seed in (42, 43, 44):
            mlp = df[(df["task"] == task) & (df["head"] == "mlp") & (df["seed"] == seed)]
            if mlp.empty:
                continue
            m = mlp.iloc[0]
            for head in ("product_pqc", "entangling_pqc"):
                o = df[(df["task"] == task) & (df["head"] == head) & (df["seed"] == seed)]
                if o.empty:
                    continue
                r = o.iloc[0]
                paired.append(
                    {
                        "head": head,
                        "d_auroc_src": r.chexpert_test_raw_auroc - m.chexpert_test_raw_auroc,
                        "d_ece_src": r.chexpert_test_raw_ece - m.chexpert_test_raw_ece,
                        "d_auroc_tgt": r.vindr_bal_raw_auroc - m.vindr_bal_raw_auroc,
                        "d_ece_tgt": r.vindr_bal_raw_ece - m.vindr_bal_raw_ece,
                    }
                )
    p = pd.DataFrame(paired)
    out = {}
    for head, g in p.groupby("head"):
        out[head] = {
            "n_pairs": int(len(g)),
            "mean_d_auroc_src": float(g.d_auroc_src.mean()),
            "std_d_auroc_src": float(g.d_auroc_src.std(ddof=1)),
            "mean_d_ece_src": float(g.d_ece_src.mean()),
            "std_d_ece_src": float(g.d_ece_src.std(ddof=1)),
            "mean_d_auroc_tgt": float(g.d_auroc_tgt.mean()),
            "std_d_auroc_tgt": float(g.d_auroc_tgt.std(ddof=1)),
            "mean_d_ece_tgt": float(g.d_ece_tgt.mean()),
            "std_d_ece_tgt": float(g.d_ece_tgt.std(ddof=1)),
            "n_auroc_src_worse": int((g.d_auroc_src < 0).sum()),
            "n_ece_src_better": int((g.d_ece_src < 0).sum()),
        }
    return out


def main() -> int:
    ensure_dirs()
    logit_dir = FIGURES_DIR / "logits"
    boot_rows = []
    bin_rows = []
    if logit_dir.is_dir():
        for path in sorted(logit_dir.glob("*.npz")):
            blob = np.load(path, allow_pickle=True)
            logits = blob["logits"].astype(np.float64)
            labels = blob["labels"].astype(np.float64)
            T = float(blob["temperature"][0])
            meta = {
                "file": path.name,
                "task": str(blob["task"]),
                "head": str(blob["head"]),
                "seed": int(blob["seed"]),
                "split": str(blob["split"]),
                "temperature": T,
            }
            boot_rows.append({**meta, **bootstrap_metrics(logits, labels, T)})
            bin_rows.append({**meta, **ece_by_bins(logits, labels, T)})
            print(f"boot {path.name}")
    if boot_rows:
        pd.DataFrame(boot_rows).to_csv(TABLES_DIR / "paper_bootstrap.csv", index=False)
        pd.DataFrame(bin_rows).to_csv(TABLES_DIR / "paper_ece_bins.csv", index=False)

    df = pd.read_csv(TABLES_DIR / "all_runs.csv")
    abl = ablation_table(df)
    abl.to_csv(TABLES_DIR / "paper_ablation_drop_degenerate_T.csv", index=False)
    paired = paired_summary(df)
    (TABLES_DIR / "paper_paired_summary.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    print(json.dumps(paired, indent=2))
    print(f"bootstrap files: {len(boot_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
