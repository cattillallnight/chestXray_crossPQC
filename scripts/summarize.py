"""Aggregate results/tables/*.json into a paper-facing summary.

Does not retrain. VinDr columns appear only if those metrics were written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qcalibnet.paths import TABLES_DIR, ensure_dirs

TASKS = [
    "no_finding",
    "cardiomegaly",
    "pleural_effusion",
    "consolidation",
    "pneumonia_proxy",
]
HEADS = ["mlp", "product_pqc", "entangling_pqc"]
SEEDS = [42, 43, 44]

METRIC_COLS = [
    "n_trainable",
    "best_val_auroc",
    "temperature",
    "epochs_run",
    "chexpert_test_raw_auroc",
    "chexpert_test_raw_auprc",
    "chexpert_test_raw_f1",
    "chexpert_test_raw_sensitivity",
    "chexpert_test_raw_specificity",
    "chexpert_test_raw_brier",
    "chexpert_test_raw_ece",
    "chexpert_test_raw_mce",
    "chexpert_test_cal_auroc",
    "chexpert_test_cal_brier",
    "chexpert_test_cal_ece",
    "chexpert_test_cal_mce",
    "vindr_bal_raw_auroc",
    "vindr_bal_raw_ece",
    "vindr_bal_raw_brier",
    "vindr_bal_cal_ece",
    "vindr_nat_raw_auroc",
    "vindr_nat_raw_ece",
    "vindr_nat_raw_brier",
    "vindr_nat_cal_ece",
]


def load_rows() -> pd.DataFrame:
    rows = []
    for path in sorted(TABLES_DIR.glob("*.json")):
        if path.name in {"summary.json", "grid_status.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["result_file"] = path.name
        rows.append(payload)
    return pd.DataFrame(rows)


def expected_jobs() -> pd.DataFrame:
    return pd.DataFrame(
        [{"task": t, "head": h, "seed": s} for t in TASKS for h in HEADS for s in SEEDS]
    )


def mean_std(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    present = [c for c in cols if c in frame.columns]
    if not present or frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(["task", "head"], dropna=False)
    out = grouped[present].agg(["mean", "std", "count"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def main() -> int:
    ensure_dirs()
    expected = expected_jobs()
    raw = load_rows()
    if raw.empty:
        print("No result JSON files in results/tables")
        return 1

    key = ["task", "head", "seed"]
    for col in key:
        if col not in raw.columns:
            print(f"results missing column {col}")
            return 1

    merged = expected.merge(raw, on=key, how="left", indicator=True)
    done = merged[merged["_merge"] == "both"].drop(columns="_merge")
    missing = merged[merged["_merge"] == "left_only"][key].reset_index(drop=True)
    extra = raw.merge(expected, on=key, how="left", indicator=True)
    extra = extra[extra["_merge"] == "left_only"]

    status = {
        "n_expected": int(len(expected)),
        "n_done": int(len(done)),
        "n_missing": int(len(missing)),
        "missing": missing.to_dict(orient="records"),
        "has_vindr": bool(
            any(c.startswith("vindr_") and done[c].notna().any() for c in done.columns if c.startswith("vindr_"))
        )
        if not done.empty
        else False,
        "note": (
            "Grid is 5 tasks × 3 heads × 3 seeds. VinDr columns are copied from "
            "each results/tables/*_seed*.json when present; they are not imputed. "
            "pneumonia_proxy is ontology_shift and belongs in the appendix."
        ),
    }

    summary = mean_std(done, METRIC_COLS)
    TABLES_DIR.joinpath("all_runs.csv").write_text(done.to_csv(index=False), encoding="utf-8")
    TABLES_DIR.joinpath("missing_runs.csv").write_text(missing.to_csv(index=False), encoding="utf-8")
    if not summary.empty:
        TABLES_DIR.joinpath("summary_mean_std.csv").write_text(summary.to_csv(index=False), encoding="utf-8")
    TABLES_DIR.joinpath("grid_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    TABLES_DIR.joinpath("summary.json").write_text(
        json.dumps(
            {
                "status": status,
                "mean_std": summary.to_dict(orient="records") if not summary.empty else [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"done {status['n_done']}/{status['n_expected']}  missing {status['n_missing']}")
    print(f"VinDr metrics present: {status['has_vindr']}")
    if not missing.empty:
        print("missing runs:")
        print(missing.to_string(index=False))
    if not summary.empty:
        show = [c for c in summary.columns if c in {"task", "head"} or "auroc" in c or "ece" in c or "brier" in c]
        print(summary[show].to_string(index=False))
    print(f"wrote {TABLES_DIR / 'summary.json'}")
    return 0 if missing.empty else 2


if __name__ == "__main__":
    raise SystemExit(main())
