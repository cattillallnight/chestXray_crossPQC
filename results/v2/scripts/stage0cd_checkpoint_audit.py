"""Stage 0c/0d: audit all 45 checkpoints and recompute metrics from logits.

0c  per checkpoint: head-only parameter count, learned scale/bias (PQC),
    logit min/max/std, probability min/max/std for the PQC heads.
0d  recompute AUROC / ECE(positive-class) / ECE(max-conf) / Brier from
    dumped logits and compare against results/tables/all_runs.csv.
    Write claims_check.csv with status reproduced/mismatch.

Reads existing checkpoints (read-only) and existing logits under
results/figures/logits/. Writes only to results/v2/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "results" / "v2"
RUNS = ROOT / "results" / "runs"
TABLES = ROOT / "results" / "tables"
LOGITS = ROOT / "results" / "figures" / "logits"

TASKS = ["no_finding", "cardiomegaly", "pleural_effusion", "consolidation", "pneumonia_proxy"]
HEADS = ["mlp", "product_pqc", "entangling_pqc"]
SEEDS = [42, 43, 44]
PRIMARY = ["no_finding", "cardiomegaly", "pleural_effusion", "consolidation"]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -80, 80)))


def ece_positive(probs, labels, n_bins=15):
    """ECE as implemented in qcalibnet.metrics: confidence = positive-class prob."""
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (probs >= lo) & (probs <= hi) if i == n_bins - 1 else (probs >= lo) & (probs < hi)
        if m.any():
            e += abs(labels[m].mean() - probs[m].mean()) * m.sum() / n
    return float(e)


def ece_maxconf(probs, labels, n_bins=15):
    """Standard ECE: confidence = max(p, 1-p), correctness = predicted == label."""
    conf = np.maximum(probs, 1 - probs)
    correct = ((probs >= 0.5).astype(float) == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        if m.any():
            e += abs(correct[m].mean() - conf[m].mean()) * m.sum() / n
    return float(e)


def auroc(probs, labels):
    """Mann-Whitney AUROC with midranks for ties (matches sklearn)."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    order = np.argsort(probs, kind="mergesort")
    sp = probs[order]
    ranks = np.empty(len(probs), dtype=float)
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    pos = labels == 1
    n_pos = pos.sum()
    n_neg = len(labels) - n_pos
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def brier(probs, labels):
    return float(np.mean((probs - labels) ** 2))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def head_params(state: dict) -> dict:
    """Separate bottleneck (proj.*) from head parameters."""
    out = {"bottleneck": 0, "head": 0, "total": 0}
    for k, v in state.items():
        n = int(np.prod(v.shape))
        out["total"] += n
        if k.startswith("bottleneck."):
            out["bottleneck"] += n
        else:
            out["head"] += n
    return out


def main() -> None:
    V2.mkdir(parents=True, exist_ok=True)
    ok_rows = []
    audit_rows = []
    split_hashes = {}

    # split hashes (must remain unchanged; recorded for the unit test)
    for task in TASKS:
        for name in ("chexpert_train.csv", "chexpert_val.csv", "chexpert_test.csv",
                     "vindr_external_test.csv", "vindr_external_test_natural.csv"):
            p = ROOT / "data" / "splits" / task / name
            if p.is_file():
                split_hashes[f"{task}/{name}"] = sha256(p)
    (V2 / "split_hashes.json").write_text(json.dumps(split_hashes, indent=2), encoding="utf-8")

    for task in TASKS:
        for head in HEADS:
            for seed in SEEDS:
                run_id = f"{task}_{head}_seed{seed}"
                ck = RUNS / f"{run_id}.pt"
                if not ck.is_file():
                    print(f"MISSING checkpoint {run_id}")
                    continue
                payload = torch.load(ck, map_location="cpu", weights_only=False)
                state = payload["state_dict"]
                sd = {k: np.asarray(v) for k, v in state.items()}
                pc = head_params(sd)
                T = float(payload.get("temperature", 1.0))

                row = {
                    "run_id": run_id, "task": task, "head": head, "seed": seed,
                    "bottleneck_params": pc["bottleneck"], "head_params": pc["head"],
                    "total_params": pc["total"], "temperature": f"{T:.6g}",
                }
                if head != "mlp":
                    scale = float(np.asarray(state["head.scale"]).reshape(-1)[0])
                    bias = float(np.asarray(state["head.bias"]).reshape(-1)[0])
                    row["scale"] = f"{scale:.6g}"
                    row["bias"] = f"{bias:.6g}"
                    row["n_layers"] = int(np.asarray(state["head.weights"]).shape[0])
                    # theta parameters on q0 only vs all
                    wsh = np.asarray(state["head.weights"]).shape
                    row["theta_params_total"] = int(np.prod(wsh))
                    row["theta_params_q0"] = int(wsh[0] * wsh[2])
                else:
                    row["scale"] = ""
                    row["bias"] = ""
                    row["n_layers"] = ""
                    row["theta_params_total"] = ""
                    row["theta_params_q0"] = ""

                # logits from dumped npz (chexpert_test)
                npz = LOGITS / f"{run_id}_chexpert_test.npz"
                if npz.is_file():
                    b = np.load(npz, allow_pickle=True)
                    z = b["logits"].astype(np.float64)
                    p = sigmoid(z)
                    logit_range = float(z.max() - z.min())
                    collapsed = logit_range < 1e-2  # degenerate: no discriminative signal
                    row.update({
                        "logit_min": f"{z.min():.4f}", "logit_max": f"{z.max():.4f}",
                        "logit_std": f"{z.std():.4f}", "logit_range": f"{logit_range:.3e}",
                        "prob_min": f"{p.min():.4f}", "prob_max": f"{p.max():.4f}",
                        "prob_std": f"{p.std():.4f}",
                        "collapsed": str(collapsed),
                    })
                audit_rows.append(row)

    # ---- 0d: recompute and compare with all_runs.csv ----
    allruns = {}
    with (TABLES / "all_runs.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            allruns[f"{r['task']}_{r['head']}_seed{r['seed']}"] = r

    checks = []
    for task in PRIMARY:
        for head in HEADS:
            for seed in SEEDS:
                run_id = f"{task}_{head}_seed{seed}"
                npz = LOGITS / f"{run_id}_chexpert_test.npz"
                if not npz.is_file() or run_id not in allruns:
                    continue
                b = np.load(npz, allow_pickle=True)
                z = b["logits"].astype(np.float64)
                y = b["labels"].astype(np.float64)
                p = sigmoid(z)
                ref = allruns[run_id]
                for met, val, col in (
                    ("AUROC", auroc(p, y), "chexpert_test_raw_auroc"),
                    ("ECE_posclass", ece_positive(p, y), "chexpert_test_raw_ece"),
                    ("Brier", brier(p, y), "chexpert_test_raw_brier"),
                ):
                    paper = float(ref[col])
                    diff = abs(val - paper)
                    checks.append({
                        "run_id": run_id, "metric": met,
                        "paper_value": f"{paper:.6f}", "recomputed": f"{val:.6f}",
                        "abs_diff": f"{diff:.2e}",
                        "status": "reproduced" if diff < 1e-6 else "mismatch",
                    })
                # ECE definition comparison
                checks.append({
                    "run_id": run_id, "metric": "ECE_maxconf_standard",
                    "paper_value": float(ref["chexpert_test_raw_ece"]),
                    "recomputed": ece_maxconf(p, y), "abs_diff": "",
                    "status": "definition differs (positive-class vs max-confidence)",
                })

    def write_csv(path, rows, fields=None):
        if not rows:
            return
        fields = fields or list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    write_csv(V2 / "stage0c_checkpoint_audit.csv", audit_rows)
    write_csv(V2 / "claims_check.csv", checks)

    n_rep = sum(1 for c in checks if c["status"] == "reproduced")
    n_mis = sum(1 for c in checks if c["status"] == "mismatch")
    print(f"checkpoint audit rows: {len(audit_rows)}")
    print(f"claims checks: {len(checks)}  reproduced={n_rep} mismatch={n_mis}")
    print(f"wrote {V2/'stage0c_checkpoint_audit.csv'}")
    print(f"wrote {V2/'claims_check.csv'}")


if __name__ == "__main__":
    main()