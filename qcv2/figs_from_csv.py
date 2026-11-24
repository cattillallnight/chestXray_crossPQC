"""Build F3 and F4 from saved CSVs (Stage 1/2 outputs)."""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from .figures import fig_F3_logit_range, fig_F4_decision
from .paths import CSV, V2


def build_F3():
    """Join Stage 0 audit (logit range) with CheXpert ECE per run."""
    audit = list(csv.DictReader((V2 / "stage0c_checkpoint_audit.csv").open(encoding="utf-8")))
    ece = {}
    s1 = CSV / "stage1_long.csv"
    if s1.exists():
        for r in csv.DictReader(s1.open(encoding="utf-8")):
            if r["stage"] == "1.3" and r["metric"] == "ece_pos" and r["domain"] == "chexpert_test":
                ece[(r["task"], r["head"], r["seed"])] = float(r["estimate"])
    rows = []
    for a in audit:
        key = (a["task"], a["head"], a["seed"])
        if key in ece:
            rows.append({"head": a["head"], "logit_range": a["logit_range"],
                         "ece": ece[key]})
    fig_F3_logit_range(rows)
    return len(rows)


def build_F4():
    """Decision curves from Stage 1.5 net-benefit rows (pool across seeds)."""
    s1 = CSV / "stage1_long.csv"
    if not s1.exists():
        return 0
    rows = list(csv.DictReader(s1.open(encoding="utf-8")))
    nb = [r for r in rows if r["stage"] == "1.5" and r["metric"] == "net_benefit"]
    if not nb:
        return 0
    pooled = defaultdict(list)
    for r in nb:
        pooled[float(r["prevalence"])].append(float(r["estimate"]))
    out = [{"head": "pooled", "kind": "raw", "threshold": t, "net_benefit": np.mean(v)}
           for t, v in sorted(pooled.items())]
    fig_F4_decision(out)
    return len(out)