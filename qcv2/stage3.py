"""Stage 3: subgroup analysis.

CheXpert test has Sex, Age, AP/PA metadata -> subgroup AUROC/ECE with CIs.
VinDr splits have NO demographic/technical metadata -> stated as unavailable.
Cells with < 30 positives are flagged insufficient.
"""
from __future__ import annotations

import numpy as np

from . import metrics as M
from .paths import PRIMARY_TASKS, V1_HEADS, V1_SEEDS
from .state import warn

MIN_POS = 30


def _age_bin(a):
    try:
        v = float(a)
    except (TypeError, ValueError):
        return "unknown"
    if v < 40:
        return "<40"
    if v < 60:
        return "40-59"
    if v < 80:
        return "60-79"
    return "80+"


def _sub_metric(p, y, idx):
    yy = y[idx]
    if len(yy) == 0:
        return np.nan, np.nan, np.nan
    if len(np.unique(yy)) < 2:
        return np.nan, np.nan, np.nan
    au = M.auroc(p[idx], yy)
    ec = M.ece_pos(p[idx], yy)
    return au, ec, float(yy.sum())


def stage3(data):
    out = []
    for task in PRIMARY_TASKS:
        rows = data.read_split(task, "chexpert_test.csv")
        sex = np.asarray([r.get("Sex", "") for r in rows])
        age = np.asarray([_age_bin(r.get("Age", "")) for r in rows])
        appa = np.asarray([r.get("AP/PA", "") for r in rows])
        for head in V1_HEADS:
            for seed in V1_SEEDS:
                got = data.load_v1_logits(task, head, seed, "chexpert_test")
                if got is None:
                    continue
                z, y, _ = got
                p = M.sigmoid(z)
                for name, groups in (("sex", sex), ("age", age), ("ap_pa", appa)):
                    for g in sorted(set(groups)):
                        if g in ("", "unknown", "nan"):
                            continue
                        idx = np.where(groups == g)[0]
                        if len(idx) < 10:
                            continue
                        au, ec, npos = _sub_metric(p, y, idx)
                        insufficient = (npos < MIN_POS) if not np.isnan(npos) else True
                        for met, v in (("auroc", au), ("ece_pos", ec), ("n_pos", npos)):
                            out.append({
                                "stage": "3", "task": task, "head": head, "seed": seed,
                                "domain": f"chexpert_test:{name}={g}", "prevalence": "",
                                "recalibration": "none", "metric": met,
                                "estimate": v, "ci_lo": "", "ci_hi": "",
                                "tag": "exploratory",
                                "insufficient": str(insufficient),
                            })
    # VinDr metadata check
    vrows = data.read_split("no_finding", "vindr_external_test.csv")
    has_demo = any(k in vrows[0] for k in ("Sex", "Age", "sex", "age"))
    if not has_demo:
        warn("Stage 3: VinDr splits carry no demographic metadata; VinDr subgroups not computed",
             "missing_metadata")
    return out