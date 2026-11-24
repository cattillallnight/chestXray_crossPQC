"""Stage 1.4 (bootstrap CIs, paired diffs) and 1.5 (operating point, decision curves).

Uses v1 logits (chexpert_test for patient-level bootstrap, vindr_bal for
image-level). Patient IDs come from the CheXpert split CSV.
"""
from __future__ import annotations

import numpy as np

from . import metrics as M
from .paths import PRIMARY_TASKS, V1_HEADS, V1_SEEDS
from .state import warn

N_BOOT = 2000


def _bootstrap_metric(probs, labels, groups, metric_fn, n_boot=N_BOOT, seed=0):
    """Group (patient) bootstrap: resample groups with replacement."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        y = labels[idx]
        if len(np.unique(y)) < 2:
            continue
        vals.append(metric_fn(probs[idx], y))
    if not vals:
        return np.nan, np.nan, np.nan
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def stage1_4(data):
    """Paired bootstrap head - MLP with 95% CI. Descriptive only, no p-values."""
    out = []
    for task in PRIMARY_TASKS:
        # patient ids for chexpert_test
        rows = data.read_split(task, "chexpert_test.csv")
        pid = np.asarray([r.get("patient_id", str(i)) for i, r in enumerate(rows)])
        for seed in V1_SEEDS:
            mlp = data.load_v1_logits(task, "mlp", seed, "chexpert_test")
            if mlp is None:
                continue
            for head in ("product_pqc", "entangling_pqc"):
                got = data.load_v1_logits(task, head, seed, "chexpert_test")
                if got is None:
                    continue
                z_m, y, _ = mlp
                z_h, _, _ = got
                p_m, p_h = M.sigmoid(z_m), M.sigmoid(z_h)
                rng = np.random.default_rng(1000 + seed)
                uniq = np.unique(pid)
                idx_by = {g: np.where(pid == g)[0] for g in uniq}
                d_auroc, d_ece = [], []
                for _ in range(N_BOOT):
                    pick = rng.choice(uniq, len(uniq), replace=True)
                    idx = np.concatenate([idx_by[g] for g in pick])
                    yy = y[idx]
                    if len(np.unique(yy)) < 2:
                        continue
                    d_auroc.append(M.auroc(p_h[idx], yy) - M.auroc(p_m[idx], yy))
                    d_ece.append(M.ece_pos(p_h[idx], yy) - M.ece_pos(p_m[idx], yy))
                for met, arr in (("d_auroc", d_auroc), ("d_ece", d_ece)):
                    if arr:
                        out.append({
                            "stage": "1.4", "task": task, "head": head, "seed": seed,
                            "domain": "chexpert_test", "prevalence": "",
                            "recalibration": "none", "metric": met,
                            "estimate": float(np.mean(arr)),
                            "ci_lo": float(np.percentile(arr, 2.5)),
                            "ci_hi": float(np.percentile(arr, 97.5)),
                            "tag": "exploratory",
                        })
    return out


def _net_benefit(p, y, thr):
    pred = p >= thr
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
    n = len(y)
    if n == 0:
        return 0.0
    return tp / n - (fp / n) * (thr / (1 - thr))


def stage1_5(data):
    """Threshold at sens=0.90 on CheXpert val -> VinDr metrics; decision curves."""
    out = []
    thr_grid = np.round(np.arange(0.05, 0.51, 0.05), 2)
    for task in PRIMARY_TASKS:
        for head in V1_HEADS:
            for seed in V1_SEEDS:
                val = data.load_v1_logits(task, head, seed, "chexpert_val")
                bal = data.load_v1_logits(task, head, seed, "vindr_bal")
                if val is None or bal is None:
                    continue
                zv, yv, _ = val
                zb, yb, _ = bal
                pv, pb = M.sigmoid(zv), M.sigmoid(zb)
                # find threshold on val for sens>=0.9
                order = np.argsort(-pv)
                tys = yv[order]
                tp_cum = np.cumsum(tys)
                n_pos = tys.sum()
                if n_pos == 0:
                    continue
                k = int(np.searchsorted(tp_cum, 0.9 * n_pos))
                k = min(k, len(pv) - 1)
                thr = pv[order][k]
                pred = pb >= thr
                tp = int(np.sum(pred & (yb == 1)))
                fp = int(np.sum(pred & (yb == 0)))
                fn = int(np.sum((~pred) & (yb == 1)))
                tn = int(np.sum((~pred) & (yb == 0)))
                sens = tp / max(tp + fn, 1)
                spec = tn / max(tn + fp, 1)
                ppv = tp / max(tp + fp, 1)
                for met, v in (("op_sens", sens), ("op_spec", spec), ("op_ppv", ppv),
                               ("op_threshold", float(thr))):
                    out.append({
                        "stage": "1.5", "task": task, "head": head, "seed": seed,
                        "domain": "vindr_bal", "prevalence": "", "recalibration": "none",
                        "metric": met, "estimate": v, "ci_lo": "", "ci_hi": "",
                        "tag": "exploratory",
                    })
                # decision curve (pooled over seeds handled by caller aggregation)
                for t in thr_grid:
                    out.append({
                        "stage": "1.5", "task": task, "head": head, "seed": seed,
                        "domain": "vindr_bal", "prevalence": str(t),
                        "recalibration": "none", "metric": "net_benefit",
                        "estimate": _net_benefit(pb, yb, t), "ci_lo": "", "ci_hi": "",
                        "tag": "exploratory",
                    })
    return out