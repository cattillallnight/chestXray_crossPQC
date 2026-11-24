"""Stage 1: post-hoc analyses on existing v1 logits (no retraining).

1.1 prevalence sweep (weighted ECE/Brier/NLL) + null predictor
1.2 recalibration ladder (source-fit rungs + oracle upper bound)
1.3 metric battery (3 ECE variants, bin modes, Murphy decomposition, AUROC, AUPRC, MCE)
1.4 bootstrap CIs (patient-level CheXpert, image-level VinDr), paired vs MLP
1.5 operating point at sens=0.90 on CheXpert val -> VinDr metrics + decision curves

All rows carry tag='exploratory' (v1 runs already inspected).
"""
from __future__ import annotations

import numpy as np

from . import metrics as M
from .guard import assert_not_vindr
from .paths import PRIMARY_TASKS, V1_HEADS, V1_SEEDS
from .recalibration import build_rung, Temperature
from .state import log, warn, append_csv, now
from .paths import CSV, TEX, V2

PREVALENCES = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70]


def _rows_for(task, head, seed, split, data):
    got = data.load_v1_logits(task, head, seed, split)
    if got is None:
        return None
    return got


def stage1_1(data, cache: dict):
    """Prevalence sweep on VinDr balanced logits via class-conditional reweighting."""
    out = []
    for task in PRIMARY_TASKS:
        for head in V1_HEADS:
            for seed in V1_SEEDS:
                got = data.load_v1_logits(task, head, seed, "vindr_bal")
                if got is None:
                    warn(f"missing vindr_bal logits {task}/{head}/{seed}", "missing_input")
                    continue
                z, y, T = got
                p = M.sigmoid(z)
                prior_chex = 0.5
                for pi in PREVALENCES:
                    w = M.weighted_prevalence_weights(y, pi)
                    for met, val in (
                        ("ece_pos", M.ece_pos(p, y, weights=w)),
                        ("ece_top", M.ece_top(p, y, weights=w)),
                        ("brier", M.brier(p, y, weights=w)),
                        ("nll", M.nll(p, y, weights=w)),
                    ):
                        out.append({
                            "stage": "1.1", "task": task, "head": head, "seed": seed,
                            "domain": "vindr_bal", "prevalence": pi, "recalibration": "none",
                            "metric": met, "estimate": val, "ci_lo": "", "ci_hi": "",
                            "tag": "exploratory",
                        })
                # null predictor: constant CheXpert-train prior
                p_null = np.full_like(p, prior_chex)
                for pi in PREVALENCES:
                    w = M.weighted_prevalence_weights(y, pi)
                    for met, val in (
                        ("ece_pos", M.ece_pos(p_null, y, weights=w)),
                        ("ece_top", M.ece_top(p_null, y, weights=w)),
                        ("brier", M.brier(p_null, y, weights=w)),
                        ("nll", M.nll(p_null, y, weights=w)),
                    ):
                        out.append({
                            "stage": "1.1", "task": task, "head": "null_prior_0.5",
                            "seed": seed, "domain": "vindr_bal", "prevalence": pi,
                            "recalibration": "none", "metric": met, "estimate": val,
                            "ci_lo": "", "ci_hi": "", "tag": "exploratory",
                        })
    return out


def _fit_rung_on_chexpert(data, task, head, seed, rung_name):
    """Fit a rung on CheXpert val logits (source only). Returns (callable, info)."""
    got = data.load_v1_logits(task, head, seed, "chexpert_val")
    if got is None:
        # v1 did not dump chexpert_val logits; cannot fit post-hoc on val
        return None, {"reason": "no chexpert_val logits in v1"}
    z, y, _ = got
    rung = build_rung(rung_name)
    if rung_name in ("bbse", "saerens_em"):
        # needs unlabelled target logits; supplied by caller
        return None, {"reason": "needs target logits; handled by caller"}
    fn = rung.fit(z, y, origin="chexpert_val")
    info = {}
    if isinstance(rung, Temperature):
        info = {"T": rung.value, "at_bound": rung.at_bound}
    return fn, info


def stage1_2(data, cache: dict):
    """Recalibration ladder. Fitted on CheXpert val; evaluated on target.

    Because v1 did not dump chexpert_val logits, source-fitted rungs that need
    val logits cannot be fit for all runs. We detect and log this.
    """
    out = []
    missing = 0
    for task in PRIMARY_TASKS:
        for head in V1_HEADS:
            for seed in V1_SEEDS:
                got_bal = data.load_v1_logits(task, head, seed, "vindr_bal")
                if got_bal is None:
                    continue
                z_bal, y_bal, T_v1 = got_bal
                for domain, (z, y) in (("vindr_bal", (z_bal, y_bal)),):
                    for rung_name in ("none", "temperature", "platt"):
                        fn, info = _fit_rung_on_chexpert(data, task, head, seed, rung_name)
                        if rung_name == "none":
                            fn = lambda zz: M.sigmoid(zz)
                        if fn is None:
                            missing += 1
                            continue
                        p = fn(z)
                        for met, val in (
                            ("ece_pos", M.ece_pos(p, y)),
                            ("ece_top", M.ece_top(p, y)),
                        ):
                            out.append({
                                "stage": "1.2", "task": task, "head": head, "seed": seed,
                                "domain": domain, "prevalence": "", "recalibration": rung_name,
                                "metric": met, "estimate": val, "ci_lo": "", "ci_hi": "",
                                "tag": "exploratory",
                            })
    if missing:
        warn(f"Stage 1.2: {missing} rung fits skipped (no chexpert_val logits in v1)", "missing_input")
    return out


def stage1_3(data, cache: dict):
    """Metric battery on both domains."""
    out = []
    for task in PRIMARY_TASKS:
        for head in V1_HEADS:
            for seed in V1_SEEDS:
                for domain, split in (("chexpert_test", "chexpert_test"),
                                      ("vindr_bal", "vindr_bal")):
                    got = data.load_v1_logits(task, head, seed, split)
                    if got is None:
                        continue
                    z, y, _ = got
                    p = M.sigmoid(z)
                    mur = M.murphy_brier(p, y)
                    vals = {
                        "auroc": M.auroc(p, y), "auprc": M.auprc(p, y),
                        "brier": M.brier(p, y), "mce": M.mce(p, y),
                        "ece_pos": M.ece_pos(p, y), "ece_top": M.ece_top(p, y),
                        "smece": M.smece(p, y),
                        "ece_pos_eqm": M.ece_pos(p, y, mode="mass"),
                        "ece_top_eqm": M.ece_top(p, y, mode="mass"),
                        "ece_pos_b10": M.ece_pos(p, y, n_bins=10),
                        "ece_pos_b20": M.ece_pos(p, y, n_bins=20),
                        "murphy_reliability": mur["reliability"],
                        "murphy_resolution": mur["resolution"],
                        "murphy_uncertainty": mur["uncertainty"],
                    }
                    for met, val in vals.items():
                        out.append({
                            "stage": "1.3", "task": task, "head": head, "seed": seed,
                            "domain": domain, "prevalence": "", "recalibration": "none",
                            "metric": met, "estimate": val, "ci_lo": "", "ci_hi": "",
                            "tag": "exploratory",
                        })
    return out


def run_stage1(data) -> dict:
    log("Stage 1: post-hoc analyses on v1 logits")
    check = data.available_v1_logits()
    have = sum(1 for c in check if c["exists"])
    log(f"  v1 logit files present: {have}/{len(check)}")
    # detect chexpert_val logs
    from .paths import LOGITS_V1
    n_val = (len(list(LOGITS_V1.glob("*_chexpert_val.npz")))
             + len(list((V2 / "logits_v1_ext").glob("*_chexpert_val.npz"))))
    log(f"  chexpert_val logits present: {n_val}")
    cache = {}
    rows = []
    rows += stage1_1(data, cache)
    rows += stage1_2(data, cache)
    rows += stage1_3(data, cache)
    return {"rows": rows, "n_val_logits": n_val}