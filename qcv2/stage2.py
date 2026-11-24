"""Stage 2: corrected-protocol training runs + downstream analyses.

One unit of work = one config (task, head, seed, ...). Resumable by config hash.
Training never sees VinDr; VinDr is evaluated once per final config and logged
to target_eval_log.csv.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import metrics as M
from .configs import cfg_hash
from .paths import CSV, PRIMARY_TASKS, SPLITS, V2
from .stage2_data import fit_bottleneck, make_split
from .state import (RunStore, append_csv, git_hash, log, warn, write_csv, now,
                    deviation)
from .train import predict, train_head

_BN = {}


def _bottleneck(task: str):
    if task not in _BN:
        _BN[task] = fit_bottleneck(task)
    return _BN[task]


def _features_meta(task):
    bn = _bottleneck(task)
    return bn


def run_one(cfg: dict, store: RunStore) -> dict | None:
    if store.is_done(cfg):
        return None
    t0 = time.time()
    task = cfg["task"]
    bn = _bottleneck(task)
    ztr, ytr, _ = make_split(task, "chexpert_train.csv", bn)
    zva, yva, _ = make_split(task, "chexpert_val.csv", bn)
    zte, yte, meta_te = make_split(task, "chexpert_test.csv", bn)
    if "n_train" in cfg and cfg["n_train"] < len(ztr):
        rng = np.random.default_rng(cfg["seed"])
        idx = rng.choice(len(ztr), cfg["n_train"], replace=False)
        ztr, ytr = ztr[idx], ytr[idx]
    res = train_head(cfg, ztr, ytr, zva, yva, seed=int(cfg["seed"]))
    head = res["model"]
    logits = {
        "chexpert_test": predict(head, zte),
        "chexpert_val": predict(head, zva),
        "vindr_bal": predict(head, make_split(task, "vindr_external_test.csv", bn)[0]),
        "vindr_nat": predict(head, make_split(task, "vindr_external_test_natural.csv", bn)[0]),
    }
    labels = {
        "chexpert_test": yte, "chexpert_val": yva,
        "vindr_bal": make_split(task, "vindr_external_test.csv", bn)[1],
        "vindr_nat": make_split(task, "vindr_external_test_natural.csv", bn)[1],
    }
    # save logits artifact for this run
    outdir = V2 / "logits_v2"
    outdir.mkdir(parents=True, exist_ok=True)
    rid = f"{cfg['stage']}_{task}_{cfg['head']}_s{cfg['seed']}_{cfg_hash(cfg)}"
    np.savez_compressed(outdir / f"{rid}.npz",
                        **{f"logits_{k}": v.astype(np.float32) for k, v in logits.items()},
                        **{f"labels_{k}": v.astype(np.int8) for k, v in labels.items()},
                        cfg=json.dumps(cfg))
    # collapsed warning
    for split, z in logits.items():
        if float(z.max() - z.min()) < 1e-2:
            warn(f"collapsed run {rid} split={split} range={z.max()-z.min():.2e}", "collapsed")
    # target eval log (once per config)
    append_csv(V2 / "target_eval_log.csv", {
        "timestamp": now(), "config_hash": cfg_hash(cfg), "seed": cfg["seed"],
        "task": task, "head": cfg["head"], "stage": cfg["stage"],
        "git": git_hash(), "n_vindr_bal": len(labels["vindr_bal"]),
        "n_vindr_nat": len(labels["vindr_nat"]),
    })
    store.mark_done(cfg, {"runtime_s": round(time.time() - t0, 1),
                          "n_params": res["n_params"], "epochs_run": res["epochs_run"]})
    return {"rid": rid, "logits": logits, "labels": labels, "cfg": cfg,
            "history": res["history"], "n_params": res["n_params"],
            "epochs_run": res["epochs_run"], "runtime_s": time.time() - t0}


def evaluate_run(cfg, logits, labels) -> list[dict]:
    """Produce long-format metric rows for one trained run."""
    rows = []
    tag = cfg["tag"]
    for domain in ("chexpert_test", "vindr_bal", "vindr_nat"):
        z = logits[domain]; y = labels[domain]
        p = M.sigmoid(z)
        mur = M.murphy_brier(p, y)
        vals = {
            "auroc": M.auroc(p, y), "auprc": M.auprc(p, y), "brier": M.brier(p, y),
            "nll": M.nll(p, y), "mce": M.mce(p, y), "ece_pos": M.ece_pos(p, y),
            "ece_top": M.ece_top(p, y), "smece": M.smece(p, y),
        }
        for met, val in vals.items():
            rows.append({
                "stage": cfg["stage"], "task": cfg["task"], "head": cfg["head"],
                "config": f"{cfg['bottleneck']}|L{cfg['layers']}|{cfg.get('train_prior','balanced')}",
                "seed": cfg["seed"], "domain": domain, "prevalence": "",
                "recalibration": "none", "metric": met, "estimate": val,
                "ci_lo": "", "ci_hi": "", "tag": tag,
            })
    return rows