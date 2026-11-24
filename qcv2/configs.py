"""Stage 2 config generation and freezing (spec 2.1-2.8).

Configs are generated deterministically, hashed, and (for primary configs)
must exist in a commit BEFORE any run so they can be tagged `confirmatory`.
"""
from __future__ import annotations

import json

from .paths import PRIMARY_TASKS, V2
from .heads import HEADS
from .state import hash_obj

SEEDS = list(range(42, 52))          # 42..51
LAYERS = 2

# Primary heads (fixed, confirmatory). Legacy learned-W MLP kept separately.
PRIMARY_HEADS = [
    "mlp", "mlp_bounded",
    "pqc_none_z0", "pqc_ring_z0", "pqc_none_linear", "pqc_ring_linear",
    "trig_add", "trig_pair", "logistic",
]

COMMON = {
    "n_components": 6,
    "bottleneck": "pca_train_only",
    "layers": LAYERS,
    "embedding": "single_Y",
    "epochs": 100,
    "patience": 12,
    "early_stop_metric": "val_nll",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "pqc_batch_size": 32,
    "train_prior": "balanced",
    "version": "v2.0",
}


def primary_configs() -> list[dict]:
    """The confirmatory grid: task x head x seed. Config is fully specified."""
    out = []
    for task in PRIMARY_TASKS:
        for head in PRIMARY_HEADS:
            for seed in SEEDS:
                cfg = dict(COMMON)
                cfg.update({"stage": "2.3", "task": task, "head": head, "seed": seed,
                            "tag": "confirmatory"})
                out.append(cfg)
    return out


def legacy_configs() -> list[dict]:
    """Legacy learned-W bottleneck, labelled. Exploratory."""
    out = []
    for task in PRIMARY_TASKS:
        for head in ("mlp", "pqc_none_z0", "pqc_ring_z0"):
            for seed in SEEDS:
                cfg = dict(COMMON)
                cfg.update({"bottleneck": "learned_w_legacy", "stage": "2.3",
                            "task": task, "head": head, "seed": seed,
                            "tag": "exploratory"})
                out.append(cfg)
    return out


def prior_ablation_configs() -> list[dict]:
    """Stage 2.6: balanced vs natural training vs balanced+logit correction."""
    out = []
    for task in PRIMARY_TASKS:
        for head in ("mlp", "pqc_ring_z0", "pqc_ring_linear", "trig_pair"):
            for seed in SEEDS:
                for mode in ("natural", "balanced_logitcorr"):
                    cfg = dict(COMMON)
                    cfg.update({"stage": "2.6", "task": task, "head": head,
                                "seed": seed, "train_prior": mode, "tag": "exploratory"})
                    out.append(cfg)
    return out


def depth_ntrain_configs() -> list[dict]:
    """Stage 2.7: re-uploading depth L in {1,2,3}; n_train in {64,256,1000,2000}."""
    out = []
    for task in PRIMARY_TASKS:
        for head in ("mlp", "pqc_ring_z0", "pqc_ring_linear"):
            for seed in SEEDS:
                for L in (1, 2, 3):
                    for ntr in (64, 256, 1000, 2000):
                        cfg = dict(COMMON)
                        cfg.update({"stage": "2.7", "task": task, "head": head,
                                    "seed": seed, "layers": L, "n_train": ntr,
                                    "tag": "exploratory"})
                        out.append(cfg)
    return out


def all_configs() -> dict:
    prim = primary_configs()
    return {
        "primary": prim,
        "legacy": legacy_configs(),
        "prior_ablation": prior_ablation_configs(),
        "depth_ntrain": depth_ntrain_configs(),
    }


def cfg_hash(cfg: dict) -> str:
    return hash_obj(cfg)


def write_frozen_configs() -> dict:
    V2.mkdir(parents=True, exist_ok=True)
    groups = all_configs()
    payload = {}
    for name, cfgs in groups.items():
        payload[name] = [{"cfg": c, "hash": cfg_hash(c)} for c in cfgs]
    (V2 / "frozen_configs.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    counts = {k: len(v) for k, v in groups.items()}
    return counts