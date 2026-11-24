"""Colorblind-safe vector-PDF figures F1-F4 + reliability diagrams.

Axis labels and legends only; no interpretive prose. Reads the long-format
CSVs written by run_all.py.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .paths import CSV, FIG

# Okabe-Ito colorblind-safe palette
CB = {
    "mlp": "#0072B2", "mlp_bounded": "#56B4E9",
    "pqc_none_z0": "#009E73", "pqc_ring_z0": "#004D40",
    "pqc_none_linear": "#E69F00", "pqc_ring_linear": "#D55E00",
    "trig_add": "#CC79A7", "trig_pair": "#F0E442", "logistic": "#999999",
    "null": "#000000",
}
MARK = {"mlp": "o", "pqc_none_z0": "s", "pqc_ring_z0": "^",
        "pqc_none_linear": "D", "pqc_ring_linear": "v",
        "trig_add": "P", "trig_pair": "X", "logistic": "*"}


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150,
    })


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_F1_prev_sweep(rows):
    """ECE vs prevalence per head with the null predictor."""
    _style()
    data = [r for r in rows if r["stage"] == "1.1" and r["metric"] == "ece_pos"]
    tasks = sorted(set(r["task"] for r in data))
    fig, axes = plt.subplots(1, len(tasks), figsize=(2.4 * len(tasks), 2.3), sharey=True)
    if len(tasks) == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        sub = [r for r in data if r["task"] == task]
        for head in sorted(set(r["head"] for r in sub)):
            hs = [r for r in sub if r["head"] == head]
            prevs = sorted(set(float(r["prevalence"]) for r in hs))
            means = []
            for pi in prevs:
                vals = [float(r["estimate"]) for r in hs if float(r["prevalence"]) == pi]
                means.append(np.mean(vals))
            col = CB.get("null" if head.startswith("null") else head, "#444")
            ls = "--" if head.startswith("null") else "-"
            mk = None if head.startswith("null") else MARK.get(head, "o")
            ax.plot(prevs, means, color=col, ls=ls, marker=mk, ms=3, lw=1.1,
                    label="null (prior)" if head.startswith("null") else head)
        ax.set_title(task, fontsize=7.5)
        ax.set_xlabel("target prevalence", fontsize=7)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel("ECE (positive-class)", fontsize=7)
    axes[-1].legend(frameon=False, fontsize=5.5, loc="upper right")
    _save(fig, "F1_ece_vs_prevalence")


def fig_F2_recal_ladder(rows):
    """ECE by recalibration rung."""
    _style()
    data = [r for r in rows if r["stage"] == "1.2" and r["metric"] == "ece_pos"]
    if not data:
        return
    rungs = ["none", "temperature", "platt"]
    heads = sorted(set(r["head"] for r in data))
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    x = np.arange(len(rungs))
    for j, head in enumerate(heads):
        means = []
        for rung in rungs:
            vals = [float(r["estimate"]) for r in data if r["head"] == head and r["recalibration"] == rung]
            means.append(np.mean(vals) if vals else np.nan)
        ax.plot(x, means, marker=MARK.get(head, "o"), ms=4, lw=1.1,
                color=CB.get(head, "#444"), label=head)
    ax.set_xticks(x); ax.set_xticklabels(rungs, fontsize=7)
    ax.set_ylabel("ECE (positive-class)", fontsize=7)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, fontsize=5.5)
    _save(fig, "F2_recalibration_ladder")


def fig_F3_logit_range(logit_range_rows):
    """Output-logit range vs ECE (Stage 0 audit)."""
    _style()
    if not logit_range_rows:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for r in logit_range_rows:
        rng = float(r["logit_range"]) if r.get("logit_range") not in ("", None) else np.nan
        ece = float(r["ece"]) if r.get("ece") not in ("", None) else np.nan
        if np.isnan(rng) or np.isnan(ece):
            continue
        ax.scatter(rng, ece, s=18, color=CB.get(r["head"], "#444"),
                   marker=MARK.get(r["head"], "o"), alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("logit range (CheXpert test)", fontsize=7)
    ax.set_ylabel("ECE", fontsize=7)
    ax.grid(alpha=0.25, lw=0.4)
    _save(fig, "F3_logit_range_vs_ece")


def fig_F4_decision(decision_rows):
    """Decision curves: net benefit vs threshold, before/after recalibration."""
    _style()
    if not decision_rows:
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for label, key in (("raw", "raw"), ("recalibrated", "cal")):
        pts = [(float(r["threshold"]), float(r["net_benefit"])) for r in decision_rows
               if r["kind"] == key and r["head"] == "pooled"]
        if pts:
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.2,
                    color="#0072B2" if key == "raw" else "#D55E00", label=label)
    ax.axhline(0, color="#888", lw=0.6, ls="--")
    ax.set_xlabel("threshold probability", fontsize=7)
    ax.set_ylabel("net benefit", fontsize=7)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, fontsize=6)
    _save(fig, "F4_decision_curves")


def fig_reliability(task, head, seed, probs, labels, tag=""):
    _style()
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    ax.plot([0, 1], [0, 1], ls="--", color="#888", lw=0.8)
    from .metrics import ece_bin_table
    tbl = ece_bin_table(probs, labels)
    xs = [b["conf"] for b in tbl if b["n"] > 0]
    ys = [b["acc"] for b in tbl if b["n"] > 0]
    ms = [b["n"] for b in tbl if b["n"] > 0]
    ax.scatter(xs, ys, s=[max(6, m * 0.4) for m in ms], color=CB.get(head, "#444"),
               edgecolor="white", lw=0.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("predicted probability", fontsize=6.5)
    ax.set_ylabel("empirical frequency", fontsize=6.5)
    ax.set_title(f"{task} · {head} · s{seed} {tag}", fontsize=6.5)
    ax.grid(alpha=0.2, lw=0.4)
    _save(fig, f"rel_{task}_{head}_s{seed}{tag}")