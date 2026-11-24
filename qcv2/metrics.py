"""Metrics for the v2 pipeline.

Three ECE variants are kept side by side because the v1 code used
positive-class confidence, which is not the standard max-confidence ECE:
  ece_top : standard max-conf ECE  (confidence = max(p,1-p), correctness = yhat==y)
  ece_pos : positive-class ECE     (confidence = p,           accuracy = mean(y))
  smece   : debiased/smoothed ECE  (binomial correction + option to smooth)

All functions take numpy arrays, never torch, never VinDr-derived fitting.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -80.0, 80.0)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def auroc(probs, labels) -> float:
    """Mann-Whitney AUROC with midranks for ties (matches sklearn)."""
    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return float("nan")
    order = np.argsort(probs, kind="mergesort")
    sp = probs[order]
    ranks = np.empty(len(probs), dtype=float)
    i = 0
    n = len(sp)
    while i < n:
        j = i
        while j + 1 < n and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = len(labels) - n_pos
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(probs, labels) -> float:
    """Average precision (step-wise, matches sklearn average_precision_score)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    order = np.argsort(-probs, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float(np.sum(prec * y) / y.sum())


def brier(probs, labels, weights=None) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    se = (probs - labels) ** 2
    if weights is None:
        return float(se.mean())
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return float(np.sum(w * se))


def nll(probs, labels, weights=None) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(labels, dtype=np.float64)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    if weights is None:
        return float(loss.mean())
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return float(np.sum(w * loss))


def _bin_index(values, n_bins, edges=None):
    if edges is None:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(values, edges[1:-1], right=False), 0, n_bins - 1)
    return idx, edges


def _equal_mass_edges(values, n_bins):
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(values, qs))
    return np.concatenate([[0.0], edges, [1.0]])


def ece_top(probs, labels, n_bins: int = 15, weights=None, mode: str = "width"):
    """Standard max-confidence ECE."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    conf = np.maximum(probs, 1 - probs)
    correct = ((probs >= 0.5).astype(float) == labels).astype(float)
    return _binned_gap(conf, correct, n_bins, weights, mode)


def ece_pos(probs, labels, n_bins: int = 15, weights=None, mode: str = "width"):
    """Positive-class confidence ECE (what v1 code computed)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return _binned_gap(probs, labels, n_bins, weights, mode)


def _binned_gap(conf, target, n_bins, weights, mode):
    conf = np.asarray(conf, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if weights is None:
        w = np.ones(len(conf), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    edges = None if mode == "width" else _equal_mass_edges(conf, n_bins)
    idx, _ = _bin_index(conf, n_bins, edges)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        wb = w[m].sum()
        if wb <= 0:
            continue
        gap = abs(np.sum(w[m] * target[m]) / wb - np.sum(w[m] * conf[m]) / wb)
        total += gap * wb
    return float(total)


def mce(probs, labels, n_bins=15, weights=None):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    conf = np.maximum(probs, 1 - probs)
    correct = ((probs >= 0.5).astype(float) == labels).astype(float)
    if weights is None:
        w = np.ones(len(conf))
    else:
        w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    idx, _ = _bin_index(conf, n_bins)
    worst = 0.0
    for b in range(n_bins):
        m = idx == b
        wb = w[m].sum()
        if wb <= 0:
            continue
        gap = abs(np.sum(w[m] * correct[m]) / wb - np.sum(w[m] * conf[m]) / wb)
        worst = max(worst, gap)
    return float(worst)


def smece(probs, labels, n_bins=15, weights=None):
    """Debiased (binomial-corrected) ECE, then averaged over bins.

    For each bin with n_b and empirical accuracy acc_b, the squared bias is
    estimated as acc_b*(1-acc_b)/(n_b-1) (a standard, unbiased-ish correction).
    Reported as the weighted mean of sqrt(max(0, gap_b^2 - var_b)).
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if weights is None:
        w = np.ones(len(probs))
    else:
        w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    idx, _ = _bin_index(probs, n_bins)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        wb = w[m].sum()
        n_b = int(m.sum())
        if wb <= 0 or n_b < 2:
            continue
        acc = np.sum(w[m] * labels[m]) / wb
        conf = np.sum(w[m] * probs[m]) / wb
        gap2 = (acc - conf) ** 2
        var = acc * (1 - acc) / (n_b - 1)
        total += wb * np.sqrt(max(0.0, gap2 - var))
    return float(total)


def murphy_brier(probs, labels, n_bins=15):
    """Murphy decomposition: Brier = reliability - resolution + uncertainty."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n = len(probs)
    obar = labels.mean()
    idx, _ = _bin_index(probs, n_bins)
    reliability = 0.0
    resolution = 0.0
    for b in range(n_bins):
        m = idx == b
        nb = int(m.sum())
        if nb == 0:
            continue
        acc = labels[m].mean()
        conf = probs[m].mean()
        reliability += nb * (conf - acc) ** 2
        resolution += nb * (acc - obar) ** 2
    reliability /= n
    resolution /= n
    uncertainty = obar * (1 - obar)
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier_check": float(reliability - resolution + uncertainty),
    }


def ece_bin_table(probs, labels, n_bins=15):
    """Per-bin rows for reliability diagrams."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    idx, edges = _bin_index(probs, n_bins)
    rows = []
    for b in range(n_bins):
        m = idx == b
        nb = int(m.sum())
        rows.append({
            "bin": b,
            "lo": float(edges[b]),
            "hi": float(edges[b + 1]),
            "n": nb,
            "conf": float(probs[m].mean()) if nb else float("nan"),
            "acc": float(labels[m].mean()) if nb else float("nan"),
        })
    return rows


def weighted_prevalence_weights(labels, target_prev):
    """Class-conditional reweighting to a target prevalence.

    Returns weights w_i with sum/len = 1 such that the weighted positive rate
    equals target_prev, keeping the within-class distributions unchanged.
    """
    labels = np.asarray(labels).astype(int)
    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both classes for prevalence reweighting")
    w_pos = target_prev / n_pos
    w_neg = (1.0 - target_prev) / n_neg
    w = np.where(labels == 1, w_pos, w_neg) * n
    return w / w.sum() * n