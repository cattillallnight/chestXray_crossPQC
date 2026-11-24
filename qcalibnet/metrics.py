"""Discrimination and calibration metrics, plus temperature scaling."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> tuple[float, float]:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not np.any(mask):
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        gap = abs(acc - conf)
        ece += gap * (mask.sum() / n)
        mce = max(mce, gap)
    return float(ece), float(mce)


def binary_metrics(logits: np.ndarray, labels: np.ndarray, n_bins: int = 15, threshold: float = 0.5) -> dict:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    probs = sigmoid(logits)
    pred = (probs >= threshold).astype(np.int32)
    out = {
        "n": int(len(labels)),
        "n_pos": int(labels.sum()),
        "prevalence": float(labels.mean()) if len(labels) else 0.0,
        "accuracy": float(accuracy_score(labels, pred)) if len(labels) else 0.0,
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "sensitivity": float(recall_score(labels, pred, zero_division=0)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "specificity": float(recall_score(1 - labels, 1 - pred, zero_division=0)),
        "brier": float(brier_score_loss(labels, probs)) if len(np.unique(labels)) else float("nan"),
    }
    if len(np.unique(labels)) < 2:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    else:
        out["auroc"] = float(roc_auc_score(labels, probs))
        out["auprc"] = float(average_precision_score(labels, probs))
    ece, mce = expected_calibration_error(probs, labels, n_bins=n_bins)
    out["ece"] = ece
    out["mce"] = mce
    return out


def fit_temperature(logits: np.ndarray, labels: np.ndarray, steps: int = 200, lr: float = 0.05) -> float:
    """Minimize BCE on a scalar temperature using the validation logits."""
    z = torch.tensor(np.asarray(logits, dtype=np.float32).reshape(-1))
    y = torch.tensor(np.asarray(labels, dtype=np.float32).reshape(-1))
    log_t = nn.Parameter(torch.zeros(1))
    opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=steps)

    def closure():
        opt.zero_grad(set_to_none=True)
        temperature = log_t.exp().clamp(min=1e-3, max=100.0)
        loss = nn.functional.binary_cross_entropy_with_logits(z / temperature, y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(min=1e-3, max=100.0).item())
