"""Stage 2.1/2.2: fixed bottleneck.

PCA to 6 dims fit on CheXpert TRAIN ONLY, scaled to [-pi, pi] with train
quantiles. W is not trained. The paper's learned-W bottleneck is kept as a
labelled legacy comparison (handled by 'legacy_w' config flag).
"""
from __future__ import annotations

import numpy as np

from .paths import SPLITS
from . import data as D


class PCABottleneck:
    def __init__(self, n_components=6):
        self.k = n_components
        self.mean_ = None
        self.components_ = None
        self.q_lo_ = None
        self.q_hi_ = None

    def fit(self, X: np.ndarray) -> "PCABottleneck":
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # economy SVD-based PCA
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt[: self.k]          # (k, d)
        proj = Xc @ self.components_.T           # (n, k)
        self.q_lo_ = np.quantile(proj, 0.01, axis=0)
        self.q_hi_ = np.quantile(proj, 0.99, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        proj = (X - self.mean_) @ self.components_.T
        lo, hi = self.q_lo_, self.q_hi_
        span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        unit = np.clip((proj - lo) / span, 0.0, 1.0)
        return unit * 2 * np.pi - np.pi          # [-pi, pi]

    def to_state(self) -> dict:
        return {"k": self.k, "mean": self.mean_.tolist(),
                "components": self.components_.tolist(),
                "q_lo": self.q_lo_.tolist(), "q_hi": self.q_hi_.tolist()}


def fit_bottleneck(task: str, n_components=6) -> PCABottleneck:
    rows = D.read_split(task, "chexpert_train.csv")
    X = D.features_for(rows)
    return PCABottleneck(n_components).fit(X)


def make_split(task: str, name: str, bottleneck: PCABottleneck):
    rows = D.read_split(task, name)
    X = D.features_for(rows)
    z = bottleneck.transform(X)
    y = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
    meta = {
        "patient_id": [r.get("patient_id", "") for r in rows],
        "sex": [r.get("Sex", "") for r in rows],
        "age": [r.get("Age", "") for r in rows],
        "ap_pa": [r.get("AP/PA", "") for r in rows],
        "image_id": [r.get("image_id", "") for r in rows],
    }
    return z.astype(np.float64), y, meta