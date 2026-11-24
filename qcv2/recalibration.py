"""Post-hoc recalibration rungs (Stage 1.2).

Every rung exposes fit(logits, labels, origin=...) -> callable(logits) -> probs.
The origin tag goes through guard.assert_not_vindr, so a VinDr tensor can only
reach a fit through the ORACLE rung with allow_vindr=True.

Rungs: none | temperature | platt | prior_shift | bbse | saerens_em | oracle_platt
"""
from __future__ import annotations

import numpy as np

from .guard import assert_not_vindr
from .metrics import sigmoid

T_BOUNDS = (0.05, 20.0)


def _bce(probs, labels, w=None, eps=1e-12):
    p = np.clip(probs, eps, 1 - eps)
    y = np.asarray(labels, dtype=np.float64)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    if w is None:
        return float(loss.mean())
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    return float(np.sum(w * loss))


class Identity:
    name = "none"

    def fit(self, logits, labels, origin="chexpert_val", allow_vindr=False, **kw):
        assert_not_vindr(origin, allow_vindr)
        return lambda z: sigmoid(z)


class Temperature:
    """Scalar temperature with bounds, fitted by golden-section on NLL."""

    name = "temperature"

    def __init__(self, bounds=T_BOUNDS):
        self.bounds = bounds
        self.value = None
        self.at_bound = False

    def fit(self, logits, labels, origin="chexpert_val", allow_vindr=False, **kw):
        assert_not_vindr(origin, allow_vindr)
        z = np.asarray(logits, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        lo, hi = self.bounds

        def nll_at(t):
            return _bce(sigmoid(z / t), y)

        a, b = np.log(lo), np.log(hi)
        gr = (np.sqrt(5) - 1) / 2
        c, d = b - gr * (b - a), a + gr * (b - a)
        for _ in range(80):
            if nll_at(np.exp(c)) < nll_at(np.exp(d)):
                b = d
            else:
                a = c
            c, d = b - gr * (b - a), a + gr * (b - a)
        t = float(np.exp((a + b) / 2))
        self.value = t
        self.at_bound = bool(t <= lo * 1.001 or t >= hi * 0.999)
        return lambda zz: sigmoid(np.asarray(zz, dtype=np.float64) / t)


class Platt:
    """a*logit + b, fitted by Newton/IRLS on weighted NLL."""

    name = "platt"

    def __init__(self):
        self.a = None
        self.b = None

    def fit(self, logits, labels, origin="chexpert_val", allow_vindr=False,
            sample_weight=None, **kw):
        assert_not_vindr(origin, allow_vindr)
        z = np.asarray(logits, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        w = np.ones(len(z)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        w = w / w.sum()

        a, b = 1.0, 0.0
        for _ in range(100):
            p = sigmoid(a * z + b)
            r = w * (p - y)
            g = np.array([np.sum(r * z), np.sum(r)])
            wgt = w * p * (1 - p)
            h = np.array([
                [np.sum(wgt * z * z), np.sum(wgt * z)],
                [np.sum(wgt * z), np.sum(wgt)],
            ]) + 1e-9 * np.eye(2)
            try:
                step = np.linalg.solve(h, g)
            except np.linalg.LinAlgError:
                break
            a -= step[0]
            b -= step[1]
            if np.max(np.abs(step)) < 1e-10:
                break
        self.a, self.b = float(a), float(b)
        return lambda zz: sigmoid(self.a * np.asarray(zz, dtype=np.float64) + self.b)


class PriorShift:
    """Intercept-only prior shift: p' = sigmoid(logit + b), b chosen so the
    mean predicted probability on a reference slice equals a target prior.

    Fitted on CheXpert val against the *CheXpert val* prior (a null correction
    unless a target prior is supplied from labels-free assumptions). For Stage
    1.2 this is fitted on CheXpert val only and applied unchanged elsewhere.
    """

    name = "prior_shift"

    def __init__(self):
        self.b = 0.0

    def fit(self, logits, labels, origin="chexpert_val", allow_vindr=False, target_prior=None, **kw):
        assert_not_vindr(origin, allow_vindr)
        z = np.asarray(logits, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        want = float(y.mean()) if target_prior is None else float(target_prior)
        lo, hi = -30.0, 30.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if sigmoid(z + mid).mean() < want:
                lo = mid
            else:
                hi = mid
        self.b = float((lo + hi) / 2)
        return lambda zz: sigmoid(np.asarray(zz, dtype=np.float64) + self.b)


class BBSE:
    """Black-box shift estimation on *unlabelled target logits*.

    BBSE does not need target labels but it needs the *trained classifier's*
    confusion on source. It estimates the target prior and applies a
    prior-correction intercept. Because it reads target logits without labels,
    that is not a forbidden fit on VinDr labels; it is passed allow_vindr via
    the caller for the unlabelled target logit vector only.
    """

    name = "bbse"

    def __init__(self):
        self.target_prior = None
        self.b = 0.0

    def fit(self, logits, labels, origin="chexpert_val", allow_vindr=False,
            target_logits=None, **kw):
        assert_not_vindr(origin, allow_vindr)
        z = np.asarray(logits, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if target_logits is None:
            raise ValueError("BBSE needs target_logits (unlabelled)")
        tz = np.asarray(target_logits, dtype=np.float64)
        # source confusion on the *hard* classifier
        pred = (sigmoid(z) >= 0.5).astype(int)
        # 2x2 confusion: rows = true class, cols = predicted
        c = np.zeros((2, 2), dtype=np.float64)
        for t in (0, 1):
            for p in (0, 1):
                c[t, p] = np.mean((y == t) & (pred == p))
        # target predicted distribution
        q = np.array([np.mean((sigmoid(tz) < 0.5)), np.mean((sigmoid(tz) >= 0.5))])
        try:
            prior = np.linalg.solve(c.T, q)
        except np.linalg.LinAlgError:
            prior = np.array([0.5, 0.5])
        prior = np.clip(prior, 1e-4, 1 - 1e-4)
        prior = prior / prior.sum()
        self.target_prior = float(prior[1])
        # apply as intercept so mean predicted prob matches estimated prior
        lo, hi = -30.0, 30.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if sigmoid(tz + mid).mean() < self.target_prior:
                lo = mid
            else:
                hi = mid
        self.b = float((lo + hi) / 2)
        return lambda zz: sigmoid(np.asarray(zz, dtype=np.float64) + self.b)


class SaerensEM:
    """Saerens-Latinne-Decaestecker EM on unlabelled target logits.

    Estimates the target prior by EM using the source-calibrated posterior,
    then applies the corresponding prior-correction shift.
    """

    name = "saerens_em"

    def __init__(self):
        self.target_prior = None
        self.b = 0.0

    def fit(self, logits, labels, origin="chexpert_val", allow_vindr=False,
            target_logits=None, **kw):
        assert_not_vindr(origin, allow_vindr)
        z = np.asarray(logits, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if target_logits is None:
            raise ValueError("Saerens-EM needs target_logits (unlabelled)")
        tz = np.asarray(target_logits, dtype=np.float64)
        src_prior = float(y.mean())
        # EM: p(y=1|x;pi) proportional to pi * p_src(y=1|x)/src_prior
        base = sigmoid(tz)
        pi = src_prior
        for _ in range(200):
            num = pi * base / src_prior
            den = num + (1 - pi) * (1 - base) / (1 - src_prior)
            post = num / np.clip(den, 1e-12, None)
            new_pi = float(np.clip(post.mean(), 1e-4, 1 - 1e-4))
            if abs(new_pi - pi) < 1e-10:
                pi = new_pi
                break
            pi = new_pi
        self.target_prior = pi
        # prior correction intercept: logit' = logit + log(pi/(1-pi)) - log(src/(1-src))
        self.b = float(np.log(pi / (1 - pi)) - np.log(src_prior / (1 - src_prior)))
        return lambda zz: sigmoid(np.asarray(zz, dtype=np.float64) + self.b)


class OraclePlatt:
    """Upper bound: Platt fitted on k labelled VinDr images.

    This is the ONLY rung allowed to fit on VinDr labels; the caller must pass
    allow_vindr=True and the results must be labelled oracle (never deployable).
    """

    name = "oracle_platt"

    def fit(self, logits, labels, origin="vindr_bal", allow_vindr=True, **kw):
        assert_not_vindr(origin, allow_vindr=True)
        return Platt().fit(logits, labels, origin=origin, allow_vindr=True)


RUNG_ORDER = ["none", "temperature", "platt", "prior_shift", "bbse", "saerens_em", "oracle_platt"]


def build_rung(name: str):
    mapping = {
        "none": Identity,
        "temperature": Temperature,
        "platt": Platt,
        "prior_shift": PriorShift,
        "bbse": BBSE,
        "saerens_em": SaerensEM,
        "oracle_platt": OraclePlatt,
    }
    if name not in mapping:
        raise ValueError(f"unknown rung {name}")
    return mapping[name]()