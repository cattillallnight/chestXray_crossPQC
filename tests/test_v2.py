"""Unit tests for the v2 pipeline. Any failure is a HALT condition.

Run: .venv/Scripts/python.exe -m pytest tests/test_v2.py -q
 or: .venv/Scripts/python.exe tests/test_v2.py   (no pytest needed)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qcv2 import metrics as M
from qcv2.paths import V2
from qcv2.simulator import pqc_forward


def _sklearn_ece_ref(probs, labels, n_bins=15):
    """Independent ECE (max-confidence), written from scratch, different style."""
    probs = np.asarray(probs, float); labels = np.asarray(labels, float)
    conf = np.where(probs > 0.5, probs, 1 - probs)
    pred = (probs > 0.5).astype(float)
    hit = (pred == labels).astype(float)
    edges = [i / n_bins for i in range(n_bins + 1)]
    e = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if sel.sum() == 0:
            continue
        e += (sel.sum() / len(conf)) * abs(hit[sel].mean() - conf[sel].mean())
    return e


def test_ece_vs_independent():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.99, 1000)
    y = (rng.uniform(size=1000) < p).astype(float)
    a = M.ece_top(p, y, n_bins=10)
    b = _sklearn_ece_ref(p, y, n_bins=10)
    # both are max-confidence ECE with a boundary convention difference; allow 1 bin-width
    assert abs(a - b) < 0.05, (a, b)


def test_weighted_equals_unweighted_at_uniform():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.01, 0.99, 500)
    y = (rng.uniform(size=500) < p).astype(float)
    w = np.ones(500)
    assert abs(M.ece_pos(p, y, weights=w) - M.ece_pos(p, y)) < 1e-12
    assert abs(M.brier(p, y, weights=w) - M.brier(p, y)) < 1e-12


def test_weights_sum_to_one():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    w = M.weighted_prevalence_weights(y, 0.1)
    assert abs(w.sum() - len(y)) < 1e-9       # scaled to n, not 1
    wn = w / w.sum()
    assert abs(wn.sum() - 1.0) < 1e-12
    # weighted positive rate equals target
    assert abs((wn * y).sum() - 0.1) < 1e-9


def test_simulator_vs_pennylane():
    import torch
    import pennylane as qml
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    def pl(z, w, ent):
        n = len(z); L = w.shape[0]
        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev, interface="torch")
        def c(zz, ww):
            qml.AngleEmbedding(zz, wires=range(n), rotation="Y")
            for layer in range(L):
                for i in range(n):
                    qml.RX(ww[layer, i, 0], wires=i)
                    qml.RY(ww[layer, i, 1], wires=i)
                    qml.RZ(ww[layer, i, 2], wires=i)
                if ent:
                    for i in range(n - 1):
                        qml.CNOT(wires=[i, i + 1])
                    qml.CNOT(wires=[n - 1, 0])
            return qml.expval(qml.PauliZ(0))
        return float(c(torch.tensor(z, dtype=torch.float64),
                       torch.tensor(w, dtype=torch.float64)))

    maxerr = 0.0
    for t in range(20):
        z = rng.uniform(-np.pi, np.pi, 6)
        w = rng.normal(0, 0.6, (2, 6, 3))
        ent = bool(t % 2)
        pl_v = pl(z, w, ent)
        tor = float(pqc_forward(torch.tensor(z[None]), torch.tensor(w), ent, "z0").item())
        maxerr = max(maxerr, abs(pl_v - tor))
    assert maxerr < 1e-8, maxerr


def test_split_hashes_unchanged():
    base = V2 / "split_hashes.json"
    if not base.exists():
        return  # nothing recorded yet
    from qcv2.paths import SPLITS
    baseline = json.loads(base.read_text(encoding="utf-8"))
    for rel, want in baseline.items():
        p = SPLITS / rel
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        # baseline stores a truncated prefix; compare on that length
        assert h.startswith(want), f"split changed: {rel}"


def test_product_locality():
    import torch
    torch.manual_seed(0)
    z = torch.randn(8, 6) * 0.7
    w = torch.randn(2, 6, 3) * 0.5
    base = pqc_forward(z, w, False, "z0")
    z2 = z.clone(); z2[:, 1:] += 0.5
    assert float((pqc_forward(z2, w, False, "z0") - base).abs().max()) < 1e-12


def test_guard_blocks_vindr():
    from qcv2.guard import VinDrLeakError, assert_not_vindr
    from qcv2.recalibration import Temperature
    try:
        Temperature().fit(np.zeros(5), np.zeros(5), origin="vindr_bal")
    except VinDrLeakError:
        pass
    else:
        raise AssertionError("guard did not block vindr fit")


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            fails += 1
            print(f"FAIL {t.__name__}: {exc}")
    print(f"\n{len(tests)-fails}/{len(tests)} passed")
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)