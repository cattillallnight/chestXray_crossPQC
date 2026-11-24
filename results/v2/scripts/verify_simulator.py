"""Cross-check the batched torch simulator against PennyLane to 1e-8.

Mirrors qcalibnet/models.py exactly (AngleEmbedding Y, L x RX/RY/RZ,
optional ring CNOT, expval(Z0) and linear readout). Prints max abs error.
Exit code 1 if any comparison exceeds 1e-8.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pennylane as qml

from qcv2.simulator import pqc_forward


def pl_circuit(z, w, entangle, readout, rw):
    n = len(z)
    L = w.shape[0]
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="torch")
    def circ(zz, ww):
        qml.AngleEmbedding(zz, wires=range(n), rotation="Y")
        for layer in range(L):
            for i in range(n):
                qml.RX(ww[layer, i, 0], wires=i)
                qml.RY(ww[layer, i, 1], wires=i)
                qml.RZ(ww[layer, i, 2], wires=i)
            if entangle:
                for i in range(n - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n - 1, 0])
        if readout == "z0":
            return qml.expval(qml.PauliZ(0))
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    if readout == "z0":
        return circ(torch.tensor(z, dtype=torch.float64),
                    torch.tensor(w, dtype=torch.float64))
    ev = torch.stack(circ(torch.tensor(z, dtype=torch.float64),
                          torch.tensor(w, dtype=torch.float64)))
    return (ev * torch.tensor(rw, dtype=torch.float64)).sum()


def main() -> int:
    rng = np.random.default_rng(0)
    maxerr = 0.0
    n_cases = 0
    for trial in range(120):
        n = 6
        L = 2
        z = rng.uniform(-np.pi, np.pi, n)
        w = rng.normal(0, 0.6, (L, n, 3))
        ent = bool(trial % 2)
        rw = rng.normal(0, 1, n)
        for readout in ("z0", "linear"):
            pl = float(np.asarray(pl_circuit(z, w, ent, readout, rw)))
            tor = float(pqc_forward(torch.tensor(z[None]), torch.tensor(w),
                                    ent, readout,
                                    torch.tensor(rw)).item())
            err = abs(pl - tor)
            maxerr = max(maxerr, err)
            n_cases += 1
    print(f"cases={n_cases} max_abs_error={maxerr:.3e}")
    if maxerr > 1e-8:
        print("FAIL: simulator vs PennyLane exceeds 1e-8")
        return 1
    print("PASS: simulator matches PennyLane to 1e-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())