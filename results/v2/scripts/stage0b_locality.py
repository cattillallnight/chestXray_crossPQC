"""Stage 0b: numerically verify the product PQC ignores z1..z5.

Independent numpy statevector simulation of the circuit in
qcalibnet/models.py (AngleEmbedding Y, then layers of RX/RY/RZ with an
optional ring of CNOTs, readout <Z0>). No torch / PennyLane needed.

Checks
  1. perturbing z1..z5 leaves <Z0> unchanged (product head)
  2. perturbing z0 changes <Z0> (product head)
  3. perturbation propagates for the entangling head
  4. finite-difference d<Z0>/dW rows 1..5 is 0 (product head)

Writes results/v2/stage0b_locality.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "stage0b_locality.csv"
N = 6


def rx(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(t):
    return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)]).astype(complex)


def apply1(psi, U, q):
    psi = psi.reshape([2] * N)
    psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [q])), 0, q)
    return psi.reshape(-1)


def cnot(psi, c, t):
    flat = psi.copy()
    for i in range(2 ** N):
        bc = (i >> (N - 1 - c)) & 1
        bt = (i >> (N - 1 - t)) & 1
        if bc == 1 and bt == 0:
            j = i | (1 << (N - 1 - t))
            flat[i], flat[j] = psi[j], psi[i]
    return flat


def expz0(psi):
    signs = np.array([1.0 if ((i >> (N - 1)) & 1) == 0 else -1.0 for i in range(2 ** N)])
    return float(np.sum(np.abs(psi) ** 2 * signs))


def circuit(z, weights, entangle):
    psi = np.zeros(2 ** N, dtype=complex)
    psi[0] = 1.0
    for i in range(N):
        psi = apply1(psi, ry(z[i]), i)
    for layer in range(weights.shape[0]):
        for i in range(N):
            psi = apply1(psi, rx(weights[layer, i, 0]), i)
            psi = apply1(psi, ry(weights[layer, i, 1]), i)
            psi = apply1(psi, rz(weights[layer, i, 2]), i)
        if entangle:
            for i in range(N - 1):
                psi = cnot(psi, i, i + 1)
            psi = cnot(psi, N - 1, 0)
    return expz0(psi)


def main() -> None:
    rng = np.random.default_rng(42)
    z0 = rng.uniform(-np.pi, np.pi, N)
    w = rng.normal(0, 0.5, (2, N, 3))
    eps = 0.37
    rows = []

    for entangle, name in ((False, "product"), (True, "entangling")):
        base = circuit(z0, w, entangle)
        for k in range(N):
            zp = z0.copy()
            zp[k] += eps
            d = circuit(zp, w, entangle) - base
            rows.append({
                "head": name,
                "perturbed_feature": f"z{k}",
                "delta_expZ0": f"{d:.3e}",
                "abs_delta": f"{abs(d):.3e}",
                "independent": str(abs(d) < 1e-12),
            })

    # finite-difference gradient of <Z0> wrt each W row (product head)
    h = 1e-6
    base_w = circuit(z0, w, False)
    for r in range(N):
        wp = w.copy()
        wp[0, r, 0] += h
        g = (circuit(z0, wp, False) - base_w) / h
        rows.append({
            "head": "product",
            "perturbed_feature": f"W_row{r}_fd_grad",
            "delta_expZ0": f"{g:.3e}",
            "abs_delta": f"{abs(g):.3e}",
            "independent": str(abs(g) < 1e-8),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"wrote {OUT}")
    for r in rows:
        print(f"{r['head']:11s} {r['perturbed_feature']:14s} "
              f"d={r['delta_expZ0']:>12s} independent={r['independent']}")

    prod = [r for r in rows if r["head"] == "product" and r["perturbed_feature"].startswith("z") and r["perturbed_feature"] != "z0"]
    ok = all(r["independent"] == "True" for r in prod)
    print()
    print("VERDICT product independent of z1..z5:", ok)


if __name__ == "__main__":
    main()