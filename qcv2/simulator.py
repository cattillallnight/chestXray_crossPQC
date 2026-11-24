"""Batched torch statevector simulator for the PQC heads (Stage 2.4).

Mirrors the ansatz in qcalibnet/models.py:
  AngleEmbedding(Y) on all wires  ->  L layers of RX/RY/RZ per wire
  ->  optional ring of CNOTs (i,i+1) ... (n-1,0) after each layer
  ->  readout: expval(Z0)  OR  learned linear combination of all expval(Zi)

Runs fully batched over a (B, n) input tensor; gradients flow via autograd.
Tensor layout: state is (B, 2, 2, ..., 2) with axis 1 = wire 0 (MSB) and
axis n = wire n-1 (LSB). Verified against PennyLane to 1e-8 in tests.
"""
from __future__ import annotations

import torch

CDTYPE = torch.complex128
DTYPE = torch.float64


def _ry(t):
    c = torch.cos(t / 2)
    s = torch.sin(t / 2)
    z = torch.zeros_like(c)
    real = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    imag = torch.stack([torch.stack([z, z], -1), torch.stack([z, z], -1)], -2)
    return torch.complex(real, imag)


def _rx(t):
    c = torch.cos(t / 2)
    s = torch.sin(t / 2)
    z = torch.zeros_like(c)
    return torch.complex(
        torch.stack([torch.stack([c, z], -1), torch.stack([z, c], -1)], -2),
        torch.stack([torch.stack([z, -s], -1), torch.stack([-s, z], -1)], -2),
    )


def _rz(t):
    h = t / 2
    z = torch.zeros_like(h)
    e_neg = torch.complex(torch.cos(h), -torch.sin(h))
    e_pos = torch.complex(torch.cos(h), torch.sin(h))
    return torch.stack([
        torch.stack([e_neg, z], -1),
        torch.stack([z, e_pos], -1),
    ], -2)


def _apply_1q(state, U, wire, n):
    """state (B, 2,..,2); U (B,2,2). Multiply U into axis (wire+1)."""
    axis = wire + 1
    moved = state.movedim(axis, 1)              # (B, 2, ...)
    shape = moved.shape
    flat = moved.reshape(shape[0], 2, -1)       # (B,2,rest)
    out = torch.einsum("bij,bjk->bik", U, flat)  # (B,2,rest)
    out = out.reshape(shape)
    return out.movedim(1, axis)


def _cnot(state, ctrl, targ, n):
    a_c = ctrl + 1
    a_t = targ + 1
    perm = [0] + [a_c, a_t] + [i for i in range(1, n + 1) if i not in (a_c, a_t)]
    s = state.permute(*perm).reshape(state.shape[0], 4, -1)
    a, b, c, d = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    new = torch.stack([a, b, d, c], dim=1).reshape(
        [state.shape[0]] + [2] * n)
    inv = [0] * (n + 1)
    for i, p in enumerate(perm):
        inv[p] = i
    return new.permute(*inv)


def _expvals(state, n):
    probs = state.abs() ** 2                      # (B,2,...,2)
    out = []
    for i in range(n):
        # sum probs over axes, weighting by sign of wire i
        sign = torch.tensor([1.0, -1.0], dtype=DTYPE)
        p = probs.movedim(i + 1, 1)               # (B,2,...)
        p = p.reshape(p.shape[0], 2, -1)
        ev = (p[:, 0, :].sum(1) - p[:, 1, :].sum(1))
        out.append(ev)
    return torch.stack(out, dim=1)


def _combined_rot(weights, wire):
    """Return (B,2,2) matrix RZ@RY@RX for a given wire, batched."""
    rx = _rx(weights[..., wire, 0])
    ry = _ry(weights[..., wire, 1])
    rz = _rz(weights[..., wire, 2])
    return torch.einsum("bij,bjk,bkl->bil", rz, ry, rx)


def _product_z0_fast(z, weights):
    """Analytic <Z0> for the product head (no entanglement).

    Proven independent of z1..z5 (Stage 0b): only wire 0 matters, and the
    circuit factors. State of wire 0 = prod_layers(U_l) . RY(z0) |0>, giving
    a 2-vector per sample. Exact, O(1) per sample, verified vs PennyLane.
    """
    B = z.shape[0]
    st = torch.stack([_ry(z[:, 0]), torch.zeros_like(_ry(z[:, 0]))], dim=-1)  # (B,2,2)? build RY|0>
    # RY(z)|0> = [cos(z/2), sin(z/2)]
    st = torch.stack([torch.cos(z[:, 0] / 2), torch.sin(z[:, 0] / 2)], dim=-1).to(CDTYPE)
    L = weights.shape[0]
    for layer in range(L):
        w = weights[layer, :, :].unsqueeze(0)      # (1,n,3)
        U = _combined_rot(w, 0).expand(B, 2, 2)    # (B,2,2)
        st = torch.einsum("bij,bj->bi", U, st)
    return (st[:, 0].abs() ** 2 - st[:, 1].abs() ** 2).real


def _product_expvals_fast(z, weights):
    """All <Zi> for the product (no-entanglement) head, each wire factored.

    Wire i evolves independently: state_i = prod_layers(U_l,i) . RY(z_i)|0>.
    Returns (B, n). Exact.
    """
    B, n = z.shape
    L = weights.shape[0]
    out = []
    for i in range(n):
        st = torch.stack([torch.cos(z[:, i] / 2), torch.sin(z[:, i] / 2)], dim=-1).to(CDTYPE)
        for layer in range(L):
            U = _combined_rot(weights[layer].unsqueeze(0), i).expand(B, 2, 2)
            st = torch.einsum("bij,bj->bi", U, st)
        out.append((st[:, 0].abs() ** 2 - st[:, 1].abs() ** 2).real)
    return torch.stack(out, dim=1)


def pqc_forward(z, weights, entangle, readout="z0", readout_weights=None):
    z = z.to(DTYPE)
    weights = weights.to(DTYPE)
    B, n = z.shape
    if not entangle:
        if readout == "z0":
            return _product_z0_fast(z, weights)
        if readout == "linear":
            ex = _product_expvals_fast(z, weights)
            return (ex * readout_weights.to(DTYPE).expand(B, n)).sum(dim=1)
    L = weights.shape[0]
    state = torch.zeros([B] + [2] * n, dtype=CDTYPE)
    state[(slice(None),) + (0,) * n] = 1.0
    for i in range(n):
        state = _apply_1q(state, _ry(z[:, i]), i, n)
    for layer in range(L):
        for i in range(n):
            state = _apply_1q(state, _combined_rot(weights[layer].unsqueeze(0), i).expand(B, 2, 2), i, n)
        if entangle:
            for i in range(n - 1):
                state = _cnot(state, i, i + 1, n)
            state = _cnot(state, n - 1, 0, n)
    expz = _expvals(state, n)
    if readout == "z0":
        return expz[:, 0]
    if readout == "linear":
        return (expz * readout_weights.to(DTYPE).expand(B, n)).sum(dim=1)
    raise ValueError(readout)