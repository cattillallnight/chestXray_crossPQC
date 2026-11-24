"""Stage 2.3 heads on identical z.

All heads map z in R^6 -> one logit. Parameter counts are printed by
head_param_count() and recorded with every run.

  mlp             6 -> 8 -> 1 (tanh)
  mlp_bounded     6 -> 8 -> tanh  -> a*tanh(h)+b   (matched to PQC logit range)
  pqc_<ent>_<ro>  6-wire PQC, entanglement {none,ring} x readout {z0,linear}
  trig_add        sum_i a_i cos z_i + b_i sin z_i + c          (bounded readout)
  trig_pair       trig_add + sum_{i<j} c_ij cos z_i cos z_j    (pairwise products)
  logistic        logistic regression (linear in z)

Every head exposes forward(z) -> logit and parameters() for the optimizer.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .simulator import pqc_forward


class MLPHead(nn.Module):
    kind = "mlp"

    def __init__(self, n=6, hidden=8):
        super().__init__()
        self.l1 = nn.Linear(n, hidden)
        self.l2 = nn.Linear(hidden, 1)

    def forward(self, z):
        return self.l2(torch.tanh(self.l1(z))).squeeze(-1)


class BoundedMLPHead(nn.Module):
    kind = "mlp_bounded"

    def __init__(self, n=6, hidden=8):
        super().__init__()
        self.l1 = nn.Linear(n, hidden)
        self.l2 = nn.Linear(hidden, 1)
        self.a = nn.Parameter(torch.tensor(2.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, z):
        h = self.l2(torch.tanh(self.l1(z))).squeeze(-1)
        return self.a * torch.tanh(h) + self.b


class PQCHead(nn.Module):
    """Wraps the batched simulator."""

    def __init__(self, n=6, layers=2, entangle=True, readout="z0"):
        super().__init__()
        self.n = n
        self.layers = layers
        self.entangle = entangle
        self.readout = readout
        self.weights = nn.Parameter(0.01 * torch.randn(layers, n, 3))
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        if readout == "linear":
            self.readout_weights = nn.Parameter(torch.ones(n) / n)
        else:
            self.register_parameter("readout_weights", None)
        self.kind = f"pqc_{'ring' if entangle else 'none'}_{readout}"

    def forward(self, z):
        rw = self.readout_weights if self.readout == "linear" else None
        ev = pqc_forward(z, self.weights, self.entangle, self.readout, rw)
        return (self.scale * ev + self.bias).to(z.dtype)


class TrigAddHead(nn.Module):
    kind = "trig_add"

    def __init__(self, n=6):
        super().__init__()
        self.a = nn.Parameter(0.01 * torch.randn(n))
        self.b = nn.Parameter(0.01 * torch.randn(n))
        self.c = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, z):
        s = (self.a * torch.cos(z) + self.b * torch.sin(z)).sum(-1) + self.c
        return self.scale * torch.tanh(s)


class TrigPairHead(nn.Module):
    kind = "trig_pair"

    def __init__(self, n=6):
        super().__init__()
        self.a = nn.Parameter(0.01 * torch.randn(n))
        self.b = nn.Parameter(0.01 * torch.randn(n))
        self.c = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.ones(1))
        self.pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        self.cp = nn.Parameter(0.01 * torch.randn(len(self.pairs)))

    def forward(self, z):
        s = (self.a * torch.cos(z) + self.b * torch.sin(z)).sum(-1) + self.c
        pr = sum(self.cp[k] * torch.cos(z[:, i]) * torch.cos(z[:, j])
                 for k, (i, j) in enumerate(self.pairs))
        return self.scale * torch.tanh(s + pr)


class LogisticHead(nn.Module):
    kind = "logistic"

    def __init__(self, n=6):
        super().__init__()
        self.w = nn.Linear(n, 1)

    def forward(self, z):
        return self.w(z).squeeze(-1)


HEADS = {
    "mlp": lambda: MLPHead(),
    "mlp_bounded": lambda: BoundedMLPHead(),
    "pqc_none_z0": lambda: PQCHead(entangle=False, readout="z0"),
    "pqc_ring_z0": lambda: PQCHead(entangle=True, readout="z0"),
    "pqc_none_linear": lambda: PQCHead(entangle=False, readout="linear"),
    "pqc_ring_linear": lambda: PQCHead(entangle=True, readout="linear"),
    "trig_add": lambda: TrigAddHead(),
    "trig_pair": lambda: TrigPairHead(),
    "logistic": lambda: LogisticHead(),
}


def build_head(name: str) -> nn.Module:
    if name not in HEADS:
        raise ValueError(f"unknown head {name}")
    return HEADS[name]()


def head_param_count(name: str) -> int:
    m = build_head(name)
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))