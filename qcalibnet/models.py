"""Frozen CXR backbone, 6-d bottleneck, matched-capacity heads.

Three heads share the same 6-dimensional bottleneck:
  mlp            classical MLP 6 → hidden → 1
  product_pqc    6-qubit PQC without entangling gates
  entangling_pqc 6-qubit strongly-entangling PQC

The full DenseNet classifier is an unmatched upper bound, not a fair
quantum-vs-classical comparison.
"""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn
from torchvision import models


def build_backbone(name: str = "densenet121") -> tuple[nn.Module, int]:
    name = name.lower()
    if name == "densenet121":
        net = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        feat_dim = net.classifier.in_features
        net.classifier = nn.Identity()
    elif name == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {name}")
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return net, feat_dim


class Bottleneck(nn.Module):
    def __init__(self, in_dim: int, n_qubits: int = 6):
        super().__init__()
        self.proj = nn.Linear(in_dim, n_qubits)
        self.n_qubits = n_qubits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(x)) * torch.pi


class MLPHead(nn.Module):
    def __init__(self, n_qubits: int = 6, hidden: int = 8, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_qubits, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def _product_layer(weights, n_qubits: int):
    for i in range(n_qubits):
        qml.RX(weights[i, 0], wires=i)
        qml.RY(weights[i, 1], wires=i)
        qml.RZ(weights[i, 2], wires=i)


def _entangling_layer(weights, n_qubits: int):
    _product_layer(weights, n_qubits)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    if n_qubits > 1:
        qml.CNOT(wires=[n_qubits - 1, 0])


def make_qnode(n_qubits: int, n_layers: int, entangle: bool):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        for layer in range(n_layers):
            if entangle:
                _entangling_layer(weights[layer], n_qubits)
            else:
                _product_layer(weights[layer], n_qubits)
        return qml.expval(qml.PauliZ(0))

    return circuit


class PQCHead(nn.Module):
    def __init__(self, n_qubits: int = 6, n_layers: int = 2, entangle: bool = True):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangle = entangle
        self.qnode = make_qnode(n_qubits, n_layers, entangle)
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # default.qubit + backprop is CPU float64. Keep the autograd tape:
        # do not detach z, otherwise the 6-d bottleneck is frozen and the
        # PQC vs MLP comparison is no longer matched-capacity.
        z_cpu = z.to(dtype=torch.float64, device="cpu")
        weights = self.weights.to(dtype=torch.float64, device="cpu")
        expvals = [self.qnode(z_cpu[i], weights) for i in range(z_cpu.shape[0])]
        out = torch.stack(expvals).to(dtype=z.dtype, device=z.device)
        return self.scale * out + self.bias


class QCalibNet(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        head: str = "mlp",
        n_qubits: int = 6,
        n_layers: int = 2,
        mlp_hidden: int = 8,
        mlp_dropout: float = 0.1,
    ):
        super().__init__()
        self.head_name = head
        self.bottleneck = Bottleneck(feat_dim, n_qubits)
        if head == "mlp":
            self.head = MLPHead(n_qubits, mlp_hidden, mlp_dropout)
        elif head == "product_pqc":
            self.head = PQCHead(n_qubits, n_layers, entangle=False)
        elif head == "entangling_pqc":
            self.head = PQCHead(n_qubits, n_layers, entangle=True)
        else:
            raise ValueError(f"Unknown head: {head}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(features)
        return self.head(z)

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
