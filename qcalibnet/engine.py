"""Train and evaluate matched-capacity heads on cached features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from qcalibnet.metrics import binary_metrics, fit_temperature
from qcalibnet.models import QCalibNet
from qcalibnet.torchdata import FeatureDataset


def select_device(head: str) -> torch.device:
    if head == "mlp" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        FeatureDataset(features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for feats, y in loader:
        feats = feats.to(device)
        out = model(feats)
        logits.append(out.detach().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def train_head(
    feat_dim: int,
    head: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    cfg: dict,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    device = select_device(head)
    model = QCalibNet(
        feat_dim=feat_dim,
        head=head,
        n_qubits=int(cfg["n_qubits"]),
        n_layers=int(cfg["n_layers"]),
        mlp_hidden=int(cfg["mlp_hidden"]),
        mlp_dropout=float(cfg["mlp_dropout"]),
    ).to(device)
    batch_size = int(cfg["pqc_batch_size"] if head != "mlp" else cfg["batch_size"])
    train_loader = make_loader(train_x, train_y, batch_size, shuffle=True)
    val_loader = make_loader(val_x, val_y, batch_size, shuffle=False)
    opt = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = None
    best_auroc = -1.0
    bad = 0
    history = []
    epochs = int(cfg["epochs"])
    patience = int(cfg["patience"])

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for feats, y in tqdm(train_loader, desc=f"{head} ep{epoch}", leave=False):
            feats = feats.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(feats)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(y)
            n_seen += len(y)
        val_logits, val_labels = collect_logits(model, val_loader, device)
        val_metrics = binary_metrics(val_logits, val_labels, n_bins=int(cfg["n_bins"]))
        auroc = val_metrics["auroc"]
        score = -1.0 if (auroc is None or (isinstance(auroc, float) and np.isnan(auroc))) else float(auroc)
        history.append({"epoch": epoch, "train_loss": running / max(n_seen, 1), "val_auroc": auroc})
        if score > best_auroc + 1e-4:
            best_auroc = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "model": model,
        "device": device,
        "history": history,
        "best_val_auroc": best_auroc,
        "n_trainable": model.n_trainable(),
        "batch_size": batch_size,
    }


def evaluate_split(model: nn.Module, features: np.ndarray, labels: np.ndarray, cfg: dict, device: torch.device, temperature: float | None = None) -> dict:
    loader = make_loader(features, labels, batch_size=int(cfg["batch_size"]), shuffle=False)
    logits, y = collect_logits(model, loader, device)
    raw = binary_metrics(logits, y, n_bins=int(cfg["n_bins"]))
    raw["temperature"] = 1.0
    if temperature is None:
        return {"raw": raw, "calibrated": None, "logits": logits, "labels": y}
    calibrated = binary_metrics(logits / temperature, y, n_bins=int(cfg["n_bins"]))
    calibrated["temperature"] = float(temperature)
    return {"raw": raw, "calibrated": calibrated, "logits": logits, "labels": y}


def fit_val_temperature(model: nn.Module, val_x: np.ndarray, val_y: np.ndarray, cfg: dict, device: torch.device) -> float:
    loader = make_loader(val_x, val_y, batch_size=int(cfg["batch_size"]), shuffle=False)
    logits, y = collect_logits(model, loader, device)
    return fit_temperature(logits, y)


def save_run(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
