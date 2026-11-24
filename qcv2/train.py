"""Train one Stage 2 head on the PCAbottleneck. All fitting on CheXpert only.

Early stopping on validation NLL (AUROC also logged). Same optimizer, lr,
batch size, epochs and early-stopping rule for ALL heads. Seeds fixed by cfg.
The VinDr guard is enforced by never passing target data into this module.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .guard import assert_not_vindr
from .heads import build_head
from .metrics import auroc, nll, sigmoid


def _standardize_weights(y: np.ndarray, mode: str) -> np.ndarray:
    """Per-example weights for the training prior ablation."""
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if mode == "balanced":
        wp = 0.5 / max(n_pos, 1)
        wn = 0.5 / max(n_neg, 1)
        w = np.where(y == 1, wp, wn)
    elif mode == "natural":
        w = np.ones(n)
    else:
        w = np.ones(n)
    return (w / w.sum() * n).astype(np.float64)


def train_head(cfg: dict, z_tr, y_tr, z_val, y_val, seed: int, device="cpu") -> dict:
    assert_not_vindr("chexpert_train")
    torch.manual_seed(seed)
    np.random.seed(seed)
    head = build_head(cfg["head"]).to(device)
    bs = cfg["pqc_batch_size"] if cfg["head"].startswith("pqc") else cfg["batch_size"]
    lr = float(cfg["lr"])
    wd = float(cfg["weight_decay"])
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)

    zt = torch.from_numpy(z_tr.astype(np.float32))
    yt = torch.from_numpy(y_tr.astype(np.float32))
    zv = torch.from_numpy(z_val.astype(np.float32))
    yv = torch.from_numpy(y_val.astype(np.float32))

    w = _standardize_weights(y_tr, cfg.get("train_prior", "balanced"))
    wt = torch.from_numpy(w.astype(np.float32))

    n = len(zt)
    best = {"val_nll": float("inf"), "state": None, "epoch": 0}
    bad = 0
    history = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        head.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            zb = zt[idx].to(device)
            yb = yt[idx].to(device)
            wb = wt[idx].to(device)
            opt.zero_grad(set_to_none=True)
            logit = head(zb)
            loss = (nn.functional.binary_cross_entropy_with_logits(
                logit, yb, reduction="none") * wb).sum() / wb.sum()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(idx)
        head.eval()
        with torch.no_grad():
            vlog = head(zv.to(device)).detach().cpu().numpy()
        vnll = nll(sigmoid(vlog), y_val)
        vauc = auroc(sigmoid(vlog), y_val)
        history.append({"epoch": epoch, "val_nll": vnll, "val_auroc": vauc})
        if vnll < best["val_nll"] - 1e-5:
            best = {"val_nll": vnll, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                break
    if best["state"] is not None:
        head.load_state_dict(best["state"])
    head.eval()
    return {"model": head, "best": best, "history": history,
            "n_params": int(sum(p.numel() for p in head.parameters() if p.requires_grad)),
            "epochs_run": len(history), "batch_size": bs}


@torch.no_grad()
def predict(head, z) -> np.ndarray:
    head.eval()
    zt = torch.from_numpy(np.asarray(z, dtype=np.float32))
    out = []
    for i in range(0, len(zt), 256):
        out.append(head(zt[i:i + 256]).detach().cpu().numpy())
    return np.concatenate(out).astype(np.float64)