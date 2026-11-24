"""Generate MIDL figures from frozen splits, all_runs.csv, and models.py.

No retraining. No invented energy/CO2. Unmatched DenseNet classifier is
labelled as an upper bound, never as a fair baseline.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(r"E:\cvchestpj")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qcalibnet.metrics import expected_calibration_error, sigmoid
from qcalibnet.paths import FIGURES_DIR, TABLES_DIR, ensure_dirs

PAPER_FIG = ROOT / "paper" / "midl2027" / "figures"
SPLITS = ROOT / "data" / "splits"

PRIMARY = ["no_finding", "cardiomegaly", "pleural_effusion", "consolidation"]
HEADS = ["mlp", "product_pqc", "entangling_pqc"]
HEAD_LABEL = {
    "mlp": "MLP",
    "product_pqc": "Product PQC",
    "entangling_pqc": "Entangling PQC",
}
TASK_LABEL = {
    "no_finding": "No finding",
    "cardiomegaly": "Cardiomegaly",
    "pleural_effusion": "Pleural effusion",
    "consolidation": "Consolidation",
    "pneumonia_proxy": "Pneumonia proxy",
}
COLORS = {
    "mlp": "#2F5F9F",
    "product_pqc": "#6BA4D9",
    "entangling_pqc": "#3D9B8F",
}
HATCHES = {"mlp": "", "product_pqc": "///", "entangling_pqc": "xxx"}
EDGE = "#1F3A5F"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save(fig, name: str) -> None:
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for folder in (FIGURES_DIR, PAPER_FIG):
        png = folder / f"{name}.png"
        pdf = folder / f"{name}.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"wrote {name}")


def rounded(ax, xy, w, h, fc, ec="#333333", lw=0.9, r=0.08, z=2):
    p = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad=0.012,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(p)
    return p


def arrow(ax, x0, y0, x1, y1, color="#333333"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.15,
                        mutation_scale=9),
        zorder=4,
    )


def load_gray(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L")
    im.thumbnail((320, 320), Image.Resampling.LANCZOS)
    return np.asarray(im)


def pick_image(task: str, domain: str, label: int) -> Path | None:
    if domain == "chexpert":
        csv = SPLITS / task / "chexpert_test.csv"
    else:
        csv = SPLITS / task / "vindr_external_test.csv"
    df = pd.read_csv(csv)
    sub = df[df["label"] == label]
    for _, row in sub.iterrows():
        p = Path(str(row["image_path"]))
        if p.is_file() and p.stat().st_size > 1024:
            return p
    return None

