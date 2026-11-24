"""Regenerate submission figures in one clean visual system.

Style reference: accepted-MIDL practice (rounded pastel panels, thin dark
borders, left-to-right flow, grouped column headers, no top/right spines,
sans-serif, generous whitespace). Reads frozen artefacts only; never retrains.

Figures
  fig_pipeline            trunk -> 6-d bottleneck -> three heads -> logit -> T
  fig_circuits            product vs entangling, readout drawn on q0 only
  fig_main_results        AUROC and ECE grouped bars, four primary tasks
  fig_calibration_transfer  dECE source vs target after source-fitted T
  fig_reliability         reliability, no finding seed 42, three heads
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
PAPER_FIG = ROOT / "paper" / "midl2027" / "figures"
LOGITS = FIG / "logits"
STATS = FIG / "stats_from_runs.json"

INK = "#1F2933"
MUTED = "#52606D"
GRID = "#CBD2D9"
PANEL = {
    "trunk": "#E6EBF0",
    "bottleneck": "#DCE7F5",
    "mlp": "#FBE3CC",
    "product": "#DDEEDD",
    "entangle": "#C6E3D1",
    "readout": "#FBF3C9",
    "calib": "#F1DFEC",
}
HEAD_COLOR = {"mlp": "#D9822B", "product_pqc": "#4E9A51", "entangling_pqc": "#2F7E63"}
HEAD_LABEL = {"mlp": "MLP", "product_pqc": "Product PQC", "entangling_pqc": "Entangling PQC"}
TASK_LABEL = {
    "no_finding": "No finding",
    "cardiomegaly": "Cardiomegaly",
    "pleural_effusion": "Pleural effusion",
    "consolidation": "Consolidation",
}
PRIMARY = ["no_finding", "cardiomegaly", "pleural_effusion", "consolidation"]


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.linewidth": 0.9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save(fig, name: str) -> None:
    for folder in (FIG, PAPER_FIG):
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", facecolor="white")
        fig.savefig(folder / f"{name}.png", bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    print(f"wrote {name}")


def box(ax, x, y, w, h, text, fc, fs=8.0, weight="normal", r=0.06, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.004,rounding_size={r}",
                                facecolor=fc, edgecolor=INK, linewidth=0.9, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=tc, zorder=3, linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, lw=0.95, ms=8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=ms, lw=lw, color=INK,
                                 shrinkA=1.5, shrinkB=1.5, zorder=1))


def header(ax, x0, x1, label, y):
    ax.text((x0 + x1) / 2, y, label, ha="center", va="bottom",
            fontsize=7.2, fontweight="bold", color=MUTED)


# ---------------------------------------------------------------------------
def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.05))
    ax.set_xlim(0, 15.4)
    ax.set_ylim(1.15, 5.35)
    ax.axis("off")

    # heads stacked with clear gaps, group centred on the trunk boxes
    hh, gap = 0.74, 0.20
    g_bottom = 1.45
    ys = {
        "entangling_pqc": g_bottom,
        "product_pqc": g_bottom + hh + gap,
        "mlp": g_bottom + 2 * (hh + gap),
    }
    g_top = ys["mlp"] + hh
    ymid = (g_bottom + g_top) / 2
    bh = 1.10
    yb = ymid - bh / 2

    header(ax, 0.20, 2.05, "INPUT", g_top + 0.28)
    header(ax, 2.65, 4.75, "FROZEN TRUNK", g_top + 0.28)
    header(ax, 5.35, 6.95, "BOTTLENECK", g_top + 0.28)
    header(ax, 7.85, 11.35, "MATCHED HEADS", g_top + 0.28)
    header(ax, 11.95, 12.85, "LOGIT", g_top + 0.28)
    header(ax, 13.45, 15.15, "CALIBRATION", g_top + 0.28)

    box(ax, 0.20, yb, 1.85, bh, "CXR\n224×224 RGB\nImageNet-norm", PANEL["trunk"], 7.3)
    box(ax, 2.65, yb, 2.10, bh, "DenseNet-121\nfrozen backbone\n$x\\in\\mathbb{R}^{1024}$", PANEL["trunk"], 7.3)
    box(ax, 5.35, yb, 1.60, bh, "$z=\\pi\\tanh(Wx{+}b)$\n$z\\in\\mathbb{R}^{6}$", PANEL["bottleneck"], 7.3)
    arrow(ax, 2.05, ymid, 2.65, ymid)
    arrow(ax, 4.75, ymid, 5.35, ymid)

    hx, hw = 7.85, 3.50
    box(ax, hx, ys["mlp"], hw, hh, "MLP   $6{\\to}8{\\to}\\tanh{\\to}1$\n6,215 params", PANEL["mlp"], 7.2)
    box(ax, hx, ys["product_pqc"], hw, hh, "Product PQC   no CNOT\n6,188 nominal params", PANEL["product"], 7.2)
    box(ax, hx, ys["entangling_pqc"], hw, hh, "Entangling PQC   CNOT ring\n6,188 nominal params", PANEL["entangle"], 7.2)

    jx = 7.45
    ax.plot([6.95, jx], [ymid, ymid], color=INK, lw=0.95, zorder=1)
    for k in ys:
        arrow(ax, jx, ymid, hx, ys[k] + hh / 2)

    rx = 11.90
    for k in ys:
        ax.plot([hx + hw, rx], [ys[k] + hh / 2, ys[k] + hh / 2], color=INK, lw=0.95, zorder=1)
    ax.plot([rx, rx], [ys["entangling_pqc"] + hh / 2, ys["mlp"] + hh / 2], color=INK, lw=0.95, zorder=1)
    box(ax, 12.05, ymid - 0.42, 0.78, 0.84, "logit", PANEL["readout"], 7.4)
    arrow(ax, rx, ymid, 12.05, ymid)

    box(ax, 13.45, ymid - 0.58, 1.70, 1.16, "$p=\\sigma(z/T)$\n$T$ fit on\nCheXpert val\nper head", PANEL["calib"], 7.0)
    arrow(ax, 12.83, ymid, 13.45, ymid)

    ax.text(0.20, 1.22, "Shared 6-d bottleneck: only the head changes between runs.",
            ha="left", va="bottom", fontsize=6.9, style="italic", color=MUTED)
    save(fig, "fig_pipeline")


# ---------------------------------------------------------------------------
def _wire(ax, x0, x1, y, q):
    ax.plot([x0, x1], [y, y], color="#7B8794", lw=0.9, zorder=1)
    ax.text(x0 - 0.14, y, f"$q_{q}$", ha="right", va="center", fontsize=6.8, color=INK)


def _embed(ax, x0, ys, w=0.95):
    for q, y in ys.items():
        ax.add_patch(Rectangle((x0, y - 0.26), w, 0.52, facecolor=PANEL["bottleneck"],
                               edgecolor=INK, lw=0.8, zorder=2))
        ax.text(x0 + w / 2, y, f"$RY(z_{q})$", ha="center", va="center", fontsize=5.8, zorder=3)


def _rot(ax, x0, ys, w=1.30):
    for q, y in ys.items():
        ax.add_patch(Rectangle((x0, y - 0.26), w, 0.52, facecolor="#FCE3C8",
                               edgecolor=INK, lw=0.8, zorder=2))
        ax.text(x0 + w / 2, y, "RX RY RZ", ha="center", va="center", fontsize=5.6, zorder=3)


def _cnot(ax, x, yc, yt):
    ax.plot([x, x], [yc, yt], color=INK, lw=0.85, zorder=3)
    ax.add_patch(Circle((x, yc), 0.05, facecolor=INK, edgecolor=INK, zorder=4))
    ax.add_patch(Circle((x, yt), 0.10, facecolor="white", edgecolor=INK, lw=0.85, zorder=4))
    ax.plot([x - 0.10, x + 0.10], [yt, yt], color=INK, lw=0.85, zorder=4)
    ax.plot([x, x], [yt - 0.10, yt + 0.10], color=INK, lw=0.85, zorder=4)


def fig_circuits() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(-0.95, 15.9)
    ax.set_ylim(-0.55, 7.05)
    ax.axis("off")
    ys = {q: 5 - q for q in range(6)}

    # (a) product
    ax.text(0.0, 6.75, "(a) Product PQC", ha="left", va="center",
            fontsize=7.6, fontweight="bold", color=INK)
    ax.text(0.0, 6.35, "no entangling gates; readout $\\langle Z_0\\rangle$ sees $q_0$ only",
            ha="left", va="center", fontsize=6.4, style="italic", color=MUTED)
    for q, y in ys.items():
        _wire(ax, 0.0, 6.55, y, q)
    _embed(ax, 0.55, ys)
    _rot(ax, 1.75, ys)
    _rot(ax, 3.15, ys)
    ax.add_patch(Rectangle((4.65, ys[0] - 0.26), 1.35, 0.52, facecolor=PANEL["readout"],
                           edgecolor=INK, lw=0.85, zorder=4))
    ax.text(5.32, ys[0], "$\\langle Z_0\\rangle$", ha="center", va="center", fontsize=6.4, zorder=5)
    ax.annotate("", xy=(6.45, ys[0]), xytext=(6.05, ys[0]),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9))
    ax.text(6.55, (ys[5] + ys[1]) / 2, "$z_{1..5}$ never\nreach $\\langle Z_0\\rangle$",
            ha="left", va="center", fontsize=6.1, style="italic", color="#B45309")

    # (b) entangling
    sh = 9.0
    ax.text(sh, 6.75, "(b) Entangling PQC", ha="left", va="center",
            fontsize=7.6, fontweight="bold", color=INK)
    ax.text(sh, 6.35, "CNOT ring couples all six $z_i$ into $q_0$",
            ha="left", va="center", fontsize=6.4, style="italic", color=MUTED)
    for q, y in ys.items():
        _wire(ax, sh, sh + 6.55, y, q)
    _embed(ax, sh + 0.55, ys)
    _rot(ax, sh + 1.75, ys)
    for i in range(5):
        _cnot(ax, sh + 3.22, ys[i], ys[i + 1])
    _cnot(ax, sh + 3.62, ys[5], ys[0])
    _rot(ax, sh + 3.85, ys)
    for i in range(5):
        _cnot(ax, sh + 5.42, ys[i], ys[i + 1])
    _cnot(ax, sh + 5.82, ys[5], ys[0])
    ax.add_patch(Rectangle((sh + 6.02, ys[0] - 0.26), 0.72, 0.52, facecolor=PANEL["readout"],
                           edgecolor=INK, lw=0.85, zorder=4))
    ax.text(sh + 6.38, ys[0], "$\\langle Z_0\\rangle$", ha="center", va="center", fontsize=5.9, zorder=5)

    save(fig, "fig_circuits")


# ---------------------------------------------------------------------------
def load_stats() -> dict:
    return json.loads(STATS.read_text())


def fig_main_results(stats: dict) -> None:
    heads = ["mlp", "product_pqc", "entangling_pqc"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    panels = [
        ("auroc_vbal", "AUROC on VinDr-CXR (balanced)", (0.50, 0.85)),
        ("ece_vbal_raw", "Raw ECE on VinDr-CXR (balanced)", (0.0, 0.30)),
    ]
    x = np.arange(len(PRIMARY))
    w = 0.26
    for ax, (key, title, (lo, hi)) in zip(axes, panels):
        for j, h in enumerate(heads):
            means = [stats[t][h][key][0] for t in PRIMARY]
            stds = [stats[t][h][key][1] for t in PRIMARY]
            ax.bar(x + (j - 1) * w, means, w, yerr=stds, capsize=2.2,
                   color=HEAD_COLOR[h], edgecolor="white", linewidth=0.6,
                   error_kw=dict(elinewidth=0.7, ecolor=MUTED, capthick=0.7),
                   label=HEAD_LABEL[h], zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABEL[t].replace(" ", "\n") for t in PRIMARY], fontsize=7.0)
        ax.set_ylim(lo, hi)
        ax.set_ylabel(title.split(" on ")[0], fontsize=7.6)
        ax.set_title(title, fontsize=7.7, fontweight="bold", color=INK, pad=7)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=6.9)
        ax.spines["left"].set_color(INK)
    axes[0].legend(frameon=False, fontsize=6.8, ncol=3, loc="upper center",
                   bbox_to_anchor=(1.06, 1.17), handlelength=1.1, columnspacing=1.1)
    fig.text(0.5, -0.03, "Mean over three seeds; whiskers = std. Lower ECE is better.",
             ha="center", fontsize=6.7, style="italic", color=MUTED)
    save(fig, "fig_main_results")


# ---------------------------------------------------------------------------
def fig_transfer(stats: dict) -> None:
    heads = ["mlp", "product_pqc", "entangling_pqc"]
    markers = {"mlp": "o", "product_pqc": "s", "entangling_pqc": "^"}
    fig, ax = plt.subplots(figsize=(3.3, 3.15))
    for h in heads:
        for t in PRIMARY:
            src = stats[t][h]["ece_chex_cal"][0] - stats[t][h]["ece_chex_raw"][0]
            tgt = stats[t][h]["ece_vbal_cal"][0] - stats[t][h]["ece_vbal_raw"][0]
            ax.scatter(src, tgt, s=44, marker=markers[h], color=HEAD_COLOR[h],
                       edgecolor=INK, linewidth=0.5, alpha=0.9, zorder=3)
    ax.axhline(0, color="#7B8794", lw=0.8, ls="--")
    ax.axvline(0, color="#7B8794", lw=0.8, ls="--")
    ax.set_xlabel("$\\Delta$ECE CheXpert  (corrected$-$raw)", fontsize=7.4)
    ax.set_ylabel("$\\Delta$ECE VinDr  (corrected$-$raw)", fontsize=7.4)
    ax.tick_params(labelsize=6.9)
    ax.grid(color=GRID, lw=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.03, 0.97, "source improves,\ntarget worsens", transform=ax.transAxes,
            fontsize=6.2, va="top", ha="left", style="italic", color="#B45309")
    ax.text(0.97, 0.03, "both improve", transform=ax.transAxes,
            fontsize=6.2, va="bottom", ha="right", style="italic", color="#2F7E63")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=markers[h], color="w", markerfacecolor=HEAD_COLOR[h],
                      markeredgecolor=INK, markersize=6, label=HEAD_LABEL[h]) for h in heads]
    ax.legend(handles=handles, frameon=False, fontsize=6.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.24), ncol=3, handletextpad=0.3, columnspacing=0.9)
    save(fig, "fig_calibration_transfer")


# ---------------------------------------------------------------------------
def _ece(probs, labels, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    e, n = 0.0, len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (probs >= lo) & (probs <= hi) if i == n_bins - 1 else (probs >= lo) & (probs < hi)
        if m.any():
            e += abs(labels[m].mean() - probs[m].mean()) * m.sum() / n
    return float(e)


def _binned(probs, labels, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys, ms = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (probs >= lo) & (probs <= hi) if i == n_bins - 1 else (probs >= lo) & (probs < hi)
        if m.any():
            xs.append(probs[m].mean()); ys.append(labels[m].mean()); ms.append(m.sum())
    return np.array(xs), np.array(ys), np.array(ms)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -80, 80)))


def _reliability_panel(task: str, title: str, out: str) -> None:
    heads = ["mlp", "product_pqc", "entangling_pqc"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    for ax, h in zip(axes, heads):
        blob = np.load(LOGITS / f"{task}_{h}_seed42_chexpert_test.npz", allow_pickle=True)
        logits = blob["logits"].astype(np.float64)
        labels = blob["labels"].astype(np.float64)
        T = float(blob["temperature"][0])
        ax.plot([0, 1], [0, 1], color="#7B8794", lw=0.9, ls="--", zorder=1)
        for probs, color, lab in ((sigmoid(logits), HEAD_COLOR[h], "raw"),
                                  (sigmoid(logits / T), MUTED, "$T$-scaled")):
            xs, ys, ms = _binned(probs, labels)
            e = _ece(probs, labels)
            ax.plot(xs, ys, color=color, lw=1.25, zorder=3, label=f"{lab}  ECE={e:.3f}")
            ax.scatter(xs, ys, s=np.sqrt(ms) * 6 + 6, color=color, edgecolor="white",
                       linewidth=0.5, zorder=4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_title(HEAD_LABEL[h], fontsize=7.6, fontweight="bold", color=INK)
        ax.set_xlabel("predicted probability", fontsize=7.0)
        if h == heads[0]:
            ax.set_ylabel("empirical frequency", fontsize=7.0)
        ax.tick_params(labelsize=6.5)
        ax.grid(color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=6.0, loc="upper left", handlelength=1.0)
    fig.suptitle(f"{title} · CheXpert test · seed 42 · marker area $\\propto$ bin mass",
                 fontsize=7.6, y=1.04, color=INK)
    save(fig, out)


def fig_reliability() -> None:
    _reliability_panel("no_finding", "No finding", "fig_reliability")
    _reliability_panel("cardiomegaly", "Cardiomegaly", "fig_reliability_cardiomegaly_seed42")
    _reliability_panel("pleural_effusion", "Pleural effusion", "fig_reliability_pleural_effusion_seed42")
    _reliability_panel("consolidation", "Consolidation", "fig_reliability_consolidation_seed42")
    _reliability_panel("pneumonia_proxy", "Pneumonia proxy", "fig_reliability_pneumonia_proxy_seed42")


def main() -> None:
    style()
    stats = load_stats()
    fig_pipeline()
    fig_circuits()
    fig_main_results(stats)
    fig_transfer(stats)
    fig_reliability()


if __name__ == "__main__":
    main()