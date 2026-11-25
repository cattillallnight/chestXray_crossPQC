"""Corrected architecture candidate PNG. Matches qcalibnet/models.py."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parents[1] / "results" / "figures" / "QCalibNet_architecture_corrected.png"


def rbox(ax, x, y, w, h, fc, ec="#2b2b2b", lw=0.85, rad=0.06, z=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={rad}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(p)


def txt(ax, x, y, s, size=7.2, weight="medium", color="#1a1a1a", ha="center", va="center"):
    ax.text(
        x, y, s, fontsize=size, fontweight=weight, color=color,
        ha=ha, va=va, zorder=5, family="DejaVu Sans",
    )


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color="#333333", lw=1.05,
            mutation_scale=9, shrinkA=0, shrinkB=0,
        ),
        zorder=4,
    )


def cnot(ax, x, y_ctrl, y_tgt, r=0.052):
    ax.add_patch(Circle((x, y_ctrl), 0.038, facecolor="#111", edgecolor="#111", zorder=6))
    ax.add_line(Line2D([x, x], [y_ctrl, y_tgt], color="#111", lw=1.0, zorder=5))
    ax.add_patch(Circle((x, y_tgt), r, facecolor="white", edgecolor="#111", lw=1.05, zorder=6))
    ax.add_line(Line2D([x - r, x + r], [y_tgt, y_tgt], color="#111", lw=0.85, zorder=7))
    ax.add_line(Line2D([x, x], [y_tgt - r, y_tgt + r], color="#111", lw=0.85, zorder=7))


def main():
    fig, ax = plt.subplots(figsize=(14.2, 7.4), dpi=220)
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    rbox(ax, 0.10, 0.18, 2.58, 7.04, "#eef3f8", "#8fa0b0", 0.95)
    rbox(ax, 2.80, 0.18, 2.08, 7.04, "#eef6ee", "#86a086", 0.95)
    rbox(ax, 5.00, 0.18, 6.95, 7.04, "#f7f4ee", "#b0a07e", 0.95)
    rbox(ax, 12.08, 0.18, 2.00, 7.04, "#f8eeee", "#b08a8a", 0.95)

    txt(ax, 1.39, 6.92, "A. Input & backbone", 8.6, "bold")
    txt(ax, 3.84, 6.92, "B. Shared bottleneck", 8.6, "bold")
    txt(ax, 8.48, 6.92, "C. Three alternative heads  (one head per run)", 8.6, "bold")
    txt(ax, 13.08, 6.92, "D. Post-hoc T", 8.6, "bold")

    # A
    rbox(ax, 0.32, 5.70, 2.16, 0.88, "#ffffff")
    txt(ax, 1.40, 6.32, "CXR  224 × 224", 8.2, "bold")
    txt(ax, 1.40, 5.96, "frontal radiograph", 6.6, color="#444")
    arrow(ax, 1.40, 5.70, 1.40, 5.42)
    rbox(ax, 0.32, 4.62, 2.16, 0.78, "#ffffff")
    txt(ax, 1.40, 5.18, "RGB  →  ImageNet-norm", 7.1, "bold")
    txt(ax, 1.40, 4.86, "no CXR-specific pretrain", 6.3, color="#444")
    arrow(ax, 1.40, 4.62, 1.40, 4.32)
    rbox(ax, 0.32, 2.55, 2.16, 1.74, "#d7e5f4")
    txt(ax, 1.40, 3.95, "DenseNet-121", 8.4, "bold")
    txt(ax, 1.40, 3.58, "ImageNet, frozen", 7.0)
    txt(ax, 1.40, 3.22, "classifier → Identity", 6.6)
    txt(ax, 1.40, 2.82, r"$x \in \mathbb{R}^{1024}$", 8.2, "bold")
    txt(ax, 1.40, 1.95, "PQC never sees pixels.", 6.6, color="#444")

    # B
    rbox(ax, 2.98, 2.55, 1.74, 3.70, "#ffffff")
    txt(ax, 3.85, 5.92, r"$z=\pi\tanh(Wx+b)$", 7.3, "bold")
    txt(ax, 3.85, 5.55, r"$W\in\mathbb{R}^{6\times 1024}$", 6.7)
    for i in range(6):
        yy = 5.10 - i * 0.40
        rbox(ax, 3.32, yy - 0.15, 1.06, 0.30, "#d5ead5", rad=0.04)
        txt(ax, 3.85, yy, rf"$z_{i}$", 7.5, "bold")
    txt(ax, 3.85, 2.18, "one shared $z$", 6.9, "bold")
    txt(ax, 3.85, 1.86, "not detached", 6.4, color="#444")
    txt(ax, 3.85, 1.54, r"$z\in\mathbb{R}^{6}$", 7.3)

    arrow(ax, 2.48, 3.42, 2.98, 3.42)
    arrow(ax, 4.72, 4.90, 5.18, 5.72)
    arrow(ax, 4.72, 3.70, 5.18, 4.18)
    arrow(ax, 4.72, 2.90, 5.18, 2.15)

    # C1 MLP
    rbox(ax, 5.18, 5.28, 6.58, 1.28, "#fff4cc")
    txt(ax, 5.38, 6.28, "Head 1  ·  MLP    6,215 params (includes $W,b$)", 7.5, "bold", ha="left")
    mlp = [
        (5.32, "z (6)"),
        (6.28, "Linear 8"),
        (7.38, "tanh"),
        (8.28, "Drop 0.1"),
        (9.42, "Linear 1"),
        (10.52, r"logit$_{\mathrm{MLP}}$"),
    ]
    for x, lab in mlp:
        rbox(ax, x, 5.48, 0.88, 0.52, "#ffffff", rad=0.045)
        txt(ax, x + 0.44, 5.74, lab, 6.2)
    for a, b in [(6.20, 6.28), (7.16, 7.38), (8.26, 8.28), (9.16, 9.42), (10.30, 10.52)]:
        arrow(ax, a, 5.74, b, 5.74)

    # C2 product
    rbox(ax, 5.18, 3.32, 6.58, 1.84, "#dff0df")
    txt(ax, 5.38, 4.92, "Head 2  ·  Product PQC    6,188 params    no CNOT", 7.5, "bold", ha="left")
    wy = [4.58, 4.34, 4.10, 3.86, 3.62, 3.38]
    for i, y in enumerate(wy):
        ax.add_line(Line2D([5.32, 11.55], [y, y], color="#8a8a8a", lw=0.7, zorder=2))
        txt(ax, 5.48, y, rf"$q_{i}$", 6.0, ha="left")
        rbox(ax, 5.88, y - 0.10, 0.78, 0.20, "#cfe4f7", rad=0.03)
        txt(ax, 6.27, y, rf"RY$(z_{i})$", 5.7)
        rbox(ax, 6.78, y - 0.10, 0.92, 0.20, "#f3d9b8", rad=0.03)
        txt(ax, 7.24, y, "RX RY RZ", 5.4)
        rbox(ax, 7.82, y - 0.10, 0.92, 0.20, "#f3d9b8", rad=0.03)
        txt(ax, 8.28, y, "RX RY RZ", 5.4)
    rbox(ax, 8.88, wy[0] - 0.12, 0.78, 0.24, "#ffe08a", rad=0.03)
    txt(ax, 9.27, wy[0], r"$\langle Z_0\rangle$", 6.3, "bold")
    rbox(ax, 9.78, wy[0] - 0.12, 1.62, 0.24, "#ffffff", rad=0.03)
    txt(ax, 10.59, wy[0], r"scale $\cdot\langle Z_0\rangle+$ bias", 5.5)

    # C3 entangling
    rbox(ax, 5.18, 0.34, 6.58, 2.86, "#e3eaf8")
    txt(ax, 5.38, 2.96, "Head 3  ·  Entangling PQC    6,188 params    sequential CNOT ring × 2", 7.3, "bold", ha="left")
    ey = [2.62, 2.28, 1.94, 1.60, 1.26, 0.92]
    for i, y in enumerate(ey):
        ax.add_line(Line2D([5.32, 11.55], [y, y], color="#8a8a8a", lw=0.7, zorder=2))
        txt(ax, 5.48, y, rf"$q_{i}$", 6.0, ha="left")
        rbox(ax, 5.88, y - 0.11, 0.76, 0.22, "#cfe4f7", rad=0.03)
        txt(ax, 6.26, y, rf"RY$(z_{i})$", 5.6)
        rbox(ax, 6.74, y - 0.11, 0.80, 0.22, "#f3d9b8", rad=0.03)
        txt(ax, 7.14, y, "RX RY RZ", 5.2)

    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    xs1 = [7.68, 7.90, 8.12, 8.34, 8.56, 8.80]
    ax.add_patch(Rectangle((7.56, 0.78), 1.40, 2.00, fill=False, ls="--", ec="#6a7aa0", lw=0.75, zorder=3))
    for x, (a, b) in zip(xs1, pairs):
        cnot(ax, x, ey[a], ey[b])
    txt(ax, 8.26, 0.66, "ring 1:  0→1, 1→2, 2→3, 3→4, 4→5, 5→0", 5.3, color="#334")

    for i, y in enumerate(ey):
        rbox(ax, 9.10, y - 0.11, 0.80, 0.22, "#f3d9b8", rad=0.03)
        txt(ax, 9.50, y, "RX RY RZ", 5.2)

    xs2 = [10.06, 10.28, 10.50, 10.72, 10.94, 11.18]
    ax.add_patch(Rectangle((9.94, 0.78), 1.40, 2.00, fill=False, ls="--", ec="#6a7aa0", lw=0.75, zorder=3))
    for x, (a, b) in zip(xs2, pairs):
        cnot(ax, x, ey[a], ey[b])
    txt(ax, 10.64, 0.66, "ring 2:  same six CNOTs", 5.3, color="#334")

    rbox(ax, 10.55, ey[0] + 0.16, 0.92, 0.22, "#ffe08a", rad=0.03)
    txt(ax, 11.01, ey[0] + 0.27, r"$\langle Z_0\rangle$ + scale/bias", 5.4)

    txt(ax, 8.48, 0.44, "Each CNOT is one two-qubit gate (control • , target ⊕). Not a multi-target CNOT.", 5.6, color="#333")

    # D
    rbox(ax, 12.24, 4.55, 1.68, 1.85, "#ffffff")
    txt(ax, 13.08, 6.05, "per head / run", 6.8, "bold")
    txt(ax, 13.08, 5.68, "CheXpert val only", 6.4)
    txt(ax, 13.08, 5.28, r"$T=\exp(\theta)$", 7.4, "bold")
    txt(ax, 13.08, 4.88, r"$T\in[10^{-3},100]$", 6.6)
    arrow(ax, 13.08, 4.55, 13.08, 4.18)
    rbox(ax, 12.24, 3.28, 1.68, 0.86, "#f6d6d6")
    txt(ax, 13.08, 3.86, r"$p=\sigma(\mathrm{logit}/T)$", 6.8, "bold")
    txt(ax, 13.08, 3.50, "not joint across heads", 5.8, color="#444")
    rbox(ax, 12.28, 1.48, 1.60, 1.52, "#ffffff")
    txt(ax, 13.08, 2.68, r"$p_{\mathrm{MLP}}$", 7.2)
    txt(ax, 13.08, 2.22, r"$p_{\mathrm{Prod}}$", 7.2)
    txt(ax, 13.08, 1.76, r"$p_{\mathrm{Ent}}$", 7.2)
    txt(ax, 13.08, 1.05, "source-fitted $T$\nneed not transfer", 6.1, color="#444")
    arrow(ax, 11.76, 5.74, 12.24, 5.50)

    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    print("wrote", OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
