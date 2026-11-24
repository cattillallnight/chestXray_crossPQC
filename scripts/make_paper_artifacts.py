"""Paper tables and figures from all_runs.csv and dumped logits.

Does not retrain. Degenerate temperature (clamp 1e-3 or 100) is flagged, not dropped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qcalibnet.metrics import expected_calibration_error, sigmoid
from qcalibnet.paths import FIGURES_DIR, TABLES_DIR, ensure_dirs

PRIMARY = ["no_finding", "cardiomegaly", "pleural_effusion", "consolidation"]
HEADS = ["mlp", "product_pqc", "entangling_pqc"]
HEAD_LABEL = {"mlp": "MLP", "product_pqc": "Product PQC", "entangling_pqc": "Entangling PQC"}
TASK_LABEL = {
    "no_finding": "No finding",
    "cardiomegaly": "Cardiomegaly",
    "pleural_effusion": "Pleural effusion",
    "consolidation": "Consolidation",
    "pneumonia_proxy": "Pneumonia proxy (appendix)",
}


def degenerate_T(t: float) -> bool:
    return (t <= 1.001e-3) or (t >= 99.0)


def fmt(mean: float, std: float, digits: int = 3) -> str:
    if pd.isna(mean):
        return "—"
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def reliability_points(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    confs, accs, weights = [], [], []
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs <= hi if i == n_bins - 1 else probs < hi)
        if not np.any(mask):
            continue
        confs.append(float(probs[mask].mean()))
        accs.append(float(labels[mask].mean()))
        weights.append(float(mask.sum() / n))
    return np.asarray(confs), np.asarray(accs), np.asarray(weights)


def write_tables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["T_degenerate"] = df["temperature"].map(degenerate_T)
    df["ece_delta_chexpert"] = df["chexpert_test_cal_ece"] - df["chexpert_test_raw_ece"]
    df["ece_delta_vindr_bal"] = df["vindr_bal_cal_ece"] - df["vindr_bal_raw_ece"]
    df["auroc_shift_bal"] = df["vindr_bal_raw_auroc"] - df["chexpert_test_raw_auroc"]

    rows = []
    for task in PRIMARY + ["pneumonia_proxy"]:
        for head in HEADS:
            sub = df[(df["task"] == task) & (df["head"] == head)]
            if sub.empty:
                continue
            n_deg = int(sub["T_degenerate"].sum())
            rows.append(
                {
                    "task": task,
                    "head": head,
                    "n_seed": int(len(sub)),
                    "n_T_degenerate": n_deg,
                    "n_trainable": int(sub["n_trainable"].iloc[0]),
                    "epochs": fmt(sub.epochs_run.mean(), sub.epochs_run.std(), 1),
                    "T": fmt(sub.temperature.mean(), sub.temperature.std(), 2),
                    "chexpert_auroc": fmt(sub.chexpert_test_raw_auroc.mean(), sub.chexpert_test_raw_auroc.std()),
                    "chexpert_auprc": fmt(sub.chexpert_test_raw_auprc.mean(), sub.chexpert_test_raw_auprc.std()),
                    "chexpert_ece_raw": fmt(sub.chexpert_test_raw_ece.mean(), sub.chexpert_test_raw_ece.std()),
                    "chexpert_ece_cal": fmt(sub.chexpert_test_cal_ece.mean(), sub.chexpert_test_cal_ece.std()),
                    "chexpert_brier_raw": fmt(sub.chexpert_test_raw_brier.mean(), sub.chexpert_test_raw_brier.std()),
                    "vindr_bal_auroc": fmt(sub.vindr_bal_raw_auroc.mean(), sub.vindr_bal_raw_auroc.std()),
                    "vindr_bal_ece_raw": fmt(sub.vindr_bal_raw_ece.mean(), sub.vindr_bal_raw_ece.std()),
                    "vindr_bal_ece_cal": fmt(sub.vindr_bal_cal_ece.mean(), sub.vindr_bal_cal_ece.std()),
                    "vindr_nat_auroc": fmt(sub.vindr_nat_raw_auroc.mean(), sub.vindr_nat_raw_auroc.std()),
                    "vindr_nat_ece_raw": fmt(sub.vindr_nat_raw_ece.mean(), sub.vindr_nat_raw_ece.std()),
                    "vindr_nat_ece_cal": fmt(sub.vindr_nat_cal_ece.mean(), sub.vindr_nat_cal_ece.std()),
                    "delta_ece_chexpert": fmt(sub.ece_delta_chexpert.mean(), sub.ece_delta_chexpert.std()),
                    "delta_ece_vindr_bal": fmt(sub.ece_delta_vindr_bal.mean(), sub.ece_delta_vindr_bal.std()),
                    "delta_auroc_shift": fmt(sub.auroc_shift_bal.mean(), sub.auroc_shift_bal.std()),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "paper_primary_mean_std.csv", index=False)

    # Seed-wise paired deltas vs MLP (same task, same seed)
    paired = []
    for task in PRIMARY:
        for seed in (42, 43, 44):
            mlp = df[(df["task"] == task) & (df["head"] == "mlp") & (df["seed"] == seed)]
            if mlp.empty:
                continue
            m = mlp.iloc[0]
            for head in ("product_pqc", "entangling_pqc"):
                o = df[(df["task"] == task) & (df["head"] == head) & (df["seed"] == seed)]
                if o.empty:
                    continue
                r = o.iloc[0]
                paired.append(
                    {
                        "task": task,
                        "seed": seed,
                        "head": head,
                        "d_chexpert_auroc": r.chexpert_test_raw_auroc - m.chexpert_test_raw_auroc,
                        "d_chexpert_ece_raw": r.chexpert_test_raw_ece - m.chexpert_test_raw_ece,
                        "d_vindr_auroc": r.vindr_bal_raw_auroc - m.vindr_bal_raw_auroc,
                        "d_vindr_ece_raw": r.vindr_bal_raw_ece - m.vindr_bal_raw_ece,
                        "d_vindr_ece_cal": r.vindr_bal_cal_ece - m.vindr_bal_cal_ece,
                        "mlp_T": m.temperature,
                        "head_T": r.temperature,
                        "head_T_degenerate": bool(r.T_degenerate),
                    }
                )
    paired_df = pd.DataFrame(paired)
    paired_df.to_csv(TABLES_DIR / "paper_paired_vs_mlp.csv", index=False)

    deg = df[df.T_degenerate][["task", "head", "seed", "temperature", "chexpert_test_raw_ece", "chexpert_test_cal_ece", "vindr_bal_raw_ece", "vindr_bal_cal_ece"]]
    deg.to_csv(TABLES_DIR / "paper_degenerate_temperature.csv", index=False)

    md_lines = [
        "# Paper tables (generated; do not edit by hand)",
        "",
        "Mean±std over seeds {42,43,44}. Discrimination is secondary. "
        "Temperature scaling is fit on CheXpert validation logits and applied as logits/T. "
        "A run is T-degenerate if T hit the optimizer clamp (0.001 or 100).",
        "",
        "## Primary tasks",
        "",
        "| Task | Head | AUROC CheXpert | ECE raw→cal CheXpert | AUROC VinDr-bal | ECE raw→cal VinDr-bal | n T-deg |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in table.itertuples():
        if r.task == "pneumonia_proxy":
            continue
        md_lines.append(
            f"| {TASK_LABEL[r.task]} | {HEAD_LABEL[r.head]} | {r.chexpert_auroc} | "
            f"{r.chexpert_ece_raw}→{r.chexpert_ece_cal} | {r.vindr_bal_auroc} | "
            f"{r.vindr_bal_ece_raw}→{r.vindr_bal_ece_cal} | {r.n_T_degenerate} |"
        )
    md_lines += [
        "",
        "## Appendix: pneumonia_proxy (ontology shift; not a pneumonia detector)",
        "",
        "| Head | AUROC CheXpert | ECE raw→cal CheXpert | AUROC VinDr-bal | ECE raw→cal VinDr-bal | n T-deg |",
        "|---|---|---|---|---|---|",
    ]
    for r in table.itertuples():
        if r.task != "pneumonia_proxy":
            continue
        md_lines.append(
            f"| {HEAD_LABEL[r.head]} | {r.chexpert_auroc} | {r.chexpert_ece_raw}→{r.chexpert_ece_cal} | "
            f"{r.vindr_bal_auroc} | {r.vindr_bal_ece_raw}→{r.vindr_bal_ece_cal} | {r.n_T_degenerate} |"
        )
    (TABLES_DIR / "paper_tables.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"wrote {TABLES_DIR / 'paper_primary_mean_std.csv'}")
    print(f"T-degenerate runs: {int(df.T_degenerate.sum())}")
    return df


def plot_metric_bars(df: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.size": 0,
        "ytick.major.size": 3,
    })
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 4.85), constrained_layout=True)
    x = np.arange(len(PRIMARY))
    width = 0.24
    colors = {"mlp": "#2F5F9F", "product_pqc": "#6BA4D9", "entangling_pqc": "#3D9B8F"}
    hatches = {"mlp": "", "product_pqc": "///", "entangling_pqc": "xxx"}
    edge = "#1F3A5F"
    specs = [
        (axes[0, 0], "chexpert_test_raw_auroc", "CheXpert test AUROC", 0.50, 0.95),
        (axes[0, 1], "vindr_bal_raw_auroc", "VinDr-CXR balanced AUROC", 0.50, 0.95),
        (axes[1, 0], "chexpert_test_raw_ece", "CheXpert test ECE (raw)", 0.00, 0.55),
        (axes[1, 1], "vindr_bal_raw_ece", "VinDr-CXR balanced ECE (raw)", 0.00, 0.55),
    ]
    handles = []
    for ax, col, title, ymin, ymax in specs:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="0.85", lw=0.6)
        for i, head in enumerate(HEADS):
            means, stds = [], []
            for task in PRIMARY:
                sub = df[(df["task"] == task) & (df["head"] == head)][col]
                means.append(float(sub.mean()))
                stds.append(float(sub.std(ddof=1)) if len(sub) > 1 else 0.0)
            xpos = x + (i - 1) * width
            bars = ax.bar(
                xpos,
                means,
                width,
                yerr=stds,
                capsize=2.2,
                error_kw={"elinewidth": 0.7, "ecolor": "0.25"},
                color=colors[head],
                edgecolor=edge,
                linewidth=0.6,
                hatch=hatches[head],
                label=HEAD_LABEL[head],
                zorder=3,
            )
            if ax is axes[0, 0]:
                handles.append(bars)
            rotate = 90 if "auroc" in col else 0
            for xi, mu, sd in zip(xpos, means, stds):
                ax.text(
                    xi,
                    mu + sd + (0.012 if rotate else 0.008),
                    f"{mu:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.8,
                    color="0.12",
                    rotation=rotate,
                    clip_on=False,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(["No finding", "Cardiomegaly", "Pleural\neffusion", "Consolidation"], fontsize=7.5)
        ax.set_title(title, fontsize=9, pad=3)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis="y", labelsize=7)
        if ax in (axes[0, 0], axes[1, 0]):
            ax.set_ylabel("Score", fontsize=8)
    fig.legend(
        handles,
        [HEAD_LABEL[h] for h in HEADS],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
        fontsize=8,
        handlelength=1.4,
        columnspacing=1.4,
    )
    path = FIGURES_DIR / "fig_auroc_ece_bars.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_auroc_ece_bars.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_ts_transfer(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.6), constrained_layout=True)
    markers = {"mlp": "o", "product_pqc": "s", "entangling_pqc": "^"}
    colors = {"mlp": "#4C72B0", "product_pqc": "#DD8452", "entangling_pqc": "#55A868"}
    for head in HEADS:
        sub = df[df["head"].eq(head) & df["task"].isin(PRIMARY)]
        ax.scatter(
            sub["ece_delta_chexpert"],
            sub["ece_delta_vindr_bal"],
            marker=markers[head],
            c=colors[head],
            s=42,
            label=HEAD_LABEL[head],
            alpha=0.85,
            edgecolors="k",
            linewidths=0.3,
        )
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("ΔECE CheXpert test (cal − raw)")
    ax.set_ylabel("ΔECE VinDr balanced (cal − raw)")
    ax.set_title("Source-fitted temperature does not reliably transfer")
    ax.legend(frameon=False)
    path = FIGURES_DIR / "fig_temperature_transfer.png"
    fig.savefig(path, dpi=300)
    fig.savefig(FIGURES_DIR / "fig_temperature_transfer.pdf")
    plt.close(fig)
    print(f"wrote {path}")


def plot_reliability_task(task: str, seed: int = 42) -> None:
    logit_dir = FIGURES_DIR / "logits"
    split_row = ["chexpert_test", "vindr_bal"]
    split_title = {"chexpert_test": "CheXpert test", "vindr_bal": "VinDr balanced"}
    needed = [logit_dir / f"{task}_{head}_seed{seed}_{split}.npz" for head in HEADS for split in split_row]
    if not any(p.is_file() for p in needed):
        print(f"no logits for {task} seed {seed} — skip reliability figure")
        return
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.6), constrained_layout=True)
    for r, split in enumerate(split_row):
        for c, head in enumerate(HEADS):
            ax = axes[r, c]
            path = logit_dir / f"{task}_{head}_seed{seed}_{split}.npz"
            ax.plot([0, 1], [0, 1], ls="--", c="0.6", lw=0.9)
            if not path.is_file():
                ax.set_title(f"{HEAD_LABEL[head]} / {split_title[split]}\n(missing logits)")
                continue
            blob = np.load(path, allow_pickle=True)
            logits = blob["logits"].astype(np.float64)
            labels = blob["labels"].astype(np.float64)
            T = float(blob["temperature"][0])
            raw_p = sigmoid(logits)
            cal_p = sigmoid(logits / T)
            for probs, name, color in ((raw_p, "raw", "#4C72B0"), (cal_p, f"T={T:.2g}", "#DD8452")):
                confs, accs, w = reliability_points(probs, labels)
                ece, _ = expected_calibration_error(probs, labels, n_bins=15)
                ax.scatter(confs, accs, s=40 + 180 * w, c=color, alpha=0.85, label=f"{name} ECE={ece:.3f}", zorder=3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_title(f"{HEAD_LABEL[head]} · {split_title[split]}", fontsize=8, pad=2)
            ax.tick_params(labelsize=6.5)
            ax.legend(frameon=False, fontsize=6.2, loc="upper left", handletextpad=0.3, borderpad=0.2)
            if r == 1:
                ax.set_xlabel("Confidence")
            if c == 0:
                ax.set_ylabel("Accuracy")
    fig.suptitle(
        f"Reliability diagrams, {TASK_LABEL[task]}, seed {seed} (bin size ∝ mass)",
        fontsize=11,
    )
    out = FIGURES_DIR / f"fig_reliability_{task}_seed{seed}.png"
    fig.savefig(out, dpi=300)
    fig.savefig(FIGURES_DIR / f"fig_reliability_{task}_seed{seed}.pdf")
    plt.close(fig)
    print(f"wrote {out}")


def plot_reliability() -> None:
    for task in PRIMARY + ["pneumonia_proxy"]:
        plot_reliability_task(task, seed=42)


def main() -> int:
    ensure_dirs()
    raw = pd.read_csv(TABLES_DIR / "all_runs.csv")
    df = write_tables(raw)
    plot_metric_bars(df)
    plot_ts_transfer(df)
    plot_reliability()
    status = {
        "n_runs": int(len(df)),
        "n_vindr": int(df["vindr_bal_raw_auroc"].notna().sum()),
        "n_T_degenerate": int(df["temperature"].map(degenerate_T).sum()),
        "figures": sorted(p.name for p in FIGURES_DIR.glob("fig_*")),
    }
    (TABLES_DIR / "paper_artifact_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
