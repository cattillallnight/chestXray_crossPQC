"""Single resumable entry point for Stages 1-3.

Usage
  python run_all.py --stage 1
  python run_all.py --stage 2 --budget-hours 6
  python run_all.py --all --budget-hours 6

Resumable: every unit of work is keyed by a config hash under
results/v2/state/ and skipped if already done. Safe to re-launch after any
interruption. Logs to results/v2/run_all.log.

HALT conditions write results/v2/HALT.md and exit non-zero.
WARN conditions append to results/v2/ANOMALIES.md and continue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from qcv2 import metrics as M
from qcv2 import data as D
from qcv2.configs import all_configs, cfg_hash, write_frozen_configs, primary_configs
from qcv2.paths import (CSV, FIG, LOGS, PRIMARY_TASKS, V2, V1_HEADS, V1_SEEDS,
                        ensure_dirs)
from qcv2.state import (Halt, RunStore, check_split_hashes, deviation, git_hash,
                        log, read_csv, warn, write_csv, write_halt, now)
from qcv2.tables import table_from_rows


# ---------------------------------------------------------------------------
def _run_tests() -> None:
    """Run the unit tests; any failure is a HALT."""
    import subprocess
    py = sys.executable
    r = subprocess.run([py, "-I", str(ROOT / "tests" / "test_v2.py")],
                       capture_output=True, text=True)
    tail = (r.stdout or "") + (r.stderr or "")
    log("unit tests:\n" + tail.strip())
    if r.returncode != 0:
        raise Halt("unit tests failed:\n" + tail.strip())


def _env_txt() -> None:
    import platform
    lines = [f"python {sys.version.split()[0]}", f"platform {platform.platform()}"]
    for mod in ("numpy", "torch", "pennylane", "sklearn", "pandas", "matplotlib", "scipy"):
        try:
            m = __import__(mod)
            lines.append(f"{mod} {getattr(m, '__version__', '?')}")
        except Exception:  # noqa: BLE001
            lines.append(f"{mod} MISSING")
    lines.append(f"git {git_hash()}")
    lines.append(f"generated {now()}")
    (V2 / "env.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
def stage1(budget_ok=True) -> dict:
    log("=" * 60)
    log("STAGE 1 start (post-hoc on v1 logits, exploratory)")
    from qcv2.stage1 import run_stage1
    from qcv2.stage1_extra import stage1_4, stage1_5
    from qcv2.dump_v1_extra import main as dump_extra

    # ensure val + natural logits exist (needed for 1.2 and 1.5)
    store = RunStore("stage1_dump")
    if not store.is_done({"extra_logits": True}):
        dump_extra()
        store.mark_done({"extra_logits": True})

    res = run_stage1(D)
    rows = res["rows"]
    # swap in extended val/nat logits for the data loader via a shim
    rows += stage1_4(D)
    rows += stage1_5(D)
    write_csv(CSV / "stage1_long.csv", rows)
    log(f"Stage 1 wrote {len(rows)} rows to csv/stage1_long.csv")

    # figures F1-F4 + reliability
    from qcv2.figures import fig_F1_prev_sweep, fig_F2_recal_ladder, fig_reliability
    from qcv2.figs_from_csv import build_F3, build_F4
    try:
        fig_F1_prev_sweep(rows)
        fig_F2_recal_ladder(rows)
        log("Stage 1 figures F1, F2 written")
    except Exception as exc:  # noqa: BLE001
        warn(f"Stage 1 figure failed: {exc}", "figure_failure")
    write_csv(CSV / "stage1_long.csv", rows)
    try:
        n3 = build_F3(); n4 = build_F4()
        log(f"figures F3 ({n3} pts), F4 ({n4} pts) written")
    except Exception as exc:  # noqa: BLE001
        warn(f"F3/F4 figure failed: {exc}", "figure_failure")
    # reliability diagrams from v1 logits, seed 42, primary tasks
    try:
        for task in PRIMARY_TASKS:
            for head in V1_HEADS:
                got = D.load_v1_logits(task, head, 42, "chexpert_test")
                if got is None:
                    continue
                z, y, _ = got
                fig_reliability(task, head, 42, M.sigmoid(z), y)
        log("reliability diagrams written")
    except Exception as exc:  # noqa: BLE001
        warn(f"reliability figure failed: {exc}", "figure_failure")

    # ranking-flip counts across prevalence (1.1)
    flips = _ranking_flips(rows)
    log(f"ranking-flip summary: {flips['n_with_any_flip']}/{flips['n_task_seed']} task-seeds "
        f"change head order across prevalence (detail in stage1_ranking_flips.json)")
    (V2 / "stage1_ranking_flips.json").write_text(json.dumps(flips, indent=2), encoding="utf-8")

    # tables (with/without collapsed not applicable to v1 here)
    table_from_rows(rows, ["auroc", "ece_pos", "ece_top", "brier", "nll"],
                    "stage1_metrics")
    return {"rows": len(rows), "flips": flips}


def _ranking_flips(rows):
    """How often the head ranking by ECE flips across prevalence."""
    from collections import defaultdict
    by = defaultdict(dict)
    for r in rows:
        if r["stage"] == "1.1" and r["metric"] == "ece_pos" and not r["head"].startswith("null"):
            by[(r["task"], r["seed"])].setdefault(float(r["prevalence"]), {})[r["head"]] = float(r["estimate"])
    flips = {"n_task_seed": 0, "n_with_any_flip": 0, "detail": []}
    for key, prevmap in by.items():
        rankings = []
        for pi in sorted(prevmap):
            order = tuple(sorted(prevmap[pi], key=lambda h: prevmap[pi][h]))
            rankings.append((pi, order))
        flips["n_task_seed"] += 1
        uniq = set(o for _, o in rankings)
        if len(uniq) > 1:
            flips["n_with_any_flip"] += 1
            flips["detail"].append({"task_seed": list(key),
                                    "rankings": [{"prevalence": p, "order": list(o)} for p, o in rankings]})
    return flips


# ---------------------------------------------------------------------------
def _benchmark_eta(groups: dict) -> float:
    """Time one run per head type, extrapolate total hours."""
    from qcv2.configs import COMMON
    from qcv2.stage2_data import fit_bottleneck, make_split
    from qcv2.train import train_head
    bn = fit_bottleneck("no_finding")
    ztr, ytr, _ = make_split("no_finding", "chexpert_train.csv", bn)
    zva, yva, _ = make_split("no_finding", "chexpert_val.csv", bn)
    per_head = {}
    for head in ("mlp", "pqc_none_z0", "pqc_ring_z0", "pqc_ring_linear", "trig_pair", "logistic"):
        cfg = dict(COMMON); cfg.update({"head": head, "task": "no_finding", "seed": 42,
                                        "epochs": 5, "patience": 2})
        t = time.time(); train_head(cfg, ztr, ytr, zva, yva, 42)
        per_head[head] = (time.time() - t) / 5.0   # seconds per epoch
    # extrapolate: assume ~60 epochs avg
    EP = 60
    total_s = 0.0
    for group, cfgs in groups.items():
        for c in cfgs:
            s = per_head.get(c["head"], 0.1) * EP
            total_s += s
    hours = total_s / 3600.0
    log(f"ETA benchmark per-epoch seconds: {per_head}")
    log(f"ETA for full requested grid (~{EP} ep/run): {hours:.2f} h")
    return hours


def stage2(budget_hours: float | None = None) -> dict:
    log("=" * 60)
    log("STAGE 2 start (corrected protocol)")
    counts = write_frozen_configs()
    log(f"frozen configs written: {counts}")
    frozen = json.loads((V2 / "frozen_configs.json").read_text(encoding="utf-8"))
    frozen_hash = hashlib.sha256((V2 / "frozen_configs.json").read_bytes()).hexdigest()[:16]
    (V2 / "STAGE2_FROZEN.md").write_text(
        f"# Stage 2 configs frozen\n\ngit: {git_hash()}\n"
        f"frozen_configs sha256[:16]: {frozen_hash}\ntime: {now()}\ncounts: {counts}\n",
        encoding="utf-8")
    log(f"Stage 2 configs frozen at hash {frozen_hash}; commit this before running.")

    groups = {k: [x["cfg"] for x in v] for k, v in frozen.items()}
    if budget_hours is None:
        budget_hours = 1e9
    eta = _benchmark_eta(groups)
    if eta > budget_hours:
        drop_order = ["depth_ntrain", "stage3", "prior_ablation"]
        for d in drop_order:
            if eta <= budget_hours:
                break
            key = "depth_ntrain" if d == "depth_ntrain" else ("prior_ablation" if d == "prior_ablation" else None)
            if key and key in groups:
                log(f"budget: dropping {key} (ETA {eta:.2f}h > budget {budget_hours}h)")
                deviation(f"dropped {key} due to budget")
                groups.pop(key)
                eta = _benchmark_eta(groups)

    from qcv2.stage2 import evaluate_run, run_one
    crashes = 0
    bench_store = RunStore("stage2_runs")
    # per-run metric files: crash-safe, resumable, deduped by config hash
    metric_dir = V2 / "state" / "stage2_metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)

    def _metric_path(cfg):
        return metric_dir / f"{cfg_hash(cfg)}.json"

    n_done = 0
    for group, cfgs in groups.items():
        log(f"  group {group}: {len(cfgs)} configs")
        for cfg in cfgs:
            try:
                out = run_one(cfg, bench_store)
                if out is not None:
                    mrows = evaluate_run(cfg, out["logits"], out["labels"])
                    _metric_path(cfg).write_text(json.dumps(mrows), encoding="utf-8")
                elif _metric_path(cfg).exists():
                    pass  # metrics already saved from a prior run
                else:
                    # run done but metrics missing (e.g. crash after mark_done): recompute
                    logits_path = V2 / "logits_v2" / f"{cfg['stage']}_{cfg['task']}_{cfg['head']}_s{cfg['seed']}_{cfg_hash(cfg)}.npz"
                    if logits_path.exists():
                        b = np.load(logits_path, allow_pickle=True)
                        lg = {k.replace("logits_", ""): b[k] for k in b.files if k.startswith("logits_")}
                        lb = {k.replace("labels_", ""): b[k] for k in b.files if k.startswith("labels_")}
                        mrows = evaluate_run(cfg, lg, lb)
                        _metric_path(cfg).write_text(json.dumps(mrows), encoding="utf-8")
                n_done += 1
                crashes = 0
                if n_done % 20 == 0:
                    log(f"    {group}: {n_done} runs handled")
            except Halt:
                raise
            except Exception as exc:  # noqa: BLE001
                crashes += 1
                warn(f"stage2 run crashed {cfg['task']}/{cfg['head']}/s{cfg['seed']}: {exc}",
                     "run_crash")
                if crashes > 3:
                    raise Halt(f">3 consecutive stage2 crashes; last: {exc}")

    # assemble the long CSV from all per-run metric files (resume-safe)
    merged = []
    for mf in sorted(metric_dir.glob("*.json")):
        merged.extend(json.loads(mf.read_text(encoding="utf-8")))
    write_csv(CSV / "stage2_long.csv", merged)
    log(f"Stage 2 metrics: {len(merged)} rows assembled from {len(list(metric_dir.glob('*.json')))} runs")

    # rerun Stage 1-style analyses on new runs (2.8)
    log("Stage 2.8: re-running Stage 1 analyses on v2 runs")
    table_from_rows(merged, ["auroc", "ece_pos", "ece_top", "brier"], "stage2_metrics")
    table_from_rows(merged, ["auroc", "ece_pos", "ece_top", "brier"], "stage2_metrics",
                    with_collapsed=False)
    return {"rows": len(merged)}


def stage3() -> dict:
    log("=" * 60)
    log("STAGE 3 start (subgroups)")
    from qcv2.stage3 import stage3 as run3
    rows = run3(D)
    write_csv(CSV / "stage3_long.csv", rows)
    log(f"Stage 3 wrote {len(rows)} rows")
    return {"rows": len(rows)}


# ---------------------------------------------------------------------------
def _claims_ledger():
    """Map each output CSV/figure to its producing script + git hash."""
    rows = []
    outdirs = [CSV, FIG]
    for d in outdirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*")):
            rows.append({"file": str(f.relative_to(V2)), "git": git_hash(),
                         "producer": "run_all.py", "generated": now()})
    write_csv(V2 / "claims_ledger.csv", rows)


def _done_md(summary: dict):
    s1 = summary.get("stage1", {})
    s2 = summary.get("stage2", {})
    s3 = summary.get("stage3", {})
    flips = s1.get("flips", {})
    lines = [
        "# DONE (numbers only)",
        "",
        f"- stage1 rows: {s1.get('rows', 0)}",
        f"- stage2 rows: {s2.get('rows', 0)}",
        f"- stage3 rows: {s3.get('rows', 0)}",
        f"- ranking flip task-seeds: {flips.get('n_with_any_flip', 0)}/{flips.get('n_task_seed', 0)}",
        f"- git: {git_hash()}",
        f"- finished: {now()}",
        "- files: results/v2/csv/, results/v2/figures/, results/v2/tables/",
    ]
    (V2 / "DONE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2, 3])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--budget-hours", type=float, default=None)
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    if (V2 / "HALT.md").exists() and not args.skip_tests:
        (V2 / "HALT.md").unlink()

    summary = {}
    try:
        check_split_hashes()
        _env_txt()
        if not args.skip_tests:
            _run_tests()
        want = []
        if args.all:
            want = [1, 2, 3]
        elif args.stage:
            want = [args.stage]
        else:
            want = [1]
        for st in want:
            if st == 1:
                summary["stage1"] = stage1()
            elif st == 2:
                summary["stage2"] = stage2(args.budget_hours)
            elif st == 3:
                summary["stage3"] = stage3()
        _claims_ledger()
        _done_md(summary)
        log("ALL REQUESTED STAGES COMPLETE")
        return 0
    except Halt as h:
        write_halt(str(h))
        log(f"HALT: {h}")
        return 2
    except Exception as exc:  # noqa: BLE001
        write_halt(f"unexpected error: {exc}\n\n{traceback.format_exc()}")
        log(f"HALT (unexpected): {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())