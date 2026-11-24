"""Sequential remaining-run queue. Skips existing results/tables/*.json.

Keeps Windows awake while the queue is running so an overnight laptop
does not sleep mid-PQC epoch.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


def keep_awake() -> None:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        )
    except Exception:
        pass


def allow_sleep() -> None:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TABLES = ROOT / "results" / "tables"
LOG = ROOT / "results" / "train_queue.log"

TASKS = [
    "no_finding",
    "cardiomegaly",
    "pleural_effusion",
    "consolidation",
    "pneumonia_proxy",
]
HEADS = ["mlp", "product_pqc", "entangling_pqc"]
SEEDS = [42, 43, 44]


def jobs() -> list[tuple[str, str, int]]:
    out = []
    for task in TASKS:
        for head in HEADS:
            for seed in SEEDS:
                out.append((task, head, seed))
    return out


def done(task: str, head: str, seed: int) -> bool:
    return (TABLES / f"{task}_{head}_seed{seed}.json").is_file()


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> int:
    keep_awake()
    pending = [j for j in jobs() if not done(*j)]
    skipped = len(jobs()) - len(pending)
    log(f"queue start  pending={len(pending)}  skipped={skipped}  python={PYTHON}")
    if not pending:
        log("nothing to run")
        allow_sleep()
        return 0

    try:
        failed = 0
        for i, (task, head, seed) in enumerate(pending, 1):
            run_id = f"{task}_{head}_seed{seed}"
            log(f"[{i}/{len(pending)}] START {run_id}")
            cmd = [
                PYTHON,
                str(ROOT / "scripts" / "train.py"),
                "--task",
                task,
                "--head",
                head,
                "--seed",
                str(seed),
            ]
            proc = subprocess.run(cmd, cwd=str(ROOT))
            if proc.returncode != 0:
                failed += 1
                log(f"[{i}/{len(pending)}] FAIL {run_id}  exit={proc.returncode}")
                continue
            log(f"[{i}/{len(pending)}] DONE {run_id}")
        if failed:
            log(f"queue finished with failures={failed}")
            return 1
        log("queue complete")
        return 0
    finally:
        allow_sleep()


if __name__ == "__main__":
    raise SystemExit(main())
