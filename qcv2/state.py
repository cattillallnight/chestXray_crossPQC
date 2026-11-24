"""Resumable run bookkeeping, HALT/WARN handling, hashing, git helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .paths import LOGS, ROOT, STATE, V2


class Halt(Exception):
    """Raised to stop the whole pipeline; main() writes HALT.md and exits."""


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, also_print: bool = True) -> None:
    line = f"{now()}  {msg}"
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "run_all.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    if also_print:
        print(line, flush=True)


def warn(msg: str, kind: str = "generic") -> None:
    """Append a WARN row to ANOMALIES.md and continue."""
    p = V2 / "ANOMALIES.md"
    new = not p.exists()
    with p.open("a", encoding="utf-8") as f:
        if new:
            f.write("# Anomalies\n\n| time | kind | detail |\n|---|---|---|\n")
        f.write(f"| {now()} | {kind} | {msg.replace('|', '/')} |\n")
    log(f"WARN [{kind}] {msg}")


def deviation(msg: str) -> None:
    p = V2 / "DEVIATIONS.md"
    new = not p.exists()
    with p.open("a", encoding="utf-8") as f:
        if new:
            f.write("# Deviations\n\n| time | detail |\n|---|---|\n")
        f.write(f"| {now()} | {msg.replace('|', '/')} |\n")
    log(f"DEVIATION {msg}")


def write_halt(cause: str) -> None:
    (V2 / "HALT.md").write_text(
        f"# HALT\n\nStopped at {now()}.\n\nCause:\n\n{cause}\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def git(*args) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True,
                             text=True, check=False)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def git_hash() -> str:
    return git("rev-parse", "--short", "HEAD") or "no-git"


class RunStore:
    """Content-addressed done-set. A unit of work is 'done' iff its key file
    exists under results/v2/state/. Key = hash(config) + step name."""

    def __init__(self, step: str):
        self.step = step
        self.dir = STATE / step
        self.dir.mkdir(parents=True, exist_ok=True)

    def key_path(self, cfg: dict) -> Path:
        return self.dir / f"{hash_obj(cfg)}.json"

    def is_done(self, cfg: dict) -> bool:
        return self.key_path(cfg).exists()

    def mark_done(self, cfg: dict, extra: dict | None = None) -> None:
        payload = {"cfg": cfg, "finished": now(), "git": git_hash(),
                   "git_full": git("rev-parse", "HEAD")}
        if extra:
            payload.update(extra)
        self.key_path(cfg).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def done_count(self) -> int:
        return len(list(self.dir.glob("*.json")))


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.elapsed = time.time() - self.t0


def check_split_hashes() -> None:
    """HALT if any split file differs from the Stage-0 baseline."""
    base = V2 / "split_hashes.json"
    if not base.exists():
        raise Halt("results/v2/split_hashes.json missing; run Stage 0 audit first")
    baseline = json.loads(base.read_text(encoding="utf-8"))
    from .paths import SPLITS
    diffs = []
    for rel, want in baseline.items():
        p = SPLITS / rel
        if not p.is_file():
            diffs.append(f"{rel}: missing")
        elif not sha256_file(p).startswith(want):
            diffs.append(f"{rel}: hash changed")
    if diffs:
        raise Halt("split integrity violated:\n  " + "\n  ".join(diffs))
    log(f"split integrity OK ({len(baseline)} files)")