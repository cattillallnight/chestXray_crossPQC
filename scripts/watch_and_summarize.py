"""Wait for the detached train queue, then write the paper summary tables.

Does not start a second queue. Safe to leave running overnight.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "watch_summarize.log"
PYTHON = sys.executable


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def queue_running() -> list[int]:
    out = subprocess.check_output(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.Name -match 'python' -and $_.CommandLine -and "
                "($_.CommandLine -match 'run_queue\\.py' -or $_.CommandLine -match 'scripts\\\\train\\.py') } | "
                "Select-Object -ExpandProperty ProcessId"
            ),
        ],
        text=True,
        cwd=str(ROOT),
    )
    pids = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def main() -> int:
    log(f"watch start python={PYTHON}")
    idle = 0
    while True:
        pids = queue_running()
        n_json = len(list((ROOT / "results" / "tables").glob("*_seed*.json")))
        if pids:
            idle = 0
            log(f"queue alive pids={pids} tables={n_json}")
            time.sleep(120)
            continue
        idle += 1
        log(f"queue absent tables={n_json} confirm={idle}/2")
        if idle < 2:
            time.sleep(30)
            continue
        break

    log("queue finished — backfill VinDr then summarize")
    back = subprocess.run([PYTHON, str(ROOT / "scripts" / "backfill_vindr.py")], cwd=str(ROOT))
    log(f"backfill exit={back.returncode}")
    proc = subprocess.run([PYTHON, str(ROOT / "scripts" / "summarize.py")], cwd=str(ROOT))
    log(f"summarize exit={proc.returncode}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
