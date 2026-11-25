"""Build patient-level CheXpert splits and VinDr external holdouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qcalibnet.config import load_config
from qcalibnet.datasets import (
    build_chexpert_cohort,
    build_vindr_cohort,
    cohort_summary,
    split_chexpert_patient_level,
    split_vindr_holdout,
    vindr_labels_available,
)
from qcalibnet.labels import TASKS
from qcalibnet.paths import SPLITS_DIR, VINDR_DIR, ensure_dirs
from qcalibnet.seed import seed_everything


def write_frame(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    seed_everything(seed)
    ensure_dirs()
    tasks = args.tasks or list(cfg["tasks"])
    summary = {"seed": seed, "tasks": {}}

    for task_name in tasks:
        spec = TASKS[task_name]
        print(f"\n=== {task_name} ===")
        chex = build_chexpert_cohort(
            spec,
            u_protocol=cfg["chexpert"]["u_protocol"],
            frontal_only=bool(cfg["chexpert"]["frontal_only"]),
        )
        print("chexpert cohort", cohort_summary(chex))
        splits = split_chexpert_patient_level(
            chex,
            n_train=int(cfg["chexpert"]["n_train"]),
            n_val=int(cfg["chexpert"]["n_val"]),
            n_test=int(cfg["chexpert"]["n_test"]),
            seed=seed,
            balanced_train=bool(cfg["chexpert"]["balanced_train"]),
        )
        task_dir = SPLITS_DIR / task_name
        for name, frame in splits.items():
            write_frame(task_dir / f"chexpert_{name}.csv", frame)
            print(f"  chexpert {name}", cohort_summary(frame))
        task_sum = {
            "chexpert_cohort": cohort_summary(chex),
            "chexpert": {k: cohort_summary(v) for k, v in splits.items()},
        }

        if vindr_labels_available():
            vindr = build_vindr_cohort(spec, vote=cfg["vindr"]["vote"])
            print("vindr cohort", cohort_summary(vindr))
            hold = split_vindr_holdout(
                vindr,
                n_test=int(cfg["vindr"]["n_test"]),
                seed=seed,
                balanced=bool(cfg["vindr"]["balanced_eval"]),
            )
            write_frame(task_dir / "vindr_external_test.csv", hold["external_test"])
            if bool(cfg["vindr"].get("natural_eval", True)):
                natural = split_vindr_holdout(
                    vindr,
                    n_test=int(cfg["vindr"]["n_test"]),
                    seed=seed + 1,
                    balanced=False,
                )
                write_frame(task_dir / "vindr_external_test_natural.csv", natural["external_test"])
                print("  vindr natural", cohort_summary(natural["external_test"]))
            print("  vindr balanced", cohort_summary(hold["external_test"]))
            task_sum["vindr_cohort"] = cohort_summary(vindr)
            task_sum["vindr_external_test"] = cohort_summary(hold["external_test"])
        else:
            print(f"VinDr labels missing: skip external splits. Place train.csv at {VINDR_DIR / 'train.csv'}")
            task_sum["vindr"] = "missing_labels"
        summary["tasks"][task_name] = task_sum

    out = SPLITS_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
