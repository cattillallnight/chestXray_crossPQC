# results/v2 manifest

Every file here is produced by `run_all.py` or a script under
`results/v2/scripts/`. Nothing in `data/splits/`, `results/runs/`,
`results/tables/` or `results/figures/logits/` is modified; those are read-only.
Split SHA-256 hashes are recorded in `split_hashes.json` and re-checked at every
launch.

## Pipeline

`run_all.py` is the single resumable entry point. Work is keyed by config hash
under `state/` and skipped if already finished. HALT writes `HALT.md` and exits;
WARN appends to `ANOMALIES.md` and continues.

## Files

| file | produced by | what it is |
|---|---|---|
| `split_hashes.json` | stage0cd_checkpoint_audit.py | SHA-256[:16] baseline of every split CSV |
| `stage0b_locality.csv` | stage0b_locality.py | product/entangling `⟨Z0⟩` sensitivity + finite-difference grads |
| `stage0c_checkpoint_audit.csv` | stage0cd_checkpoint_audit.py | per-run params, scale/bias, logit & prob ranges, collapse flag |
| `claims_check.csv` | stage0cd_checkpoint_audit.py | paper vs recomputed, per run/metric, with status |
| `stage0_report.md` | written | Stage 0 factual summary |
| `frozen_configs.json` | run_all.py stage 2 | all Stage 2 configs with hashes, written before any run |
| `STAGE2_FROZEN.md` | run_all.py stage 2 | commit hash and freeze time |
| `csv/stage1_long.csv` | run_all.py stage 1 | Stage 1 long-format metrics |
| `csv/stage2_long.csv` | run_all.py stage 2 | Stage 2 long-format metrics |
| `csv/stage3_long.csv` | run_all.py stage 3 | Stage 3 subgroup metrics |
| `figures/F1_ece_vs_prevalence.pdf` | run_all.py stage 1 | ECE vs prevalence + null predictor |
| `figures/F2_recalibration_ladder.pdf` | run_all.py stage 1 | ECE by recalibration rung |
| `figures/F3_logit_range_vs_ece.pdf` | figs_from_csv.py | logit range vs ECE |
| `figures/F4_decision_curves.pdf` | figs_from_csv.py | net benefit vs threshold |
| `figures/rel_*.pdf` | run_all.py stage 1 | reliability diagrams from v1 logits, seed 42 |
| `tables/*.tex` | tables.py | LaTeX table bodies (numbers only, no captions) |
| `target_eval_log.csv` | run_all.py stage 2 | each Stage 2 config's single VinDr eval |
| `claims_ledger.csv` | run_all.py | each output file → producer, git hash, timestamp |
| `stage1_ranking_flips.json` | run_all.py stage 1 | head-rank order at each prevalence |

## Stage 2 config freeze

`frozen_configs.json` fixes the grid before any run:

| group | runs | tag |
|---|---|---|
| primary | 360 (4 tasks × 9 heads × 10 seeds) | confirmatory |
| legacy | 120 (learned-W bottleneck) | exploratory |
| prior_ablation | 320 (2.6) | exploratory |
| depth_ntrain | 1440 (2.7) | exploratory |

Confirmatory grid: 2 layers, single Y-embedding, seeds 42–51, early stopping on
validation NLL, no tuning. Freeze recorded at git `2256cd1` (see
`STAGE2_FROZEN.md` for the exact commit written at freeze time).

## Environment

See `env.txt` (written at each launch). Current: Python 3.11.9, torch 2.6.0+cpu,
PennyLane 0.38.0, numpy 1.26.4, sklearn 1.5.2, pandas 2.2.3, matplotlib 3.11.2.
Project venv: `.venv/` (gitignored). Git repository initialised at project root.

## Anomalies (see ANOMALIES.md for the running log)

- `cardiomegaly_product_pqc_seed43`: collapsed v1 checkpoint (logit range 6.7e-5).
- `pneumonia_proxy_product_pqc_seed44`: collapsed v1 checkpoint (logit range 4.2e-3).
- v1 ECE = positive-class confidence, not max-confidence. v2 reports all three
  (`ece_pos`, `ece_top`, `smece`).
- VinDr official test labels absent locally (3,000 PNGs, no CSV).
- VinDr splits carry no demographic metadata; Stage 3 VinDr subgroups not computed.