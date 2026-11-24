# results/v2 — pipeline outputs

Factual index. Every file is produced by `run_all.py` or a script under
`results/v2/scripts/`. Nothing in `data/splits/`, `results/runs/`,
`results/tables/` or `results/figures/logits/` is modified; those are read-only.

## Entry point

```
python run_all.py --stage 1                 # Stage 1 only
python run_all.py --stage 2 --budget-hours H
python run_all.py --all --budget-hours H
```

Resumable: each unit of work is keyed by a config hash under `state/` and
skipped if already finished. Re-launch after any interruption. Logs append to
`logs/run_all.log`.

HALT conditions write `HALT.md` and exit non-zero (split hash change, unit-test
failure, VinDr guard triggered, missing input, >3 consecutive run crashes).
WARN conditions append to `ANOMALIES.md` and continue.

## Files

| path | what it is |
|---|---|
| `csv/stage1_long.csv` | Stage 1 long-format metrics (prevalence sweep, recal ladder, metric battery, bootstrap, operating point) |
| `csv/stage2_long.csv` | Stage 2 long-format metrics (corrected-protocol runs) |
| `csv/stage3_long.csv` | Stage 3 subgroup metrics (CheXpert Sex/Age/AP-PA) |
| `figures/F1_ece_vs_prevalence.pdf` | ECE vs target prevalence per head + null predictor |
| `figures/F2_recalibration_ladder.pdf` | ECE by recalibration rung |
| `figures/F3_logit_range_vs_ece.pdf` | output-logit range vs ECE |
| `figures/F4_decision_curves.pdf` | net benefit vs threshold |
| `figures/rel_*.pdf` | reliability diagrams recomputed from logits |
| `tables/*.tex` | LaTeX table bodies (numbers only, no captions) |
| `target_eval_log.csv` | every Stage 2 config's single VinDr evaluation (timestamp, config hash, seed) |
| `claims_ledger.csv` | each output file → producer, git hash, timestamp |
| `split_hashes.json` | SHA-256[:16] baseline of every split CSV (integrity) |
| `frozen_configs.json` | all Stage 2 configs with hashes, written before any run |
| `STAGE2_FROZEN.md` | commit hash at freeze time |
| `stage0_report.md` | Stage 0 audit findings |
| `env.txt` | interpreter, package versions, git hash |
| `MANIFEST.md`, `ANOMALIES.md`, `DEVIATIONS.md`, `DONE.md` | provenance / warnings / deviations / final counts |

## Stage 2 config freeze

`frozen_configs.json` fixes the confirmatory grid (2 layers, single Y-embedding,
seeds 42–51, early stopping on validation NLL). A Stage 2 run is tagged
`confirmatory` only if its config is in a commit that precedes the run. Configs
for re-uploading / n_train / prior ablations are tagged `exploratory`.

## VinDr guard

All training, temperature, Platt, prior-shift, BBSE and Saerens-EM fits call
`guard.assert_not_vindr(origin)`, which raises if the data is tagged VinDr. The
only exception is the labelled `oracle_platt` rung, which passes
`allow_vindr=True` and is reported as an upper bound, never as deployable.