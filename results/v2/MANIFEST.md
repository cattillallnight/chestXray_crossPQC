# results/v2 manifest

All files here are produced by scripts in `results/v2/scripts/`. Nothing in
`data/splits/`, `results/runs/`, `results/tables/`, or `results/figures/logits/`
was modified. Split SHA-256 hashes are recorded in `split_hashes.json` and are
re-checked by `tests/test_v2.py`.

| file | produced by | what it is |
|---|---|---|
| `split_hashes.json` | stage0cd_checkpoint_audit.py | SHA-256[:16] of every split CSV at audit time (integrity baseline) |
| `stage0b_locality.csv` | stage0b_locality.py | product/entangling `⟨Z0⟩` sensitivity to each input and finite-difference grads |
| `stage0c_checkpoint_audit.csv` | stage0cd_checkpoint_audit.py | per-run params, scale/bias, logit & prob ranges, collapse flag |
| `claims_check.csv` | stage0cd_checkpoint_audit.py | paper value vs recomputed, per run/metric, with status |
| `stage0_report.md` | (written) | factual Stage 0 summary |

## Environment recorded at Stage 0

- Python 3.11.9
- torch 2.15.0+cpu (throwaway venv `%TEMP%/venvqc`, outside the repo)
- numpy 2.4.6
- PennyLane: **not installed** (Stage 0b verified with a numpy simulator instead)
- git: **no repository** at the project root (cannot record a commit hash)

## Anomalies logged

- `cardiomegaly_product_pqc_seed43`: collapsed (logit range 6.7e-5).
- `pneumonia_proxy_product_pqc_seed44`: collapsed (logit range 4.2e-3).
- ECE in the codebase = positive-class confidence, not max-confidence.
- VinDr official test labels absent locally.