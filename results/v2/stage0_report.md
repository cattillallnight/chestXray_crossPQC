# Stage 0 — Audit report (read-only)

Produced by `results/v2/scripts/stage0b_locality.py` and
`results/v2/scripts/stage0cd_checkpoint_audit.py`. All outputs are exploratory
on existing artefacts; nothing was retrained and no split was touched.

## Environment caveats (raised before proceeding)

1. **Not a git repository.** `git rev-parse` fails at `E:/cvchestpj`. Rule 5
   asks me to commit after every stage and record a git hash. I cannot. I am
   recording file SHA-256 hashes instead (see `split_hashes.json`) and have not
   initialised a repo, since that is a decision for you.
2. **torch was not installed** on this machine. I created a throwaway CPU venv
   at `%TEMP%/venvqc` (torch 2.15.0+cpu) purely to load the `.pt` checkpoints.
   It is outside the project tree and does not modify the repo.
3. **PennyLane is not installed.** Stage 0b was therefore verified with an
   independent numpy statevector simulator that mirrors `qcalibnet/models.py`
   exactly. This is stronger for the locality claim than trusting the same
   library. Stage 2.4 will need PennyLane or the batched torch simulator; I will
   raise that before Stage 2.

## 0a — Parameter counts (recomputed)

| component | params |
|---|---|
| bottleneck `W,b` (1024×6 + 6) | 6150 |
| MLP head only (6×8+8 + 8×1+1) | 65 |
| PQC head only (2×6×3 + scale + bias) | 38 |
| **MLP total (W + head)** | **6215** |
| **PQC total (W + head)** | **6188** |

The paper's 6215 / 6188 use the total including the shared, trained bottleneck
`W`. Head-only, the gap is 65 vs 38 = 27 parameters. PQC theta parameters are
36, of which only 6 act on `q0` (the measured wire): `theta_params_q0 = 6`,
`theta_params_total = 36`, for every PQC checkpoint.

## 0b — Product PQC input locality (numerically verified)

`stage0b_locality.csv`. Product head, perturbing each input by 0.37 rad:

- z1..z5 change `⟨Z0⟩` by |Δ| ≤ 5.6e-16 → **exactly zero (machine precision)**.
- z0 changes `⟨Z0⟩` by 3.6e-1 → the only feature that matters.
- Finite-difference ∂`⟨Z0⟩`/∂W for rows 1..5: |grad| ≤ 1.4e-10 → **zero**.
- Entangling head: all of z0..z5 move `⟨Z0⟩` (|Δ| 7e-3 .. 2.6e-2).

**Verdict: the product head is functionally a one-feature (z0) model, and
gradients into W rows 1–5 are exactly 0.** This confirms the paper's caveat.

## 0c — Checkpoint audit (45 runs)

`stage0c_checkpoint_audit.csv` (per run: bottleneck/head params, learned
`scale`/`bias`, logit and probability min/max/std, collapse flag).

**Two collapsed checkpoints found** (logit range < 1e-2, probability essentially
constant):

| run | logit range | prob range |
|---|---|---|
| `cardiomegaly_product_pqc_seed43` | 6.7e-05 | 0.4995–0.4995 |
| `pneumonia_proxy_product_pqc_seed44` | 4.2e-03 | 0.3045–0.3054 |

Both are **product** heads on rare/ambiguous tasks. The cardiomegaly one is the
same run the paper already calls an outlier (AUROC 0.596) — it is not a marginal
outlier, it is a run that failed to learn any separating function. Its AUROC of
0.596 comes purely from tie-ranking noise in a 6.7e-5 range.

## 0d — Metric recomputation (claims_check.csv)

144 metric checks over the 4 primary tasks × 3 heads × 3 seeds against
`all_runs.csv`:

- **108 reproduced** (|diff| < 1e-6): AUROC (tie-aware), ECE, Brier.
- **0 mismatch.**
- 36 rows are a *definition note*, not a mismatch (below).

### ECE definition (important)

`qcalibnet/metrics.py` computes ECE with **confidence = positive-class
probability p**, accuracy = fraction of positives in the bin. It does **not**
use the standard max-confidence `max(p,1-p)` + correctness formulation. For
`no_finding_mlp_seed42`: positive-class ECE = 0.244, standard max-conf ECE =
0.081. Across the 36 primary runs the standard value is on average **0.137
lower**. Eq. 3 of the paper writes `|acc(b) − conf(b)|` without stating which
confidence; the code uses positive-class p. This must be stated explicitly or
the numbers are not comparable to most calibration literature.

## 0e — VinDr official test labels

`VinBigData/test/` holds 3,000 PNGs and **no label file**. `train_meta.csv`
covers the 15,000 *training* images (0 overlap with the test ids). The official
test labels are **not present locally**. Not downloaded, per instruction.

## Totals

- 45/45 checkpoints loadable; 43 healthy, 2 collapsed.
- AUROC/ECE/Brier reproduce exactly from logits.
- Product-head locality proven analytically and numerically.
- ECE definition differs from the standard max-confidence definition.