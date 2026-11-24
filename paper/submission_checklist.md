# Submission checklist (honest)

## Framing (must remain true)

- [x] Title matches the codebase: "What a Six-Qubit Head Can and Cannot Change"
- [x] RQ2 (calibration / entanglement) is primary; AUROC is secondary
- [x] Readout-locality caveat stated: product head is functionally a one-feature model
- [x] Clean classical-vs-quantum comparison identified as MLP vs entangling PQC
- [x] DenseNet full classifier is unmatched and not used as a quantum baseline
- [x] `pneumonia_proxy` is appendix / ontology_shift
- [x] No hardware, no quantum advantage, no 1024-on-6-qubits claim
- [x] No "novel architecture" claim; circuit is standard PennyLane
- [x] Limitations in the main text, not only a supplement
- [x] Seven T-degenerate runs retained in the main table
- [x] Citations follow `midl.cls` author–year (`plainnat`), not IEEE numbered

## Metrics (all promised endpoints now reported)

- [x] AUROC (Table 2)
- [x] AUPRC, F1@0.5, Brier, MCE (Table 3)
- [x] ECE on VinDr balanced (Table 3) and natural-prevalence (Table 5)
- [x] ECE bin-width sensitivity discussed in text

## Experiments already frozen

- [x] 45/45 JSON in `results/tables/`
- [x] VinDr keys present on every run
- [x] Splits not resampled after the grid
- [x] Checkpoints in `results/runs/`

## Figures (regenerated, one visual system)

- [x] `fig_pipeline.pdf` (trunk → bottleneck → heads → logit → T)
- [x] `fig_circuits.pdf` (product vs entangling, readout locality)
- [x] `fig_main_results.pdf` (AUROC and ECE bars)
- [x] `fig_calibration_transfer.pdf` (temp transfer scatter)
- [x] `fig_reliability*.pdf` (all five tasks, seed 42)
- [x] Built by `scripts/make_figures_final.py`

## MIDL 2027 Main Track (this is the submission)

- [x] Official `midl.cls` (no `anon`; single-blind)
- [x] `paper/midl2027/midl-fullpaper.tex` (full paper, not short)
- [x] Compiles cleanly (tectonic, 0 overfull boxes) — body 12 pages, 19 total
- [x] OpenReview abstract / comments files updated
- [ ] Re-upload rebuilt `QCalibNet_MIDL2027.zip` to Overleaf; compile with pdfLaTeX
- [ ] Confirm body ≤ 14 pages before references / appendix (currently 12)
- [ ] English copy-edit by a second reader
- [ ] Create OpenReview submission before 30 Nov 2026 AoE; full PDF by 4 Dec 2026 AoE
- [ ] Do not dual-submit the same manuscript to a journal while MIDL is under review

## Do not do before submit

- Do not retrain the 45 runs
- Do not silently "fix" the product readout and re-report; either keep as-is (documented) or retrain and disclose
- Do not drop T-degenerate runs from the main table (ablation only)
- Do not report Wilcoxon/DeLong as confirmatory with n=3
- Do not cite this as a CXR detector