# Submission checklist (honest)

## Framing (must remain true)

- [x] Title matches the codebase, not a pneumonia SOTA claim
- [x] RQ2 (calibration / entanglement) is primary; AUROC is secondary
- [x] DenseNet full classifier is unmatched and not used as a quantum baseline
- [x] `pneumonia_proxy` is appendix / ontology_shift
- [x] No hardware, no quantum advantage, no 1024²-on-6-qubits claim
- [x] No “novel architecture” claim; circuit is standard PennyLane
- [x] Limitations in the main text, not only a supplement
- [x] Seven T-degenerate runs retained in the main table
- [x] Citations follow `midl.cls` author–year (`plainnat`), not IEEE numbered

## Experiments already frozen

- [x] 45/45 JSON in `results/tables/`
- [x] VinDr keys present on every run
- [x] Splits not resampled after the grid
- [x] Checkpoints in `results/runs/`

## MIDL 2027 Main Track (this is the submission)

- [x] Official `midl.cls` (no `anon`; single-blind)
- [x] `paper/midl2027/midl-fullpaper.tex` (not short paper)
- [x] `paper/QCalibNet_MIDL2027.zip` rebuilt after the last TeX edit
- [x] OpenReview abstract / comments files
- [x] Tables copied from `paper_primary_mean_std.csv`
- [x] Figures: AUROC/ECE bars, temperature transfer, reliability (all five tasks, seed 42)
- [ ] Compile on Overleaf with **pdfLaTeX** after upload (`midl-fullpaper.tex`)
- [ ] Confirm **body ≤ 14 pages** before acknowledgements / references / appendix
- [ ] English copy-edit by a second reader
- [ ] Create OpenReview submission before 30 Nov 2026 AoE; full PDF by 4 Dec 2026 AoE
- [ ] Do not dual-submit the same manuscript to a journal while MIDL is under review

## Journal Overleaf package (not the MIDL upload)

- [ ] Journal template only after a MIDL decision, or if MIDL is withdrawn

## Do not do before submit

- Do not retrain the 45 runs
- Do not add qubits or unfreeze DenseNet
- Do not drop T-degenerate runs from the main table (ablation only)
- Do not report Wilcoxon/DeLong as confirmatory with n=3
- Do not cite this as a CXR detector
- Do not invent a special track or claim guaranteed acceptance
