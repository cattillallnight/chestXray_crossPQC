# Q-CalibNet paper folder

Working files for

> Matched-Capacity Hybrid Quantum-Classical Networks for Confidence Calibration under Cross-Population Chest X-Ray Shift

- `main.tex` — submit-facing manuscript (article class; swap to the journal template after venue choice)
- `references.bib` — calibration, CXR shift, PQC
- `manuscript.md` — same claims in Markdown
- `cover_letter.md`
- `submission_checklist.md`

Figures are generated, not drawn by hand:

```
python scripts/make_paper_artifacts.py
python scripts/dump_logits.py
python scripts/bootstrap_and_sensitivity.py
```

Do not retrain `results/runs/*.pt` to change a sentence.
