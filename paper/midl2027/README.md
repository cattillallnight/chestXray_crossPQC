# MIDL 2027 full paper (Main Track)

**Track:** Main Track — full paper (PMLR)  
**Not:** Short paper (3 pages, no appendix, not PMLR)  
**Not:** a journal special issue. Do not dual-submit while MIDL is under review.

Official template: `midl.cls` from https://github.com/MIDL-Conference/MIDLLatexTemplate  
Do not load `times`, `subcaption`, `cleveref`, `geometry`, or `setspace`. Do not change fonts or margins. Geometry is owned by `jmlr`/`midl`. Citations are the official natbib **author–year** style (`plainnat`). Do not force numbered IEEE citations.

Author email in the PDF: `quyentttiace200993@gmail.com`

## Overleaf

Upload `paper/QCalibNet_MIDL2027.zip` (rebuilt after the last TeX edit).

1. New Project → Upload Project → `QCalibNet_MIDL2027.zip`
2. Main document: `midl-fullpaper.tex`
3. Compiler: **pdfLaTeX** (not XeLaTeX / LuaLaTeX)
4. Recompile from scratch (pdfLaTeX → BibTeX → pdfLaTeX × 2)
5. Check **body ≤ 10 pages** (Main Track) before `\clearpage` / acknowledgements / references / appendix. Those sections do not count toward the limit. After rebuttal the Main Track camera-ready limit is 12 pages. The 14/16-page limit is the Clinical Translation & Validation special track, not this paper.

Zip contents: `midl-fullpaper.tex`, `midl.cls`, `references.bib`, `latexmkrc`, `orcid.png`, figures, OpenReview text files, `cover_letter.md`. Do not include `template_src/`.

## OpenReview (create the submission *before* 30 Nov 2026 AoE)

| Item | Value |
|---|---|
| Abstract registration | 30 Nov 2026 AoE |
| Full PDF | 4 Dec 2026 AoE |
| Review release | 10 Jan 2027 |
| Rebuttal | 18–24 Jan 2027 |
| Decision | 12 Feb 2027 |
| Camera-ready | 12 Mar 2027 |
| Conference | Porto, 14–16 Jul 2027 |

Use `quyentttiace200993@gmail.com` on OpenReview. Review is **single-blind**: author name stays in the PDF (`\documentclass{midl}`, no `anon` option).

Paste `openreview_abstract.txt` into the OpenReview abstract box. Select **Main Track**. Full papers are archival (PMLR): do not dual-submit the same manuscript to a journal while MIDL is under review.

## Claims that must stay true

- Not a new architecture. Standard AngleEmbedding-Y + RX/RY/RZ ± CNOT ring + ⟨Z₀⟩.
- Matched-capacity 6-d bottleneck. MLP 6215, PQC 6188.
- Frozen ImageNet DenseNet-121. PQC never sees pixels.
- Primary tasks: no finding, cardiomegaly, pleural effusion, consolidation.
- `pneumonia_proxy` is ontology shift → appendix.
- No quantum advantage, no hardware, no SOTA CXR detector.
- Unmatched full DenseNet classifier is not a baseline.
- Seven $T$-degenerate runs stay in the main table.
- No confirmatory $p$-values ($n=3$ seeds).

Numbers are copied from `results/tables/paper_primary_mean_std.csv` after the 45/45 grid. Do not edit a mean by hand.
