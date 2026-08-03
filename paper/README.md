# Paper: Privacy-Preserving, LLM-Enhanced Popularity Calibration

Federating the SUPER framework for fair, diverse, and private recommendations.

## Compile on Overleaf (recommended)

No local LaTeX installation is present on this machine. The source is a clean,
Overleaf-ready NeurIPS-style paper. To compile:

1. Create a new project on [Overleaf](https://overleaf.com) (blank project).
2. Upload these files, keeping the directory structure:
   - `main.tex`
   - `bibliography.bib`
   - `neurips.sty`
   - `extra_pkgs.tex`
   - `Makefile` (optional; Overleaf auto-detects)
   - `figures/fig1_architecture.pdf`
   - `figures/fig2_calibration_invariance.pdf`
   - `figures/fig3_pareto.pdf`
   - `figures/fig4_llm_sweep.pdf`
   - `figures/fig5_trajectory.pdf`
3. Set compiler to **pdfLaTeX** and hit Recompile (twice, plus one run for BibTeX).
   With Overleaf, BibTeX runs automatically.

## Regenerating figures

Figures were produced by `src/make_paper_figures.py` from the real experiment
metric JSONs in `experiments/H*/results/`:

```
python src/make_paper_figures.py
```

## Layout

- `main.tex` — the full paper (Abstract, Introduction, Related Work, Method,
  Experiments, Limitations, Conclusion, Reproducibility).
- `bibliography.bib` — 17 verified reference entries; all 15 cited keys match.
- `neurips.sty` + `extra_pkgs.tex` — style and package preamble.
- `figures/` — 5 publication-quality PDFs.
- `Makefile` — `make` produces `main.pdf` when pdflatex + bibtex are available
  (absent on this machine; use Overleaf).

## Key results in the paper

- FedSUPER matches centralized SUPER calibration: RMSE-PC = 0.055, MRMC = 0.151 (H1).
- LLM-FedSUPER (lambda=0.7): recall 0.0366 (+23%), LTC 0.015 -> 0.038, calibration unchanged (H3).
- DP is redundant for calibration inside the blueprint, but debiases plain FedNCF (H2).
