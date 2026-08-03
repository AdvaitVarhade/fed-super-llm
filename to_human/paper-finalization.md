# Paper Finalization Note (to_human)

**Status:** paper-draft complete, Overleaf-ready.

The full NeurIPS-style paper is in `paper/main.tex` (398 lines) with `paper/bibliography.bib`
(17 entries), `paper/neurips.sty`, `paper/extra_pkgs.tex`, and 5 figures in `paper/figures/`.
Everything is committed: `research(paper): Privacy-Preserving Federated SUPER + LLM paper complete`.

## You must compile on Overleaf

This machine has **no pdflatex/bibtex/TeX Live/MiKTeX**, so a local PDF build is not possible.
To produce the PDF:

1. Go to https://overleaf.com -> New Project -> Blank.
2. Upload these files (keep the `figures/` subfolder):
   - `main.tex`
   - `bibliography.bib`
   - `neurips.sty`
   - `extra_pkgs.tex`
   - `figures/fig1_architecture.pdf` ... `figures/fig5_trajectory.pdf`
3. Compiler: **pdfLaTeX**. Recompile twice; Overleaf runs BibTeX automatically.

The `Makefile` also works if you later install TeX Live/MiKTeX and run `make` in `paper/`.

## Verified before finalizing

- All 15 `\cite{...}` keys match `bibliography.bib` (17 keys; 2 uncited but valid).
- Every metric in Table 1 and the hypothesis narrative was cross-checked against the raw
  `experiments/*/results/metrics_*.json` files.
- All 5 figures referenced (`fig1`..`fig5`) exist as PDFs; `fig1_architecture.pdf` now appears in
  the Method section.
- LaTeX environments balanced (18 begin / 18 end); no placeholder or garbled text; no non-ASCII
  characters that would break pdfLaTeX.

## Headline claims (grounded in the JSONs)

| Claim | Value |
|---|---|
| FedSUPER calibration == centralized SUPER | RMSE-PC 0.055 / MRMC 0.151 |
| LLM-FedSUPER (lambda=0.7) recall | 0.0366 (+23% vs 0.0298) |
| LTC gain | 0.015 -> 0.038 |
| DP debiases plain FedNCF | RMSE-PC 0.705 -> 0.283 (eps=2) |
| DP redundant under blueprint | RMSE-PC stays 0.055; recall 0.030 -> 0.013 |

## Suggested next steps (yours)

- Compile on Overleaf and review the PDF layout (figures/tables placement).
- Optionally adjust the `\title` / author block before submission.
