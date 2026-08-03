# Research Log

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-08-01 | bootstrap | Initialized workspace for SUPER extension. Goal: privacy-preserving FL+DP+LLM fair recommender. |
| 2 | 2026-08-01 | bootstrap | Located SUPER paper via Crossref (Yavru et al., IEEE Access 2026). Read full 24-page PDF. Core: Pareto partition (alpha=0.20), dual model training, blueprint merge (hard quota/soft blueprint). |
| 3 | 2026-08-01 | bootstrap | arXiv search: 59 papers across 6 clusters. 5 gaps found. 5 hypotheses. |
| 4 | 2026-08-01 | inner-loop | H1 v2: FedSUPER achieves Rmse-PC=0.055, MRMC=0.151 — identical to centralized SUPER. CONFIRMED. |
| 5 | 2026-08-01 | inner-loop | H2 v2: DP sweep eps={8,4,2}. FedSUPER recall drops from 0.030 to 0.013 (57%). DP is redundant for calibration when blueprint merge is active. FedNCF-DP alone gets Rmse-PC 0.705→0.283. |
| 6 | 2026-08-02 | inner-loop | H3 v2: Sentence-BERT LLM user profiling. lam=0.7 yields recall 0.0366 (+23% over 0.0298), LTC 0.038 (2.6x). Calibration unaffected (Rmse-PC=0.055). CONFIRMED. |
| 7 | 2026-08-02 | inner-loop | H4: Multi-obj popularity-dispersion penalty sweep (alpha 0..001..0.05). GKPI +0.0003 max. Negative — blueprint merge already dominates. |
| 8 | 2026-08-02 | inner-loop | H5: Adaptive DP (per-item pop-scaled noise, amp=2). FedSUPER-DP recall 0.0137 vs uniform 0.0131 (+5%). PARTIAL CONFIRM. |
| 9 | 2026-08-02 | outer-loop | Synthesis: H1+H3 core contribution. LLM-FedSUPER achieves private calibration (Rmse-PC=0.055) + LLM accuracy recovery (recall 0.037). Paper ready. |
| 10 | 2026-08-02 | report | Completing final tax: findings.md written. Proceeding to progress report + paper. |
| 11 | 2026-08-03 | paper | Wrote complete NeurIPS-style paper (paper/main.tex, 398 lines): Intro, Related Work, Method (invariance theorem + proof, LLM blend, DP inside/outside, adaptive DP), Experiments (H1-H5 + tabular + 5 figures), Limitations, Conclusion, Reproducibility. Verified all 15 \cite keys against bibliography.bib (17 keys). Added fig1_architecture to Method. Overleaf-ready (no local LaTeX). |