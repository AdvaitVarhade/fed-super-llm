# H4 Protocol: Multi-objective scalarization over F/D/N/A (corrected)

## Hypothesis

A multi-objective scalarized training objective combining BPR (accuracy), popularity-dispersion (coverage/novelty), and fairness-aware terms during the M_pop/M_tail training will yield a Pareto-dominant point over the best single-axis FedSUPER variants. The composite FDN-A (or per-paper GKPI) should improve by >=0.05 on the FDN-A composite we previously used.

## Mechanism

FedSUPER + LLM achieves perfect calibration (Rmse-PC=0.055) but recall has reached a ceiling near 0.037 because the blueprint merge is constrained by per-user quota. Adding multi-objective training regularization during M_pop and M_tail training (e.g., a popularity-dispersion penalty that explicitly pushes predicted scores for tail items up) can expand the per-pool top-N with semantically richer candidates, lifting recall+coverage beyond the LLM-only configuration.

## Variants

We add a popularity-dispersion regularizer during BPR training:
  L_total = L_BPR - alpha_popdisp * popularity_dispersion(predicted_scores_in_batch)

Where popularity_dispersion penalizes the case where popular items dominate batch scores:
  popularity_dispersion = sum_i( score(batch_i) * pop(batch_i) )

Minimizing L = -alpha * popularity_dispersion means we EXPLICITLY REDUCE the mean predicted score weighted by popularity -> the model is incentivized to NOT over-rank popular items in BPR.

Sweep alpha_popdisp in {0.0, 0.001, 0.005, 0.01, 0.05} for M_pop model.
M_tail uses the BPR baseline only (tail is already popularity-poor).

Then apply SUPER blueprint merge, optionally with LLM augmentation at lam=0.7 (best from H3 v2).

## Setup
- Use the same FedSUPER-noDP framework as H3 v2 (no DP).
- Train M_pop with multi-obj alpha; M_tail with BPR only.
- Compare Recall, nDCG, APLT, LTC, Rmse-PC, MRMC, GKPI.

## Stop criterion
If best alpha yields GKPI >= best LLM-FedSUPER-noDP's 0.031 + 0.005, H4 CONFIRMATORY.

## Pre-registration
Locked BEFORE running.
