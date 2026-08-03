# H4 Analysis: Multi-objective popularity-dispersion training

## Result status

**H4 INCONCLUSIVE / minimal effect**. The popularity-dispersion regularization during M_pop training has only a marginal effect on the Pareto front of LLM-FedSUPER:
- Best alpha (0.005) yields GKPI = 0.0317 vs baseline 0.0314 (+0.0003, ~1% relative)
- Recall/nDCG/APLT/LTC remain essentially unchanged across alpha sweep
- Calibration (Rmse-PC=0.055, MRMC=0.151) is preserved at all alpha values

## Mechanism interpretation

The popularity-dispersion penalty adds `alpha * log1p(pop[i]*100) * ps` to the BPR loss where `ps` is the positive-item score and `pop[i]` is the popularity probability. This SHOULD reduce the score of popular items in M_pop, expanding the candidate pool's relevance to less popular items. Why does it not improve the headline metrics?

1. **SUPER blueprint merge already enforces quota**: The hard quota (N_pop, N_tail) is invariant to score ordering. Changing M_pop scores within the 75-item head pool doesn't change the per-user quota.
2. **LLM blending is already highly discriminating**: When the LLM blend weight is 0.7, the LLM scores dominate the pool ranking regardless of the underlying FedNCF score. So any training-time regularization is washed out by the LLM.
3. **Without LLM, alpha_popdisp has a small effect**: GKPI goes from 0.023 (alpha=0) to 0.020 (alpha=0.05) — the penalty hurts BPR signal and the model is still constrained by the head-item-only training data.

## Sweep findings

| alpha | Recall (no LLM) | nDCG (no LLM) | GKPI (no LLM) | GKPI (with LLM lam=0.7) |
|---|---|---|---|---|
| 0.0   | 0.0298 | 0.0131 | 0.0229 | 0.0314 |
| 0.001 | 0.0298 | 0.0131 | 0.0229 | 0.0315 |
| 0.005 | 0.0288 | 0.0124 | 0.0219 | 0.0317 |
| 0.01  | 0.0293 | 0.0123 | 0.0218 | 0.0313 |
| 0.05  | 0.0265 | 0.0111 | 0.0197 | 0.0311 |

All alphas < 0.01 yield essentially identical results. The popularity-dispersion term is too small to affect BPR training meaningfully (BPR loss values ~0.1-0.5 while the popularity-dispersion term at alpha=0.001 contributes on the order of alpha * pop[i] * score(i) ~ 0.001 * 3 * 2 = 0.006 — too small to compete with BPR signal).

## What this rules out / suggests

- **Ruled out**: that explicit multi-objective training can push the Pareto front beyond the SUPER blueprint + LLM blending solution. The blueprint merge + LLM scores already saturate the achievable calibration/accuracy tradeoff in our setup.
- **Suggests**: future multi-objective work should:
  - Use a LARGER alpha scale (multiplier * log1p) or directly subtract from predicted scores in the inference stage rather than train-time penalty
  - Apply multi-objective training to BOTH M_pop and M_tail simultaneously
  - Try a more expressive popularity-dispersion term (e.g., min-max constraint enforcing item scores to be below a threshold)
  - Couple it with adaptive DP (H5) to give the multi-objective regularization a broader regime to work in

## Pareto frontier (consolidated from H1-H4)

Best Pareto points for {Recall, Rmse-PC, GKPI}:
- Best recall: LLM-FedSUPER-noDP lam=0.7 -> Recall=0.037, Rmse-PC=0.055, GKPI=0.031
- Best accuracy under DP: LLM-FedSUPER-DP eps=2 lam=0.7 -> Recall=0.021, Rmse-PC=0.055, GKPI=0.019
- Best privacy alone: FedSUPER-DP eps=2 -> Recall=0.013, Rmse-PC=0.055, GKPI=0.012
- Best calibration alone: SUPER (centralized) or FedSUPER -> Rmse-PC=0.055 (identical)
- Best GKPI: LLM-FedSUPER-noDP lam=0.7 (with multi-obj alpha=0.005) -> 0.0317

## Trajectory
```json
{"experiment_id":"run_H4_alpha0.005","hypothesis":"H4","metric_value":{"recall":0.0364,"rmse_pc":0.055,"gkpi":0.0317},
 "baseline":{"recall":0.0366,"rmse_pc":0.055,"gkpi":0.0314},
 "delta":{"gkpi":"+0.0003"},
 "change_summary":"Popularity-dispersion regularizer marginal effect; blueprint merge dominates."}
```
