# H5 Protocol: Adaptive DP budget allocation (popularity-aware)

## Hypothesis (locked)
Allocating more DP noise to popular-item gradients and less to tail-item gradients will improve the FDN-A composite over uniform-allocation at the same total privacy budget epsilon. This mirrors BGTplanner's intuition that noise concentrated on head items maximizes fairness gains while preserving tail accuracy.

## Mechanism
Popular items have larger gradient magnitudes (more local interactions at clients). Adaptive DP scales sigma[i] = sigma_base * pop_factor[i] where pop_factor is high for popular items and low for tail. This means MORE noise on popular-item gradients (suppressing their scores) and LESS noise on tail-item gradients (preserving accuracy).

## Implementation
Add `adaptive_pop_dp=True` and `pop_amplification=2.0` arguments to `train_federated`. sigma_i = sigma_base * (1 + pop_amplification * log1p(pop[i]*100) / max_log_pop) so popular items get up to 3x the noise.

## Variants compared
1. FedNCF-DP uniform eps=2 (H2 v2 baseline)
2. FedNCF-DP adaptive eps=2 pop_amp=2
3. FedSUPER-DP uniform eps=2 (H2 v2 baseline)
4. FedSUPER-DP adaptive eps=2 pop_amp=2

## Stop criterion
If adaptive variant shows >=0.005 GKPI improvement over uniform at same nominal eps, H5 CONFIRMED.

## Pre-registration
Locked BEFORE running.
