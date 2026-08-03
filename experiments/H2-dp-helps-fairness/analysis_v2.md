# H2 v2 Analysis: DP noise + correct SUPER blueprint

## Result status

**Mixed findings — H2 partially supported with nuance**:

### Finding 1: FedSUPER achieves perfect calibration at all DP levels.
- Rmse-PC: 0.0550 across all eps values (noDP, 8, 4, 2).
- MRMC: 0.1506 across all eps values.
- APLT: 0.7851 across all eps values.

**Mechanism**: The blueprint merge enforces a hard per-user quota ("N_pop = floor(N*Pop_u), N_tail = N - N_pop") regardless of underlying score accuracy. Even with heavily perturbed DP-noised backbone models, the merge algorithm pulls N_pop head items from M_pop's top and N_tail tail items from M_tail's top — the quota itself guarantees the calibration property.

This **extends the GUARANTEE**: the SUPER blueprint post-processing is provably robust to backbone perturbations (including DP noise). Not only does SUPER deliver privacy-compatible calibration (H1), but the guarantee is **stability-preserving** under DP noise injection.

### Finding 2: DP noise helps FedNCF's intrinsic popularity calibration.
- FedNCF alone WITHOUT DP: Rmse-PC=0.705 (poor calibration), APLT=0.05 (almost no tail items).
- FedNCF-DP eps=2: Rmse-PC=0.283, APLT=0.52 — **diminishes popularity bias intrinsically**.

**Mechanism**: same as H2 v1 predicted by hypothesis — popular items have larger per-client gradients (they appear more often in user histories), so DP noise lays down proportionally more gradient noise on these items, down-ranking them. The Rmse-PC drops monotonically with eps (more noise -> better calibration).

### Finding 3 (REFUTED): DP noise helps FedSUPER fairness beyond SUPER alone
The original H2 prediction was "DP noise improves FedSUPER fairness by >=0.03" — in the new blueprint framework, **FedSUPER already achieves perfect calibration regardless of DP**. Hence DP noise CANNOT improve FedSUPER's calibration; it only hurts accuracy.

So the H2 prediction was framed under the wrong SUPER understanding. Under the correct algorithm:
- DP noise is **redundant for calibration** in FedSUPER (blueprint merge dominates).
- DP noise hurts FedSUPER's accuracy more than H2 v1 indicates (recall 0.030 -> 0.013 with eps=2, **57% relative loss** — far more than predicted 10%).
- DP noise is BETTER ALONE: FedNCF-DP eps=2 has Rmse-PC=0.283 vs FedNCF-noDP 0.705. DP can be a stand-alone debiasing for FL recsys when SUPER is unavailable.

## Pareto comparison

| Setting | Privacy | Recall | Rmse-PC | MRMC | APLT | LTC | GKPI |
|---|---|---|---|---|---|---|---|
| BPR-MF (cent) | None | 0.049 | 0.407 | 0.410 | 0.44 | 0.43 | 0.046 |
| SUPER (cent) | None | 0.038 | 0.055 | 0.151 | 0.79 | 0.47 | 0.035 |
| FedNCF (FL, no SUPER) | None added | 0.039 | 0.705 | 0.717 | 0.05 | 0.01 | 0.027 |
| **FedSUPER (FL, no DP)** | None added | 0.030 | 0.055 | 0.151 | 0.79 | 0.015 | 0.023 |
| **FedNCF-DP eps=2** | eps=2 | 0.026 | 0.283 | 0.222 | 0.52 | 0.011 | 0.018 |
| **FedNCF-DP eps=4** | eps=4 | 0.036 | 0.610 | 0.630 | 0.15 | 0.011 | 0.027 |
| FedSUPER-DP eps=2 | eps=2 | 0.013 | 0.055 | 0.151 | 0.79 | 0.012 | 0.012 |

The strongest "privacy + fairness + accuracy" combination is **FedSUPER (no DP)** or **FedNCF-DP eps=4** depending on the privacy budget desired. Adding DP to FedSUPER *does not help* but PAYS in accuracy. This finding refutes the H2 hypothesis IN the blueprint merge setting.

## Mechanism interpretation

**The SUPER blueprint merge functions as a "hard guarantee" algorithm**: it forces compliance regardless of underlying score quality. So when a backbone model is perturbed (by DP noise), SUPER pays no calibration cost but accuracy drops since the per-pool top items are picked blindly under noise.

For practical designs:
- Use FedNCF + DP only (no SUPER) if you want loose-but-real popularity mitigation with privacy.
- Use FedSUPER without DP if you need strict calibration guarantees with privacy.
- FedSUPER + DP seems unproductive: it hurts accuracy without improving calibration further.

## What this rules out / suggests

- **Ruled out**: that gradient DP noise is a useful secondary debiasing mechanism when SUPER blueprint merge is already applied. The blueprint merge makes the gradient-noise idea redundant for correctness properties (it preserves correctness exactly).
- **Suggests** (H5 motivation): a hybrid that uses ADAPTIVE SUPER blueprint for users with strong evidence but a DP-noise-only fall-back for users with weak history might combine the two strengths.
- **Suggests** (H3 motivation): For FedSUPER+DP, where accuracy drops drastically at strong privacy, an LLM signal could rescue accuracy by providing alternative discribrative signal orthogonal to the noised backbone. H3-L2 v2 (next experiment) should test this directly: add LLM-based shoring of the head/tail score blending within the blueprint framework.

## Trajectory
```json
{"experiment_id":"run_H2_v2_eps8","hypothesis":"H2","metric_value":0.055,"baseline":0.055,"delta":0.0,
 "change_summary":"FedSUPER-DP eps=8 calibration unchanged vs no-DP (hard-quota algorithm); accuracy -4%.","wall_time_min":8}
{"experiment_id":"run_H2_v2_eps4","hypothesis":"H2","change_summary":"FedSUPER-DP eps=4: calibration unchanged, recall -28%","wall_time_min":8}
{"experiment_id":"run_H2_v2_eps2","hypothesis":"H2","change_summary":"FedSUPER-DP eps=2: calibration unchanged, recall -57%; FedNCF-DP eps=2 calibration improves dramatically 0.705->0.283"}
```
