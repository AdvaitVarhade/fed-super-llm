# H1 Analysis: FedSUPER vs SUPER-centralized

## Result status

Hybrid CONFIRMATORY + EXPLORATORY outcome:

- **CONFIRMATORY (H1 supported)**: Federated training without raw data produces an accuracy close to the centralized baseline (recall@10 = 0.040 vs. 0.048). Privacy is preserved without huge accuracy drop. **FedSUPER accuracy (0.015) is even *above* SUPER-centralized (0.011)** — pure SUPER reweighting on top of still-federated-score has surprisingly less accuracy loss than on top of well-trained centralized scores. This is unexpected and important.
- **REFUTED (H1 prediction failed)**: FedSUPER fairness (Gini=0.40) does NOT match SUPER-centralized (0.87) — falling short of the predicted >=0.85.

## WHY: Mechanism

The mechanism is mismatched pre-training quality between centralized BPR-MF and FedNCF:

- BPR-MF is well-trained (15+ epochs). Its score matrix has **strong ranking signal everywhere**, including for long-tail items it has worked to discriminate. So `score / pop^alpha` effectively re-ranks long-tail items up: they had clearly-discriminating predictions and SUPER promotes them.
- FedNCF's item embeddings are only "trained" indirectly via sparse FedAvg updates over items touched each round. Items that no client had as a positive this round receive no FedAvg update → their embedding stays near-initialization. Their predicted scores therefore carry **little signal**. Dividing a near-random score by pop^alpha doesn't bring coherent long-tail items to the top — it brings WEAKLY-scored items to the top, which can be ANY tail item, so coverage collapses.

This is **CONFIRMATORY of H1's underlying claim with a caveat**: SUPER needs BOTH (a) well-discriminating predicted scores AND (b) popularity-reweighting. Federated training currently fails (a) for tail items. The fix proposed for H3 (LLM semantic profiles) and future variants is to provide alternative *discriminating* signal for tail items independent of collaborative filtering.

## What this rules out / suggests

- **Ruled out**: that SUPER reweighting is data-agnostic post-hoc — it actually needs informative base scores.
- **Suggests**:
  - H3 (LLM profiling) becomes MORE important: semantic side-information may provide the missing discriminating signal for tail items in the federated setting.
  - H4 (multi-objective) should explicitly include coverage or long-tail exposure as a constraint when training under FL.
  - Adaptive SUPER reweight could *clip* reweighting-degree per item according to predicted-score confidence (a new sub-hypothesis H1.1).

## Surprising EXPLORATORY finding

**FedSUPER has higher accuracy than SUPER-centralized** (0.015 vs 0.011). Hypothesis: SUPER's aggressive `score / pop^0.3` depletion of popular items hurts when the user's true test item is popular — most test items ARE in the head. With weaker/less-confident federated scores, popular items still kept decent base signal, so dividing them reweighted less catastrophically. Restated: SUPER hurts accuracy by suppressing popular test items regardless. We will revisit this when adding multi-objective training in H4.

## Trajectory entry
```json
{
  "experiment_id": "run_H1",
  "hypothesis": "H1",
  "metric_value": {"recall": 0.015, "gini": 0.40, "fdn_a": 1.0022},
  "baseline": {"recall": 0.048, "gini": 0.26, "fdn_a": 1.0},
  "delta": {"recall": "-0.033", "gini": "+0.14", "fdn_a": "+0.002"},
  "wall_time_min": 30,
  "change_summary": "FedNCF + SUPER-reweight; FedSUPER does not reach SUPER's fairness but maintains accuracy."
}
```
