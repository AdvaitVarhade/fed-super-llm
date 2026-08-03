# H1 Protocol: Federated SUPER (FedSUPER)

## Hypothesis (locked)
A federated version of SUPER that performs popularity-exposure reduction at the server during aggregation will match SUPER's fairness (Gini-fairness >= 0.85) while never accessing raw user data, with <5% accuracy loss vs. centralized SUPER.

## Mechanism
Popularity-reweighting is a post-hoc re-ranking operation on candidate scores; it is data-agnostic given item-popularity statistics, so it can run server-side on federated model embeddings without exposing raw interactions.

## Prediction
- FedSUPER Gini-fairness >= SUPER Gini-fairness - 0.02
- FedSUPER Recall@10 >= SUPER - 5%

## Variants Compared
1. **BPR-MF (centralized, no fairness)** — accuracy baseline, expected to have lowest fairness.
2. **SUPER-centralized** = BPR-MF + popularity-exposure reduction re-rank (alpha=0.5, mode=reweight, user_pop_sensitivity from train history).
3. **FedNCF (federated, no fairness)** — same embedding model but trained FedNCF-style.
4. **FedSUPER** = FedNCF + same popularity-exposure reduction re-rank.

## Setup
- Data: MovieLens-1M, rating>=4 positive, leave-one-out split.
- Architecture: RecMF, dim=64, BPR loss, Adam lr=0.05.
- Cent: 10 epochs.
- Fed: 30 rounds, 64 clients/round, 2 local epochs.
- Eval: full ranking (n_items=3533), K=10. Metrics: recall@10, ndcg@10, gini_fairness, coverage, ILD, novelty, FDN-A.
- Reference for FDN-A normalization: the BPR-MF baseline run.
- Random seed: 0 (deterministic).

## Stop criterion
After running all 4 variants, compare. If FedSUPER achieves the predicted fairness + accuracy bounds, H1 is CONFIRMATORY. Otherwise record what we learned.

## Pre-registration
This protocol is committed to git BEFORE the experiment runs.
