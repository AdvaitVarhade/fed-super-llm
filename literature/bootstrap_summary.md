# Bootstrap Summary & Hypotheses — SUPER Extension

## Research Question
Can the SUPER popularity-exposure-reduction framework be extended with Federated Learning, Differential Privacy, and LLM-based user profiling to simultaneously improve recommendation fairness, diversity, novelty, and accuracy while preserving user privacy?

## Base Paper: SUPER (Yavru et al., IEEE Access 2026, DOI 10.1109/access.2026.3671645)
- User-centric re-ranking that adapts popularity-exposure reduction to per-user popularity sensitivity.
- Improves fairness + diversity with limited accuracy loss.
- **Gap**: assumes centralized data, no privacy mechanism, no LLM signal, limited multi-objective optimization.

## Literature Gaps (from 59 papers across 6 clusters)

1. **No privacy-preserving SUPER-style method exists.** Popularity-bias mitigation literature (Abdollahpouri et al. 2019-2023, Zhang 2021 causal intervention, Klimashevskaia 2023 survey) is entirely centralized. SUPER inherits this limitation.
2. **Federated recommenders inherit popularity bias** from client gradients because popular items dominate local interaction histories (Sun et al. 2022 survey, FedNCF lineage). No work jointly performs FL training + popularity-aware re-ranking at the server.
3. **DP noise hurts accuracy but its fairness effect is unstudied.** BGTplanner (Zhang 2024) optimizes DP budget for *accuracy* in FL recsys — but no work targets *fairness/diversity* with DP. A key open question: does DP noise help long-tail items (by flattening popular-item signals) or hurt them?
4. **LLM-based recsys ignores fairness.** LLM-Rec (Lyu 2023), GenRec (Ji 2023), and surveys (Zhao 2023, Wu 2023) focus on accuracy/explainability/zero-shot. "Is ChatGPT Fair?" (Zhang 2023) is the only fairness-LLM paper and shows LLMs amplify bias. Work combining LLM user profiling with debiasing is absent.
5. **Multi-objective optimization is partial.** Multi-FR (Wu 2021) and Multi-Objective RecSys tutorial (Zheng 2021) optimize 2-3 objectives; none jointly handle fairness+diversity+novelty+accuracy under FL+DP constraints.

## Hypotheses

Each hypothesis has mechanistic reasoning (X because Y, predicting Z).

### H1: Federated SUPER (FedSUPER) — privacy without fairness loss
**Statement**: A federated version of SUPER that performs popularity-exposure reduction at the server during the aggregation step will match SUPER's fairness (Gini-fairness ≥ 0.85) while never accessing raw user data, with <5% accuracy loss vs. centralized SUPER.
**Mechanism**: Popularity-reweighting is a post-hoc re-ranking operation on candidate scores; it is data-agnostic given item-popularity statistics, so it can run server-side on Federated-FedNCF-style embeddings without exposing raw interactions.
**Prediction**: FedSUPER fairness ≥ SUPER fairness − 0.02; FedSUPER Recall@10 ≥ SUPER − 5%.
**Priority**: high

### H2: DP gradient noise shifts exposure toward long-tail (privacy improves fairness)
**Statement**: Adding Gaussian DP noise to client gradients in FedSUPER will reduce popular-item score concentration (lower popularity lift) and increase tail-item exposure, improving Gini-fairness by ≥0.03 over FedSUPER at moderate privacy (ε≤8), at the cost of <10% accuracy.
**Mechanism**: Popular items have larger gradient magnitudes (more local interactions) → receive proportionally more DP noise → their predicted scores shrink relative to tail items → exposure redistributes toward long tail.
**Prediction**: At ε=8, Gini-fairness increases by ≥0.03 and Recall@10 decreases by ≤8% vs FedSUPER-no-DP.
**Priority**: high (this would be a novel "privacy helps fairness" finding)

### H3: LLM user profiles reduce the accuracy cost of popularity mitigation
**Statement**: Augmenting FedSUPER with LLM-generated user interest profiles (free-text → embedding) and using them as a side-information term in the score function will recover ≥50% of the accuracy lost by popularity-exposure reduction, while keeping fairness/diversity gains.
**Mechanism**: LLM profiles encode semantic interest signals independent of interaction frequency, so for users whose true interests include popular items, the LLM profile re-elevates those items appropriately (unlike a pure uniform debiasing which suppresses all popular items). This makes popularity-reduction user-adaptive in a way SUPER's interaction-only signal cannot be.
**Prediction**: Recall@10 with LLM-profile augmentation ≥ SUPER-no-DP Recall − 2%, while Gini-fairness maintains ≥ baseline-FedSUPER level.
**Priority**: high (key novelty contribution)

### H4: Multi-objective scalarization dominates single-objective tuned variants
**Statement**: A scalarized multi-objective loss combining fairness, diversity, novelty, and accuracy terms (Pareto-weighted) outperforms any single-objective variant on the FDN-A composite score, and the Pareto frontier dominates fixed-weight SUPER on at least 2 of 4 axes.
**Mechanism**: SUPER optimizes fairness/accuracy as a weighted combination but does not explicitly model diversity and novelty (which partially conflict with fairness). Adding them lets the optimizer find non-dominated points that any 2-axis method misses.
**Prediction**: Multi-objective FDN-A ≥ best single-axis FDN-A by ≥0.05, and Pareto-dominance holds on ≥2 axis pairs.
**Priority**: medium

### H5: Adaptive DP budget allocation = better fairness/accuracy tradeoff than uniform
**Statement**: Allocating more DP budget to popular-item gradients (where noise has larger fairness benefit) and less to tail-item gradients gives better FDN-A than uniform allocation at the same total ε.
**Mechanism**: Since popular items carry the bias signal (H2), concentrating noise there maximizes fairness gain while preserving tail-item accuracy. BGTplanner exploits this for accuracy; we adapt it for fairness.
**Prediction**: Adaptive-ε FDN-A ≥ uniform-ε FDN-A by ≥0.03 at same total ε.
**Priority**: medium (extends BGTplanner to fairness objective)

## Evaluation Plan

### Datasets
- **MovieLens-1M** (6k users, 4k items): standard, fast on CPU/GPU.
- **Steam** (interactive, free-text reviews for LLM profiling): enables LLM-vs-no-LLM contrast.
- Optional: LastFM-2k if time permits.

### Metrics
- **Accuracy**: Recall@10, NDCG@10
- **Fairness**: Gini-fairness across item popularity deciles, Coverage (% of catalog recommended)
- **Diversity**: Intra-list diversity (avg 1-cosine-sim between recommended item embeddings)
- **Novelty**: Self-information based on global popularity (-log2 p(item))
- **Composite FDN-A**: normalized weighted sum of F, D, N, A axes. Locked upfront.

### Baselines (locked)
1. BPR-MF / FedNCF: accuracy-only, federated, no fairness.
2. SUPER (reimplemented): centralized, popularity-exposure reduction.
3. SUPER-Fed: SUPER applied federated (H1).
4. Each component ablated: −DP, −LLM, −multiobj.

### Compute
- RTX 3050 Laptop (4GB), CPU 16 cores. MovieLens-1M fits comfortably.
- LLM profiling: small open model (e.g., all-MiniLM-L6-v2 sentence embeddings for user/item text). No external API needed.

### Proxy metric for inner loop
**FDN-A composite @ top-10** (computed in <1 min per evaluation). Baselines locked before experiments.
