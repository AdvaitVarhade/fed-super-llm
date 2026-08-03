# H3 Protocol: LLM user profiling recovers SUPER-caused accuracy loss

## Hypothesis (locked)

Augmenting FedSUPER with LLM-generated user interest profiles (free-text -> embedding)
and using them as a side-information term in the score function will recover >=50% of
the accuracy lost to popularity-exposure reduction, while maintaining fairness/diversity
gains.

## Mechanism

LLM profiles encode semantic interest signals independent of interaction frequency.
For users whose genuine interests include popular items (e.g., a user who really likes
blockbusters AND many niche items would have a profile encoding {action, drama, ...}
largely). Their interaction-history-derived fed recsys score for popular items gets
over-suppressed by SUPER's popularity reweight. The LLM profile similarity score **re-
elevates those items appropriately** — popularity-reduction becomes user-adaptive in
a way SUPER's interaction-only signal cannot be.

## Prediction

Recall@10 with LLM-profile augmentation >= FedSUPER-noDP Recall - 2%
(i.e., >=50% of the gap [FedSUPER vs FedNCF] is recovered)
while Gini-fairness maintains >= FedSUPER-noDP - 0.05.

Reference numbers from H1:
- FedNCF-noDP Recall = 0.0402
- FedSUPER-noDP Recall = 0.0103 (gap = 0.030)
- We expect LLM-FedSUPER Recall >= 0.0103 + 0.015 = 0.025 (50% recovery).

## Setup

### LLM Profile construction

For each item in MovieLens-1M, embed its title and genres using a small Sentence-BERT
model (all-MiniLM-L6-v2). Item text: "{title} ({genres})".
For each user, summarize their training interactions: top-K genres + a sample of titles.
Use the same Sentence-BERT to embed the user profile text.
For privacy, this profile is computed locally (does not leave the user's device - simulating on-device LLM).

### Score blending

Final score(u) = (1 - lambda) * FedNCF_score(u) + lambda * cos(user_profile_emb, item_emb)
where lambda in {0.1, 0.2, 0.3, 0.5}. We sweep over lambda to find the optimal tradeoff.

### SUPER reweight

After blending, apply SUPER popularity-exposure reduction with alpha=0.3 (same as H1).

### Variants compared

1. FedSUPER (no LLM) - reproduced from H1 / H2 (no DP, no LLM).
2. LLM-FedSUPER (no DP): lambda in {0.1, 0.2, 0.3, 0.5}
3. Optional: LLM-FedSUPER-DP eps=2 (best from H2) at chosen lambda

### Compute

Sentence-Bert all-MiniLM-L6-v2 (22M params). Embed MovieLens item titles/genres once,
build user profiles on the fly. ~10 min total per dataset.

## Pre-registration

Committed to git BEFORE the experiment runs.
