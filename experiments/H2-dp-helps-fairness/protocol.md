# H2 Protocol: DP gradient noise shifts exposure toward long tail (privacy improves fairness)

## Hypothesis (locked)

Adding Gaussian DP noise to client gradients in FedNCF will reduce popular-item score concentration and increase tail-item exposure, improving Gini-fairness by >=0.03 over FedNCF (no DP) at moderate privacy (eps<=8), at the cost of <10% recall.

## Mechanism

Popular items have larger gradient magnitudes (more local interactions at clients) -> receive proportionally more DP noise (Gaussian mechanism scales with gradient L2 norm) -> their predicted scores shrink relative to tail items -> exposure redistributes toward long tail.

## Prediction

At eps=8, Gini-fairness(FedNCF-DP) >= Gini-fairness(FedNCF) + 0.03 and Recall@10(FedNCF-DP) >= 0.9 * Recall@10(FedNCF).

## Setup

H1 produced FedNCF baseline (no DP) numbers. Here we train FedNCF with DP at multiple eps values: eps = {1, 2, 4, 8}. Also evaluate the SUPER reweight on top (FedSUPER-DP-eps) at each eps.

- Speed optimization: use reduced rounds (40) and clients (256), since sanity loss showed signs of plateau around round 40-50.
- Other parameters identical to H1 FedNCF.

## Variants compared

1. FedNCF (no DP) — same as H1 baseline. We will just copy from H1 results for reference.
2. FedNCF + DP eps=8
3. FedNCF + DP eps=4
4. FedNCF + DP eps=2
5. FedNCF + DP eps=1
6. FedSUPER + DP at each eps (same alpha=0.3 SUPER reweight as H1)

## Stop criterion

If at least one eps shows Gini improvement >=0.03 with Recall loss <10%, H2 CONFIRMATORY. If ALL variants show fairness drop or >10% recall loss, H2 REFUTED (DP doesn't help fairness, or only noise that doesn't differentiate pop vs tail).

## Pre-registration

Committed to git BEFORE the experiment runs.
