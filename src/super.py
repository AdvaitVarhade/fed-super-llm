"""Correct SUPER framework implementation (Yavru et al., IEEE Access 2026).

Pipeline:
  1. Pareto partition of item catalog into (Head set H, Tail set T) by cumulative
     interaction volume threshold alpha=0.20 (NOT 20% of items; 20% of interactions).
  2. Train TWO separate recommenders: M_pop on D_pop (head interactions only),
     M_tail on D_tail (tail interactions only).
  3. Per user, compute popularity inclination Pop_u = |C_u ∩ H| / |C_u|
     (fraction of user's interactions that are in the head set).
  4. Recommendation top-N for user u:
       N_pop = floor(N * Pop_u), N_tail = N - N_pop
       C_pop = top-N_pop items from M_pop scores for u
       C_tail = top-N_tail items from M_tail scores for u
       Blueprint B = head/tail indicator sequence of user's training history
                    Lu sorted by descending preference (or by timestamp).
       RecList = merge(C_pop, C_tail) following blueprint, falling back to the
                 other pool when a quota is exhausted. (hard quota / soft blueprint.)
"""
import numpy as np


def pareto_partition(pop, alpha=0.20):
    """Return (head_idx, tail_idx) where head items account for alpha fraction
        of total interactions.

    pop: (n_items,) interaction counts per item
    alpha: cumulative-volume threshold (default 0.20 per paper)
    """
    total = pop.sum()
    threshold = total * alpha
    order = np.argsort(-pop)  # descending interaction count
    cum = np.cumsum(pop[order])
    # Head = items in `order` whose cumulative interaction volume is <= threshold
    # (Include at least one item to avoid degenerate empty head).
    n_head = int(np.searchsorted(cum, threshold, side="right")) + 1
    n_head = max(1, min(n_head, len(order)))
    head_idx = order[:n_head]
    tail_idx = order[n_head:]
    # Handle 0-popularity items -> tail
    zero_pop = np.where(pop == 0)[0]
    if len(zero_pop):
        already = set(head_idx.tolist()) | set(tail_idx.tolist())
        for z in zero_pop:
            if z not in already:
                tail_idx = np.append(tail_idx, z)
    return head_idx, tail_idx


def user_popularity_inclination(train_matrix, head_idx):
    """Pop_u = |C_u ∩ H| / |C_u| per user.

    train_matrix: (n_users, n_items) integer interaction counts
    head_idx: array of Head item indices from pareto_partition
    Returns (n_users,) array of Pop_u in [0, 1].
    """
    head_set = set(head_idx.tolist())
    user_interactions = (train_matrix > 0).astype(np.float32)
    per_user_count = user_interactions.sum(axis=1)
    head_count = user_interactions[:, list(head_set)].sum(axis=1)
    pop_u = np.where(per_user_count > 0, head_count / np.maximum(per_user_count, 1.0), 0.0)
    return pop_u.astype(np.float32)


def super_blueprint_merge(scores_pop, scores_tail, train_matrix, head_idx, N=10,
                          blueprint_order="preference"):
    """Blueprint-based merge to construct top-N per user (Yavru et al., Algorithm 2).

    Inputs are score matrices restricted to the appropriate item subset:
      scores_pop: (n_users, n_items) full matrix, but only head items used.
      scores_tail: (n_users, n_items) full matrix, but only tail items used.
    train_matrix: (n_users, n_items) used for blueprint construction.
    head_idx / not used / blueprint ordering:
      'preference' -> sort each user's training history by descending interaction count
      'timestamp' -> (requires ts col; not used here)
    Returns: (n_users, N) array of selected item indices (the RecList).

    We respect hard quota (N_pop, N_tail); the blueprint is used as the merge order
    for which pool to draw from next, with deterministic fall-back to the other pool.
    """
    n_users = train_matrix.shape[0]
    head_set = set(head_idx.tolist())
    tail_set = set(range(train_matrix.shape[1])) - head_set

    # Mask scores outside their pool: set non-pool items to -inf
    pop_mask = np.full(scores_pop.shape, -np.inf, dtype=np.float32)
    pop_mask[:, list(head_set)] = scores_pop[:, list(head_set)]
    tail_mask = np.full(scores_tail.shape, -np.inf, dtype=np.float32)
    tail_mask[:, list(tail_set)] = scores_tail[:, list(tail_set)]

    # Per-user list of training item indices sorted by descending preference count
    # (we use interaction counts as a proxy for preference).
    out = np.zeros((n_users, N), dtype=np.int64) - 1

    for u in range(n_users):
        row_count = train_matrix[u]
        # Blueprint = head/tail indicator at each rank in user's historical items
        # Sorted descending by interaction count
        if row_count.sum() == 0:
            # fallback: take top-N from mixed model (we can pick tail-only)
            cand = np.argsort(-tail_mask[u])[:N]
            out[u] = cand[:N]
            continue
        # user's items sorted by descending count
        u_items_sorted = np.argsort(-row_count)
        u_items_sorted = u_items_sorted[row_count[u_items_sorted] > 0]
        blueprint = np.array([1 if i in head_set else 0 for i in u_items_sorted], dtype=np.int8)

        # Hard quota
        Pop_u = blueprint.mean() if len(blueprint) > 0 else 0.5
        N_pop = int(np.floor(N * Pop_u))
        N_tail = N - N_pop
        # Cap pools to user's potential scores
        C_pop = np.argsort(-pop_mask[u])[:max(N_pop, 0)]
        C_tail = np.argsort(-tail_mask[u])[:max(N_tail, 0)]
        idx_pop = 0
        idx_tail = 0
        rec = []
        for k in range(min(N, len(blueprint))):
            if blueprint[k] == 1 and idx_pop < N_pop and idx_pop < len(C_pop):
                rec.append(int(C_pop[idx_pop])); idx_pop += 1
            elif blueprint[k] == 0 and idx_tail < N_tail and idx_tail < len(C_tail):
                rec.append(int(C_tail[idx_tail])); idx_tail += 1
            elif idx_pop < N_pop and idx_pop < len(C_pop):
                rec.append(int(C_pop[idx_pop])); idx_pop += 1
            elif idx_tail < N_tail and idx_tail < len(C_tail):
                rec.append(int(C_tail[idx_tail])); idx_tail += 1
            else:
                break
        # Fill remaining (greedy residual quota)
        while len(rec) < N and idx_pop < N_pop and idx_pop < len(C_pop):
            rec.append(int(C_pop[idx_pop])); idx_pop += 1
        while len(rec) < N and idx_tail < N_tail and idx_tail < len(C_tail):
            rec.append(int(C_tail[idx_tail])); idx_tail += 1
        # If still short (e.g., user had < N pos items or thresholds hit), fill from best overall fallback
        if len(rec) < N:
            # fallback: any items by combined score
            combined = np.maximum(pop_mask[u], tail_mask[u])
            fallback = np.argsort(-combined)
            for it in fallback:
                if it not in rec:
                    rec.append(int(it))
                    if len(rec) >= N:
                        break
        out[u] = rec[:N]
    return out


def mask_user_trainpos(scores, train_matrix):
    """Set scores[u, train item i] = -inf so they don't appear in final lists.

    scores: (n_users, n_items) numpy
    train_matrix: (n_users, n_items) binary or counts
    """
    masked = scores.copy()
    masked[train_matrix > 0] = -np.inf
    return masked


def gkpi_score(ndcg, aplt, entropy, novelty, ltc):
    """Compute SUPER-paper-style General KPI = arithmetic mean of harmonic means
    of nDCG with each beyond-accuracy metric.

    H(x, y) = 2xy / (x + y); fallback to 0 if x+y == 0.
    """
    def H(x, y):
        return (2 * x * y) / (x + y) if (x + y) > 0 else 0.0
    return float(np.mean([H(ndcg, aplt), H(ndcg, entropy), H(ndcg, novelty), H(ndcg, ltc)]))
