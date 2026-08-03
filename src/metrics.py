"""Evaluation metrics for recommender systems.

Locks the evaluation criteria upfront to prevent unconscious metric gaming.

Metrics implemented:
  Accuracy:  Recall@K, NDCG@K
  Fairness:  Gini-fairness across item popularity deciles, Catalog Coverage
  Diversity: Intra-list diversity (avg 1-cos items)
  Novelty:  Mean self-information -log2(p(item))
  Composite: FDN-A = mean of {F_norm, D_norm, N_norm, (Re@K_norm + ndcg@K_norm)/2}
"""
import numpy as np


# ---------- Accuracy ----------
def recall_at_k(scores, test_dict, K=10, n_items=None, per_user_negs=None,
                num_neg_sample=99, rng=None):
    """Standard leave-one-out Recall@K with sampled negatives.

    scores: (n_users, n_items) predicted score matrix on CPU.
    test_dict: {user: [held_out_pos_item]}
    Per user, sample `num_neg_sample` negatives and rank the true item vs them.
    """
    rng = rng or np.random.default_rng(42)
    hits = 0
    n_users = len(test_dict)
    for u, gt in test_dict.items():
        if not gt:
            continue
        true_item = gt[0]
        all_items = list(range(scores.shape[1]))
        # negatives = items user hasn't interacted with
        seen = per_user_negs.get(u, set()) if per_user_negs else set()
        cand_pool = [i for i in all_items if i not in seen and i != true_item]
        negs = rng.choice(cand_pool, size=min(num_neg_sample, len(cand_pool)), replace=False)
        items = np.concatenate([[true_item], negs])
        score_items = scores[u, items]
        topk = K if K <= len(items) else len(items)
        topk_idx = np.argpartition(-score_items, topk - 1)[:topk]
        if 0 in topk_idx[np.argsort(-score_items[topk_idx])][:topk]:
            hits += 1
    return hits / n_users


def ndcg_at_k(scores, test_dict, K=10, per_user_negs=None, num_neg_sample=99, rng=None):
    """Leave-one-out NDCG@K with sampled negatives (binary relevance)."""
    rng = rng or np.random.default_rng(42)
    total = 0.0
    n_users = len(test_dict)
    all_items = list(range(scores.shape[1]))
    for u, gt in test_dict.items():
        if not gt:
            continue
        true_item = gt[0]
        seen = per_user_negs.get(u, set()) if per_user_negs else set()
        cand_pool = [i for i in all_items if i not in seen and i != true_item]
        negs = rng.choice(cand_pool, size=min(num_neg_sample, len(cand_pool)), replace=False)
        items = np.concatenate([[true_item], negs])
        score_items = scores[u, items]
        order = np.argsort(-score_items)[:K]
        # rank position 0 means 1st; if true item at index 0 in `items`
        rank_of_true = np.where(order == 0)[0]
        if len(rank_of_true) == 0:
            continue
        total += 1.0 / np.log2(rank_of_true[0] + 2.0)  # +2 because rank starts at 0
    return total / n_users


def full_rank_eval(scores, test_dict, K=10, item_emb=None, pop_prob=None,
                  head_idx=None, train_matrix=None, reclist_matrix=None,
                  method="argsort"):
    """Full ranking evaluation over ALL items, returning all 4 axis scores **+
       paper-compatible metrics** (APLT, LTC, Entropy, Rmse-PC, MRMC, GKPI).

    Two modes:
      (A) method='argsort' (default): top-K per user via score ranking.
      (B) method='reclist': reclist_matrix is (n_users, K) array of selected
          item indices (e.g., from SUPER blueprint merge). Test is held out item.
    """
    K_eff = K
    n_users, n_items = scores.shape

    if method == "reclist" and reclist_matrix is not None:
        topk = reclist_matrix
    else:
        # rank top-K per user from scores
        topk_idx = np.argpartition(-scores, K, axis=1)[:, :K]
        rows = np.arange(n_users)[:, None]
        order_vals = scores[rows, topk_idx]
        sort_idx = np.argsort(-order_vals, axis=1)
        topk = topk_idx[rows, sort_idx]

    # ---- Accuracy: Recall means: is GT in user's top-K? (full rank)
    rec_hits = 0
    ndcg_sum = 0.0
    for u, gt in test_dict.items():
        if not gt:
            continue
        t = gt[0]
        if t in topk[u]:
            rec_hits += 1
            rank = np.where(topk[u] == t)[0][0]
            ndcg_sum += 1.0 / np.log2(rank + 2.0)
    rec_at_k = rec_hits / max(len(test_dict), 1)
    ndcg = ndcg_sum / max(len(test_dict), 1)

    # ---- Fairness: Gini-fairness across item popularity deciles (higher = fairer)
    rec_count = np.bincount(topk.flatten(), minlength=n_items)
    pop = pop_prob if pop_prob is not None else np.ones(n_items) / n_items
    sort = np.argsort(pop)
    decile_size = max(n_items // 10, 1)
    decile_props = []
    for d in range(10):
        idx = sort[d * decile_size:(d + 1) * decile_size]
        decile_props.append(rec_count[idx].sum() / max((topk.size), 1))
    props = np.array(decile_props)
    mean = props.mean() if props.mean() > 0 else 1e-9
    gini = 1.0 - np.abs(props - mean).sum() / (2.0 * mean * len(props))
    coverage = (rec_count > 0).mean()

    # ---- Paper-compatible metrics (APLT, LTC, Entropy, Novelty, Rmse-PC, MRMC, GKPI)
    pop_per_item = pop  # pop_prob normalized to [0,1] sum
    # Reconstruct integer interaction counts if pop_prob is normalized counts
    if train_matrix is not None:
        item_pop_count = train_matrix.sum(axis=0)
    else:
        item_pop_count = pop_per_item * max(pop_per_item.sum(), 1) * n_items

    # APLT: per-user fraction of recommended items that are tail items
    # LTC: fraction of unique tail items seen across all users' top-N
    if head_idx is not None:
        head_set = set(head_idx.tolist())
        head_set_arr = np.array(sorted(head_set), dtype=np.int64)
        is_tail_item = np.ones(n_items, dtype=np.int8)
        is_tail_item[head_set_arr] = 0
        # per-user APLT
        aplt_per_user = is_tail_item[topk].mean(axis=1)  # (n_users,)
        aplt = float(aplt_per_user.mean())
        # long-tail coverage (% of tail items appearing at least once)
        rec_tail_items = np.where(is_tail_item == 1)[0]
        rec_tail_unique = np.unique(topk)
        rec_tail_unique_in_tail = np.intersect1d(rec_tail_unique, rec_tail_items)
        ltc = float(len(rec_tail_unique_in_tail) / max(len(rec_tail_items), 1))
        head_set_np = np.array(list(head_set))
        # Rmse-PC: RMSE between fraction of head items in user's recommendations vs Pop_u (historical)
        if train_matrix is not None:
            Pop_u = (train_matrix[:, head_set_arr].sum(axis=1) /
                     np.maximum(train_matrix.sum(axis=1), 1))
            # fraction of head items in user's RecList
            head_in_rec = is_tail_item[topk] == 0  # equal 1 when item IS head (not tail)
            head_frac_rec = head_in_rec.mean(axis=1)
            rmse_pc = float(np.sqrt(((head_frac_rec - Pop_u) ** 2).mean()))
        else:
            rmse_pc = 0.0
        # MRMC: mean rank miscalibration = mean over positions of |head_frac_rec[k] - Pop_u|
        # Interpretation: difference between cumulative head exposure in top-k and user's Pop_u.
        # We use the paper-aligned definition: average over k in [1..K] of cumulative bias.
        if train_matrix is not None:
            head_indicators = (is_tail_item[topk] == 0).astype(np.float32)
            cum_head = np.cumsum(head_indicators, axis=1) / (np.arange(1, K + 1)[None, :])
            mrmc_per_user = np.abs(cum_head - Pop_u[:, None]).mean(axis=1)
            mrmc = float(mrmc_per_user.mean())
        else:
            mrmc = 0.0
    else:
        aplt = 0.0
        ltc = float(coverage)
        rmse_pc = 0.0
        mrmc = 0.0

    # Entropy of recommendation distribution over items (Shannon)
    item_count = rec_count.astype(np.float32)
    p_item = item_count / max(item_count.sum(), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent_terms = np.where(p_item > 0, p_item * np.log2(p_item), 0.0)
    entropy = float(-ent_terms.sum())

    # ---- Diversity: intra-list diversity via item similarity (cosine)
    if item_emb is not None:
        ilds = []
        for u in range(n_users):
            sel = topk[u]
            embs = item_emb[sel]
            if len(sel) > 1:
                norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
                sim = norm @ norm.T
                off = sim - np.eye(len(sel))
                ilds.append(off[np.triu_indices(len(sel), k=1)].mean())
            else:
                ilds.append(0.0)
        ild = 1.0 - np.mean(ilds)
    else:
        ild = 1.0 - (1.0 / K_eff)

    # ---- Novelty: mean self-information -log2(p) over recommended items
    if pop_prob is not None:
        nov_per_rec = -np.log2(pop_prob[topk] + 1e-12)
        novelty = float(nov_per_rec.mean())
    else:
        novelty = float(np.log2(n_users))

    # ---- GKPI (per SUPER paper Eq.2): arithmetic mean of harmonic(nDCG each beyond-accuracy)
    def H(x, y):
        return (2 * x * y) / (x + y) if (x + y) > 0 else 0.0
    gkpi = float(np.mean([H(ndcg, aplt), H(ndcg, entropy), H(ndcg, novelty), H(ndcg, ltc)]))

    return {
        "recall@K": float(rec_at_k),
        "ndcg@K": float(ndcg),
        "gini_fairness": float(np.clip(gini, 0, 1)),
        "coverage": float(coverage),
        "ild": float(np.clip(ild, 0, 1)),
        "novelty": novelty,
        "aplt": float(aplt),
        "ltc": float(ltc),
        "entropy": entropy,
        "rmse_pc": rmse_pc,
        "mrmc": mrmc,
        "gkpi": gkpi,
        "decile_props": [float(x) for x in decile_props],
    }


COMPOSITE_WEIGHTS = {"F": 0.25, "D": 0.25, "N": 0.25, "A": 0.25}


def fdn_a_composite(metrics, ref_for_norm=None):
    """Compose the four axes into FDN-A. We use stable normalization against a
    reference dict (e.g., the FINAL baseline numbers) so deltas across runs are
    interpretable. Falls back to self-reference when no reference provided.

    Each reference axis is keyed as 'F','D','N','A' inside a tiny dict.
    """
    if ref_for_norm is None:
        ref_for_norm = dict.fromkeys(("F", "D", "N", "A"), None)
    def safe(v):
        return v if v and v > 1e-9 else 1.0
    ref = {
        "F": safe(ref_for_norm.get("F")),
        "D": safe(ref_for_norm.get("D")),
        "N": safe(ref_for_norm.get("N")),
        "A": safe(ref_for_norm.get("A")),
    }
    if all(ref_for_norm.get(k) is None for k in ("F","D","N","A")):
        ref["F"] = safe(metrics["gini_fairness"])
        ref["D"] = safe(metrics["ild"])
        ref["N"] = safe(metrics["novelty"])
        ref["A"] = safe(metrics["recall@K"] + metrics["ndcg@K"])
    F = min(metrics["gini_fairness"] / ref["F"], 2.0)
    D = min(metrics["ild"] / ref["D"], 2.0)
    N = min(metrics["novelty"] / ref["N"], 2.0)
    A = min((metrics["recall@K"] + metrics["ndcg@K"]) / ref["A"], 2.0)
    fdna = (COMPOSITE_WEIGHTS["F"] * F + COMPOSITE_WEIGHTS["D"] * D
            + COMPOSITE_WEIGHTS["N"] * N + COMPOSITE_WEIGHTS["A"] * A)
    return float(fdna), {"F": F, "D": D, "N": N, "A": A}
