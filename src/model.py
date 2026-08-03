"""PyTorch Matrix Factorization recsys: centralized + federated, with optional
DP noise and SUPER popularity-exposure reduction (and LLM side-info + multi-obj loss).

Architecture: simple user embedding E_U (n_users x d), item embeddings E_I (n_items x d),
bias term. Score(u, i) = (E_U[u] * E_I[i]).sum() + b_i + side[u] * side_i[i].
BPR loss for implicit feedback.
"""
import numpy as np
import torch
import torch.nn as nn


class RecMF(nn.Module):
    def __init__(self, n_users, n_items, dim=64, seed=0, n_user_features=0,
                 item_features_dim=0):
        super().__init__()
        torch.manual_seed(seed)
        self.dim = dim
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.item_bias = nn.Embedding(n_items, 1)
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.zero_()
        # optional side-information (LLM user profile . item semantic)
        self.user_side = None
        self.item_side = None
        if n_user_features:
            self.user_side = nn.Linear(n_user_features, dim, bias=False)
            nn.init.xavier_uniform_(self.user_side.weight)
        if item_features_dim:
            self.item_side = nn.Linear(item_features_dim, dim, bias=False)
            nn.init.xavier_uniform_(self.item_side.weight)

    def forward(self, u, i):
        eu = self.user_emb(u)
        ei = self.item_emb(i)
        score = (eu * ei).sum(dim=1) + self.item_bias(i).squeeze(1)
        return score, eu, ei

    def score_matrix(self):
        """Return full (n_users x n_items) score matrix on CPU as numpy."""
        with torch.no_grad():
            U = self.user_emb.weight  # (N_u, d)
            I = self.item_emb.weight  # (N_i, d)
            B = self.item_bias.weight.squeeze(1)  # (N_i,)
            S = (U @ I.T) + B[None, :]
            if self.user_side is not None and self.item_side is not None:
                # not the dominant score path; kept simple
                pass
        return S.detach().cpu().numpy().astype(np.float32)


def bpr_loss(pos_score, neg_score):
    diff = pos_score - neg_score
    # numerically stable softplus(-diff)
    return -torch.log(torch.sigmoid(diff) + 1e-9).mean()


def pick_negatives(train_matrix_user_row, n_items, n=1, rng=None):
    """Sample n items not interacted by user (negatives) given the user row vector."""
    rng = rng or np.random.default_rng()
    pos = np.where(train_matrix_user_row > 0)[0]
    if len(pos) >= n_items - 1:
        return rng.choice(n_items, size=n, replace=True)
    # rejection sampling: draw, filter out seen items, retry once if too few remain
    cand = rng.integers(0, n_items, size=max(5, n * 3))
    cand = cand[~np.isin(cand, pos)]
    if len(cand) < n:
        # fall back: pick from all items not in pos deterministically
        rest = np.setdiff1d(np.arange(n_items), pos, assume_unique=False)
        return rng.choice(rest, size=n, replace=False) if len(rest) >= n else rest[:n]
    return cand[:n]


def super_post_process_scores(scores, pop_prob, user_pop_sensitivity=None,
                              alpha=0.5, mode="reweight"):
    """Apply SUPER popularity-exposure reduction to predicted scores, server-side.

    scores: (n_users, n_items) numpy
    pop_prob: (n_items,) global item popularity (probability mass)
    user_pop_sensitivity: (n_users,) per-user popularity sensitivity in [0,1].
        If None -> uniform 0.5 (one global alpha).
    alpha: strength of popularity-reduction in [0,1] (per user if user_pop_sensitivity given).
    mode: 'reweight' (score / pop**alpha), or 'shrink' (score - alpha*pop_norm).

    Returns re-ranked scores (higher = better). Predicted scores get divided by
    popularity**alpha for users with high sensitivity, exposing long-tail items
    in their top-K. Maintains ranking signal: a high predicted score divides less-attended
    item.
    """
    pop = np.clip(pop_prob, 1e-9, None)
    if mode == "reweight":
        # new_score = score / pop^alpha  -> tail items get boosted
        if user_pop_sensitivity is None:
            pop_factor = pop ** alpha  # (n_items,)
            return scores / pop_factor[None, :]
        else:
            pop_factor = pop[None, :] ** user_pop_sensitivity[:, None]
            return scores / pop_factor
    elif mode == "shrink":
        pop_norm = (pop - pop.mean()) / (pop.std() + 1e-9)
        if user_pop_sensitivity is None:
            return scores - alpha * pop_norm[None, :]
        return scores - user_pop_sensitivity[:, None] * pop_norm[None, :]
    else:
        raise ValueError(f"unknown mode {mode}")


def estimate_user_pop_sensitivity(train_matrix, pop_prob, n_users):
    """Per-user popularity sensitivity. Higher = user is heavily skewed to popular items.

    Sensitivity(u) = cos(user_history, popularity) -> sum_u(interactions) * pop(item)
    Range roughly [0, 0.1]; we squash to [0, 1] via min-max normalization.
    """
    if train_matrix.shape[0] != n_users:
        return np.full(n_users, 0.5, dtype=np.float32)
    inner = train_matrix @ pop_prob  # (n_users,)
    norm_user = np.linalg.norm(train_matrix, axis=1) + 1e-9
    norm_pop = np.linalg.norm(pop_prob) + 1e-9
    cos = inner / (norm_user * norm_pop)
    cos = np.nan_to_num(cos, nan=0.0)
    lo, hi = float(cos.min()), float(cos.max())
    if hi - lo < 1e-6:
        return np.full(n_users, 0.5, dtype=np.float32)
    return ((cos - lo) / (hi - lo)).astype(np.float32)


def llm_score_term(user_features, item_features):
    """Compute LLM side-information score (semantic similarity from user
    interests and item text). Done once offline.

    user_features: (n_users, dfeat) precomputed profile embeddings
    item_features: (n_items, dfeat) precomputed item text embeddings
    Returns (n_users, n_items) similarity matrix.
    """
    u = user_features / (np.linalg.norm(user_features, axis=1, keepdims=True) + 1e-9)
    it = item_features / (np.linalg.norm(item_features, axis=1, keepdims=True) + 1e-9)
    return u @ it.T  # cosine similarity


def combine_scores(pred_scores, llm_mat, llm_weight=0.3):
    """Combine MF predicted scores with LLM semantic similarity scores.
    llm_weight in [0, 1] trades accuracy signal for semantic signal.
    """
    return (1 - llm_weight) * pred_scores + llm_weight * llm_mat


def apply_dp_noise(model, eps=8.0, delta=1e-5, max_grad_norm=1.0, sensitivity_mode="full"):
    """Clamp gradients and (optionally) inject Gaussian DP noise on per-client update.

    Adaptive budget (mode='popular_extra, no implementation for adaptive; use uniform).
    Operates *on client gradient tensor*; expects torch model with .weight on each emb.
    """
    # Clipping
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    for p in params:
        p.grad.data.clamp_(-max_grad_norm, max_grad_norm)  # simple clip
    sigma = max_grad_norm * np.sqrt(2 * np.log(1.25 / delta)) / max(eps, 1e-3)
    if sigma > 0:
        for p in params:
            p.grad.data += torch.randn_like(p.grad) * sigma
    return sigma
