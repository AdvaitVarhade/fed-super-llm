"""Shared utilities: data loading, train/eval splits, interaction matrices.

MovieLens-1M loader. Returns sparse interaction dict-based structures that
the rest of the codebase consumes.
"""
import os, numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ML1M = os.path.join(ROOT, "data", "ml-1m")

def load_ml1m(min_rating=4):
    """Load MovieLens-1M. Treat rating>=min_rating as positive interaction."""
    rpath = os.path.join(ML1M, "ratings.dat")
    with open(rpath, "r", encoding="latin-1") as f:
        raw = [line.strip().split("::") for line in f if line.strip()]
    df = pd.DataFrame(raw, columns=["user", "item", "rating", "ts"]).astype({
        "user": "int32", "item": "int32",
        "rating": "int32", "ts": "int64"})
    df = df[df["rating"] >= min_rating].copy()

    # Reindex users/items to [0, n)
    u_ids = sorted(df["user"].unique())
    i_ids = sorted(df["item"].unique())
    u_map = {u: i for i, u in enumerate(u_ids)}
    i_map = {it: i for i, it in enumerate(i_ids)}
    df["u"] = df["user"].map(u_map).astype("int32")
    df["i"] = df["item"].map(i_map).astype("int32")
    n_users, n_items = len(u_map), len(i_map)

    # Global item popularity (used by SUPER-popularity-exposure-reduction)
    pop = df.groupby("i").size().reindex(range(n_items), fill_value=0).values.astype(np.float32)
    pop_prob = pop / max(pop.sum(), 1)

    print(f"ML-1M: {n_users} users, {n_items} items, {len(df)} positive interactions "
          f"(rating>={min_rating})")
    return df.reset_index(drop=True), n_users, n_items, pop_prob, u_map, i_map


def leave_one_out_split(df, n_users, n_items, holdout_ratio_for_train_filter=0.0):
    """Per-user leave-one-out splitting -> train / test. Holdout = last interaction.

    Returns train_df, test_dict {user: [items]} and per-user positive-item set for train.
    """
    df = df.sort_values("ts").reset_index(drop=True)
    train, test = [], {}
    for u, grp in df.groupby("u"):
        rows = grp.to_dict("records")
        test[u] = [rows[-1]["i"]]
        for r in rows[:-1]:
            train.append(r)
    train_df = pd.DataFrame(train)
    return train_df, test


def build_train_matrix(train_df, n_users, n_items):
    """Dense interaction frequency matrix used as model input (n_users x n_items)."""
    M = np.zeros((n_users, n_items), dtype=np.float32)
    for u, i in zip(train_df["u"].values, train_df["i"].values):
        M[u, i] += 1
    return M
