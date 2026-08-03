"""H3b: LLM profiles applied as a POST-SUPER rescue re-rank.

Hypothesis update: SUPER reweight shoves popular items to rank-100. Pre-blending
LLM signal doesn't help because we still re-weight after. Instead, AFTER SUPER
reweight, add LLM similarity score back as a rescue bonus for popular items,
un-shoving genuinely-semantically-fit popular items.

Final score(u) = SUPER_reweight(FedScore(u)) + bonus * LLM_cos(u, popular_test_item)

Implementation: two-pass scoring
  S1 = SUPER_reweight(FedScore)  # first pass: popularity-exposure reduction
  S2 = S1 + bonus * LLM_cos  # second pass: rescue

Bonus we sweep over.
"""
import os, sys, json, time, csv
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from recdata import load_ml1m, leave_one_out_split, build_train_matrix
from metrics import full_rank_eval, fdn_a_composite
from model import super_post_process_scores, estimate_user_pop_sensitivity

RESULT_DIR = os.path.join(ROOT, "experiments", "H3-llm-profiling", "results")
DEVICE = "cpu"

df, n_users, n_items, pop_prob, u_map, i_map = load_ml1m(min_rating=4)
train_df, test = leave_one_out_split(df, n_users, n_items)
train_matrix = build_train_matrix(train_df, n_users, n_items)
train_pos = [np.where(train_matrix[u] > 0)[0] for u in range(n_users)]
pop_prob = pop_prob.astype(np.float32)
user_pop_sens = estimate_user_pop_sensitivity(train_matrix, pop_prob, n_users)

# Load previously saved FedNCF scores and LLM item/user embeddings are rebuilt (already
# saved into metrics_h3.json context). Recompute via cached artifacts.
print("\nLoading item_emb + user_emb + fed_model.artifacts...")
h3_data = np.load(os.path.join(RESULT_DIR, "item_emb.npz"))
item_emb = h3_data["item_emb"].astype(np.float32)
print("item_emb:", item_emb.shape)

# We need user_emb and fed_scores. We don't have user_emb saved, but we DO need to
# re-train fed once OR load from cached. To stay synced, run via cached metrics file?
# Simplest: re-build user_emb via the same Sentence-Bert + profile texts as run_h3.
# Since embeddings are deterministic with same seed and same model, save once now.
USER_EMB_PATH = os.path.join(RESULT_DIR, "user_emb.npz")
if not os.path.exists(USER_EMB_PATH):
    ML1M_MOVIES = os.path.join(ROOT, "data", "ml-1m", "movies.dat")
    with open(ML1M_MOVIES, "r", encoding="latin-1") as f:
        raw_movies = [line.strip().split("::") for line in f if line.strip()]
    item_text = [None] * n_items
    for row in raw_movies:
        mid, title, genres = row
        mid = int(mid)
        if mid in i_map:
            idx = i_map[mid]
            item_text[idx] = f"{title} ({genres.replace('|', ', ')})".strip()
    for i, t in enumerate(item_text):
        if t is None:
            item_text[i] = f"item_{i}"
    item_text = [t[:200] for t in item_text]

    from sentence_transformers import SentenceTransformer
    model_sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                       device=DEVICE)
    # Build user profile texts
    print("Building user profiles...")
    profile_texts = []
    for u in range(n_users):
        pos = train_pos[u]
        if len(pos) == 0:
            profile_texts.append(f"user_{u}")
            continue
        genre_counter = {}
        sample_titles = []
        for i in pos[:200]:
            t = item_text[i]
            sample_titles.append(t.split(" (")[0])
            gstr = t.split("(")[-1].rstrip(")").split(",") if "(" in t else []
            for g in gstr:
                g = g.strip()
                if g:
                    genre_counter[g] = genre_counter.get(g, 0) + 1
        top_genres = sorted(genre_counter.items(), key=lambda x: -x[1])[:8]
        genre_str = ", ".join(g for g, _ in top_genres) if top_genres else "unknown"
        rng = np.random.default_rng(u + 42)
        if len(sample_titles) > 5:
            sample_titles = list(rng.choice(sample_titles, size=5, replace=False))
        profile_texts.append(f"A user who likes {genre_str} movies including: {', '.join(sample_titles)}.")
    user_emb = model_sbert.encode(profile_texts, batch_size=64, show_progress_bar=False,
                                  convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    np.savez(USER_EMB_PATH, user_emb=user_emb)
else:
    user_emb = np.load(USER_EMB_PATH)["user_emb"].astype(np.float32)
print("user_emb:", user_emb.shape)

llm_scores = user_emb @ item_emb.T  # (n_users, n_items)
print("llm_scores range", llm_scores.min(), llm_scores.max())

# Re-train FedNCF (or load cached scores). Since we never saved the scores file,
# train fresh.
print("\nTraining FedNCF base...")
import torch
from train import train_federated
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
m_fed = train_federated(train_matrix, n_users, n_items, dim=64,
                  rounds=40, clients_per_round=256, local_epochs=2,
                  lr=0.3, dp_eps=None, max_grad_norm=1.0,
                  device=DEVICE, seed=0, verbose=True)
fed_scores = m_fed.score_matrix()
del m_fed
import gc; gc.collect(); torch.cuda.empty_cache()
np.savez(os.path.join(RESULT_DIR, "fed_scores.npz"), fed_scores=fed_scores)

def mask_trainpos(scores):
    for u in range(n_users):
        scores[u, train_pos[u].tolist()] = -1e9
    return scores

# Compute baseline SUPER-reweighted score once
super_scores = super_post_process_scores(fed_scores.copy(), pop_prob,
                                          user_pop_sensitivity=user_pop_sens,
                                          alpha=0.3, mode="reweight")

# Two-pass rescue sweep: bonus multiplier for LLM similarity
results = {}
BONUSES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
for bonus in BONUSES:
    name = f"LLM-Rescue-bonus={bonus}"
    resc = super_scores + bonus * llm_scores
    resc = mask_trainpos(resc)
    m = full_rank_eval(resc, test, K=10, pop_prob=pop_prob)
    m["name"] = name
    print(f"{name}: {[(k, round(v,4)) for k,v in m.items() if k not in ('decile_props', 'name')]}")
    results[name] = m

# Reference: FedSUPER (no LLM rescue)
ref_super = mask_trainpos(super_scores.copy())
m_ref = full_rank_eval(ref_super, test, K=10, pop_prob=pop_prob)
m_ref["name"] = "FedSUPER-noDP (rescuebaseline)"
results["FedSUPER-noDP (rescuebaseline)"] = m_ref

with open(os.path.join(RESULT_DIR, "metrics_h3b.json"), "w") as f:
    json.dump(results, f, indent=2)

# FDN-A table
ref = m_ref
ref_for_norm = {"F": max(ref["gini_fairness"], 1e-9), "D": max(ref["ild"], 1e-9),
                "N": max(ref["novelty"], 1e-9), "A": max(ref["recall@K"]+ref["ndcg@K"], 1e-9)}
print("\n--- H3b results (FDN-A vs FedSUPER baseline) ---")
print(f"{'name':35s} {'Recall':>8s} {'NDCG':>8s} {'Gini-F':>8s} {'Cov':>8s} {'Nov':>8s} {'FDN-A':>8s}")
for name, m in results.items():
    fdna, _ = fdn_a_composite(m, ref_for_norm)
    print(f"{name:35s} {m['recall@K']:8.4f} {m['ndcg@K']:8.4f} {m['gini_fairness']:8.4f} {m['coverage']:8.4f} {m['novelty']:8.4f} {fdna:8.4f}")
