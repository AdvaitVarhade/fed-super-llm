"""H3 experiment: LLM-based user profiling for FedSUPER accuracy recovery.

Pipeline:
  1. Build item text from ML-1M movies.dat: "{title} ({genres})".
  2. Use sentence-transformers MiniLM-L6-v2 to embed all items. Save to disk.
  3. For each user, profile text = user-top genres + sample of liked titles. Embed profiles.
  4. Score_blend = (1-lambda) * FedNCF_scores + lambda * cos(user_emb, item_emb_emb)
  5. Apply SUPER reweight (alpha=0.3) on top.
  6. Eval at lambda grid.

We reuse the FedNCF weights from a trained model (saved). Train fresh here for reproducibility.
"""
import os, sys, json, time, csv
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from recdata import load_ml1m, leave_one_out_split, build_train_matrix
from metrics import full_rank_eval, fdn_a_composite
from model import super_post_process_scores, estimate_user_pop_sensitivity
from train import train_federated

RESULT_DIR = os.path.join(ROOT, "experiments", "H3-llm-profiling", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

ML1M_MOVIES = os.path.join(ROOT, "data", "ml-1m", "movies.dat")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load MovieLens data ----------
df, n_users, n_items, pop_prob, u_map, i_map = load_ml1m(min_rating=4)
train_df, test = leave_one_out_split(df, n_users, n_items)
train_matrix = build_train_matrix(train_df, n_users, n_items)
train_pos = [np.where(train_matrix[u] > 0)[0] for u in range(n_users)]
pop_prob = pop_prob.astype(np.float32)
user_pop_sens = estimate_user_pop_sensitivity(train_matrix, pop_prob, n_users)

# ---------- Build item text and LLM item embeddings ----------
print("\n=== Building Sentence-BERT item embeddings ===")
with open(ML1M_MOVIES, "r", encoding="latin-1") as f:
    raw_movies = [line.strip().split("::") for line in f if line.strip()]
print(f"raw movies: {len(raw_movies)}")
# Build (orig_movie_id -> text) then map through i_map
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
# truncate text length for batching efficiency
item_text = [t[:200] for t in item_text]
print("sample item texts:")
for i in [0, 1, 2, 100, 2000]:
    print(" ", i, "->", item_text[i])

ITEM_EMB_PATH = os.path.join(RESULT_DIR, "item_emb.npz")
print("\nLoading Sentence-Bert all-MiniLM-L6-v2...")
from sentence_transformers import SentenceTransformer
model_sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                   device=DEVICE)
print("Encoding item texts...")
item_emb = model_sbert.encode(item_text, batch_size=64, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
print("item_emb shape:", item_emb.shape)
np.savez(ITEM_EMB_PATH, item_emb=item_emb)

# ---------- Build per-user profiles and embeddings ----------
print("\n=== Building per-user profile embeddings ===")
def build_user_profile(u):
    """Profile text: top genres from their items + 5 liked sample titles."""
    pos = train_pos[u]
    if len(pos) == 0:
        return f"user_{u}"
    # extract genres from item_text (text between parens)
    genre_counter = {}
    sample_titles = []
    for i in pos[:200]:  # cap history
        t = item_text[i]
        sample_titles.append(t.split(" (")[0])
        gstr = t.split("(")[-1].rstrip(")").split(",") if "(" in t else []
        for g in gstr:
            g = g.strip()
            if g:
                genre_counter[g] = genre_counter.get(g, 0) + 1
    top_genres = sorted(genre_counter.items(), key=lambda x: -x[1])[:8]
    genre_str = ", ".join(g for g, _ in top_genres) if top_genres else "unknown"
    # sample 5 titles
    rng = np.random.default_rng(u + 42)
    if len(sample_titles) > 5:
        sample_titles = list(rng.choice(sample_titles, size=5, replace=False))
    return f"A user who likes {genre_str} movies including: {', '.join(sample_titles)}."

print("Encoding user profiles (this takes ~1-2 min)...")
profile_texts = [build_user_profile(u) for u in range(n_users)]
user_emb = model_sbert.encode(profile_texts, batch_size=64, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
print("user_emb shape:", user_emb.shape)

# LLM score matrix: cosine similarity (both normalized -> dot product)
llm_scores = user_emb @ item_emb.T  # (n_users, n_items)
print("llm_scores: range", llm_scores.min(), llm_scores.max(), "mean", llm_scores.mean())

# ---------- Train FedNCF (cached) ----------
print("\n=== Training FedNCF base (no DP) ===")
t = time.time()
model_fed = train_federated(train_matrix, n_users, n_items, dim=64,
                  rounds=40, clients_per_round=256, local_epochs=2,
                  lr=0.3, dp_eps=None, max_grad_norm=1.0,
                  device=DEVICE, seed=0, verbose=True)
print(f"trained in {time.time()-t:.1f}s")
fed_scores = model_fed.score_matrix()
del model_fed
import gc; gc.collect(); torch.cuda.empty_cache()

def mask_trainpos(scores):
    for u in range(n_users):
        scores[u, train_pos[u].tolist()] = -1e9
    return scores

# ---------- Sweep lambda ----------
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
results = {}
for lam in LAMBDAS:
    name = f"LLM-FedSUPER-lam={lam}"
    print(f"\n=== {name} ===")
    raw = (1.0 - lam) * fed_scores + lam * llm_scores
    # apply SUPER weight
    super_scores = super_post_process_scores(raw.copy(), pop_prob,
                                              user_pop_sensitivity=user_pop_sens,
                                              alpha=0.3, mode="reweight")
    super_scores = mask_trainpos(super_scores)
    m = full_rank_eval(super_scores, test, K=10, pop_prob=pop_prob)
    m["name"] = name
    print("SUPER-LLM:", {k: round(v,4) for k,v in m.items() if k not in ('decile_props', 'name')})
    results[name] = m

# Also evaluate raw (no SUPER) at one lambda to see ground impact
no_super = mask_trainpos(0.5 * fed_scores + 0.5 * llm_scores)
m_nosup = full_rank_eval(no_super, test, K=10, pop_prob=pop_prob)
m_nosup["name"] = "LLM-FedNCF-lam=0.5 (no SUPER)"
print("LLM-FedNCF (no SUPER, lam=0.5):", {k: round(v,4) for k,v in m_nosup.items() if k not in ('decile_props', 'name')})
results["LLM-FedNCF-lam=0.5 (no SUPER)"] = m_nosup

# Reference baseline for FDN-A normalization
ref_blank = full_rank_eval(mask_trainpos(fed_scores.copy()), test, K=10, pop_prob=pop_prob)
results["FedNCF-noDP (reference)"] = {**ref_blank, "name": "FedNCF-noDP (reference)"}

# Add H1 FedSUPER (no DP) reference too
ref_fedsuper = super_post_process_scores(fed_scores.copy(), pop_prob,
                                          user_pop_sensitivity=user_pop_sens,
                                          alpha=0.3, mode="reweight")
ref_fedsuper = mask_trainpos(ref_fedsuper)
m_fedsuper = full_rank_eval(ref_fedsuper, test, K=10, pop_prob=pop_prob)
m_fedsuper["name"] = "FedSUPER-noDP (reference)"
results["FedSUPER-noDP (reference)"] = m_fedsuper

# Save metrics
with open(os.path.join(RESULT_DIR, "metrics_h3.json"), "w") as f:
    json.dump(results, f, indent=2)

# FDN-A table
ref = m_fedsuper
ref_for_norm = {"F": max(ref["gini_fairness"], 1e-9), "D": max(ref["ild"], 1e-9),
                "N": max(ref["novelty"], 1e-9), "A": max(ref["recall@K"]+ref["ndcg@K"], 1e-9)}
print("\n--- H3 results (FDN-A normalized vs FedSUPER-noDP) ---")
print(f"{'name':40s} {'Recall':>8s} {'NDCG':>8s} {'Gini-F':>8s} {'Cov':>8s} {'Nov':>8s} {'FDN-A':>8s}")
table = []
for name, m in results.items():
    fdna, ax = fdn_a_composite(m, ref_for_norm)
    print(f"{name:40s} {m['recall@K']:8.4f} {m['ndcg@K']:8.4f} {m['gini_fairness']:8.4f} {m['coverage']:8.4f} {m['novelty']:8.4f} {fdna:8.4f}")
    table.append({"name": name, "recall": m["recall@K"], "ndcg": m["ndcg@K"],
                  "gini": m["gini_fairness"], "coverage": m["coverage"],
                  "novelty": m["novelty"], "fdn_a": fdna})
with open(os.path.join(RESULT_DIR, "trajectory_h3.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=table[0].keys())
    w.writeheader()
    w.writerows(table)
print("\ntrajectory written to trajectory_h3.csv")
