"""H3c: Sweep SUPER alpha with LLM and DP fusion.

Final aggregate design: combine FOUR signals multiplicatively for our best recipe.
Sweep ALPHA in {0.1, 0.2, 0.3, 0.5} with DP eps=2 (best from H2) and LLM lambda=0.5.
"""
import os, sys, json
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from recdata import load_ml1m, leave_one_out_split, build_train_matrix
from metrics import full_rank_eval, fdn_a_composite
from model import super_post_process_scores, estimate_user_pop_sensitivity
from train import train_federated

RESULT_DIR = os.path.join(ROOT, "experiments", "H3-llm-profiling", "results")

df, n_users, n_items, pop_prob, u_map, i_map = load_ml1m(min_rating=4)
train_df, test = leave_one_out_split(df, n_users, n_items)
train_matrix = build_train_matrix(train_df, n_users, n_items)
train_pos = [np.where(train_matrix[u] > 0)[0] for u in range(n_users)]
pop_prob = pop_prob.astype(np.float32)
user_pop_sens = estimate_user_pop_sensitivity(train_matrix, pop_prob, n_users)

# Load LLM
item_emb = np.load(os.path.join(RESULT_DIR, "item_emb.npz"))["item_emb"].astype(np.float32)
user_emb = np.load(os.path.join(RESULT_DIR, "user_emb.npz"))["user_emb"].astype(np.float32)
llm_scores = user_emb @ item_emb.T

# Load FedNCF base
fed_scores = np.load(os.path.join(RESULT_DIR, "fed_scores.npz"))["fed_scores"]

def mask_trainpos(scores):
    for u in range(n_users):
        scores[u, train_pos[u].tolist()] = -1e9
    return scores

# Pre-trained FedNCF + LLM lambda = 0.5 raw
lam = 0.5
raw_blended = (1.0 - lam) * fed_scores + lam * llm_scores

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5]
results = {}
print("\n--- H3c: alpha sweep with LLM lambda=0.5 (no DP) ---")
for a in ALPHAS:
    name = f"LLM-lam0.5-a={a}"
    sc = super_post_process_scores(raw_blended.copy(), pop_prob,
                                    user_pop_sensitivity=user_pop_sens,
                                    alpha=a, mode="reweight")
    sc = mask_trainpos(sc)
    m = full_rank_eval(sc, test, K=10, pop_prob=pop_prob)
    m["name"] = name
    print(f"{name:30s} Recall={m['recall@K']:.4f} Gini={m['gini_fairness']:.4f} "
          f"Cov={m['coverage']:.4f} Nov={m['novelty']:.4f} ILD={m['ild']:.4f}")
    results[name] = m

# Also a=0 with no LLM (raw FedNCF) as accuracy baseline
m_nol = mask_trainpos(fed_scores.copy())
m_acc = full_rank_eval(m_nol, test, K=10, pop_prob=pop_prob)
m_acc["name"] = "FedNCF (alpha=0, no LLM)"
results[m_acc["name"]] = m_acc

with open(os.path.join(RESULT_DIR, "metrics_h3c.json"), "w") as f:
    json.dump(results, f, indent=2)

# FDN-A vs accuracy baseline
ref = m_acc
ref_for_norm = {"F": max(ref["gini_fairness"], 1e-9), "D": max(ref["ild"], 1e-9),
                "N": max(ref["novelty"], 1e-9), "A": max(ref["recall@K"]+ref["ndcg@K"], 1e-9)}
print(f"\n{'name':30s} {'Recall':>8s} {'Gini-F':>8s} {'Cov':>8s} {'Nov':>8s} {'FDN-A':>8s}")
for name, m in results.items():
    fdna, _ = fdn_a_composite(m, ref_for_norm)
    print(f"{name:30s} {m['recall@K']:8.4f} {m['gini_fairness']:8.4f} "
          f"{m['coverage']:8.4f} {m['novelty']:8.4f} {fdna:8.4f}")
