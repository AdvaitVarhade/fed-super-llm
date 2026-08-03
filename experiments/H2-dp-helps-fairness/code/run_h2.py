"""H2 experiment: DP gradient noise vs fairness/accuracy tradeoff in FedNCF.

Runs FedNCF at multiple eps values; evaluates metrics at each. Also evaluates
FedSUPER-DP (SUPER reweight on top).
"""
import os, sys, json, time, copy
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from recdata import load_ml1m, leave_one_out_split, build_train_matrix
from metrics import full_rank_eval
from model import super_post_process_scores, estimate_user_pop_sensitivity
from train import train_federated

RESULT_DIR = os.path.join(ROOT, "experiments", "H2-dp-helps-fairness", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df, n_users, n_items, pop_prob, _, _ = load_ml1m(min_rating=4)
train_df, test = leave_one_out_split(df, n_users, n_items)
train_matrix = build_train_matrix(train_df, n_users, n_items)
train_pos = [set(np.where(train_matrix[u] > 0)[0].tolist()) for u in range(n_users)]
pop_prob = pop_prob.astype(np.float32)
user_pop_sens = estimate_user_pop_sensitivity(train_matrix, pop_prob, n_users)

def mask_trainpos(scores):
    for u in range(n_users):
        scores[u, list(train_pos[u])] = -1e9
    return scores

def eval_variant(name, scores_no_mask, with_super):
    if with_super:
        scores = super_post_process_scores(scores_no_mask.copy(), pop_prob,
                                            user_pop_sensitivity=user_pop_sens,
                                            alpha=0.3, mode="reweight")
    else:
        scores = scores_no_mask.copy()
    scores = mask_trainpos(scores)
    m = full_rank_eval(scores, test, K=10, pop_prob=pop_prob)
    m["name"] = name
    return m

all_results = {}

EPS_GRID = [8.0, 4.0, 2.0, 1.0]
ROUNDS = 40
CLIENTS = 256
LR = 0.3

# 1. DP eps sweep
for eps in EPS_GRID:
    name_fed = f"FedNCF-DP_eps={eps}"
    print(f"\n=== {name_fed} ===")
    t = time.time()
    m = train_federated(train_matrix, n_users, n_items, dim=64,
                        rounds=ROUNDS, clients_per_round=CLIENTS, local_epochs=2,
                        lr=LR, dp_eps=eps, max_grad_norm=1.0,
                        device=DEVICE, seed=0, verbose=True)
    print(f"trained in {time.time()-t:.1f}s")
    sc = m.score_matrix()
    m_fed = eval_variant(name_fed, sc, with_super=False)
    m_fedsup = eval_variant(f"FedSUPER-DP_eps={eps}", sc, with_super=True)
    print(name_fed, {k: round(v,4) for k,v in m_fed.items() if k not in ("decile_props", "name")})
    print(f"FedSUPER-DP eps={eps}", {k: round(v,4) for k,v in m_fedsup.items() if k not in ("decile_props", "name")})
    all_results[name_fed] = m_fed
    all_results[f"FedSUPER-DP_eps={eps}"] = m_fedsup
    # free GPU
    del m
    import gc; gc.collect(); torch.cuda.empty_cache()

# 2. No-DP reference (re-run once for fair comparison with same random seed)
print("\n=== FedNCF (no DP, baseline) ===")
t = time.time()
m_base = train_federated(train_matrix, n_users, n_items, dim=64,
                         rounds=ROUNDS, clients_per_round=CLIENTS, local_epochs=2,
                         lr=LR, dp_eps=None, max_grad_norm=1.0,
                         device=DEVICE, seed=0, verbose=True)
print(f"trained in {time.time()-t:.1f}s")
sc_b = m_base.score_matrix()
all_results["FedNCF-noDP"] = eval_variant("FedNCF-noDP", sc_b, with_super=False)
all_results["FedSUPER-noDP"] = eval_variant("FedSUPER-noDP", sc_b, with_super=True)
del m_base; import gc; gc.collect(); torch.cuda.empty_cache()

with open(os.path.join(RESULT_DIR, "metrics_h2.json"), "w") as f:
    json.dump(all_results, f, indent=2)
print("\nsaved metrics_h2.json")

# Print table
print("\n--- H2 results ---")
print(f"{'variant':28s} {'Recall':>8s} {'NDCG':>8s} {'Gini-F':>8s} {'Cov':>8s} {'Nov':>8s} {'FDN-A':>8s}")
ref = all_results["FedNCF-noDP"]
ref_for_norm = {"F": max(ref["gini_fairness"], 1e-9), "D": max(ref["ild"], 1e-9),
                "N": max(ref["novelty"], 1e-9), "A": max(ref["recall@K"]+ref["ndcg@K"], 1e-9)}
from metrics import fdn_a_composite
for name, m in all_results.items():
    fdna, _ = fdn_a_composite(m, ref_for_norm)
    print(f"{name:28s} {m['recall@K']:8.4f} {m['ndcg@K']:8.4f} {m['gini_fairness']:8.4f} {m['coverage']:8.4f} {m['novelty']:8.4f} {fdna:8.4f}")
