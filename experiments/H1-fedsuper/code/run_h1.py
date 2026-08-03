"""H1 experiment: Centralized vs Federated, with and without SUPER popularity-exposure reduction.

Variants:
  1) BPR-MF (centralized, no fairness)
  2) SUPER-centralized  (BPR-MF + popularity reweight)
  3) FedNCF (federated, no fairness)
  4) FedSUPER (FedNCF + popularity reweight)

Saves results JSON + CSV trajectory to experiments/H1-fedsuper/results/
"""
import os, sys, json, time
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from recdata import load_ml1m, leave_one_out_split, build_train_matrix
from metrics import full_rank_eval, fdn_a_composite
from model import super_post_process_scores, estimate_user_pop_sensitivity
from train import train_centralized, train_federated

RESULT_DIR = os.path.join(ROOT, "experiments", "H1-fedsuper", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| cuda?", torch.cuda.is_available())

# ---- Load data ----
df, n_users, n_items, pop_prob, umap, imap = load_ml1m(min_rating=4)
train_df, test = leave_one_out_split(df, n_users, n_items)
train_matrix = build_train_matrix(train_df, n_users, n_items)
print("train matrix:", train_matrix.shape, "nnz:", (train_matrix > 0).sum())

# Per-user training positives (for masking scores during eval)
train_pos = [set(np.where(train_matrix[u] > 0)[0].tolist()) for u in range(n_users)]
pop_prob = pop_prob.astype(np.float32)

# ---- User popularity sensitivity for SUPER re-rank ----
user_pop_sens = estimate_user_pop_sensitivity(train_matrix, pop_prob, n_users)
print("user_pop_sens: range", user_pop_sens.min(), user_pop_sens.max(), "mean", user_pop_sens.mean())

SUPER_ALPHA_REWEIGHT = 0.3
SUPER_MODE = "reweight"

def run_centralized():
    print("\n=== Centralized BPR-MF ===")
    t = time.time()
    model_c = train_centralized(train_matrix, n_users, n_items, dim=64,
                                epochs=20, lr=0.05, batch_users=1024,
                                device=DEVICE, seed=0, verbose=True)
    print(f"trained in {time.time()-t:.1f}s")
    scores_c = model_c.score_matrix()
    # mask train positives -> confident negative
    for u in range(n_users):
        scores_c[u, list(train_pos[u])] = -1e9
    metrics_c = full_rank_eval(scores_c, test, K=10, pop_prob=pop_prob)
    print("BPR-MF:", {k: round(v,4) for k,v in metrics_c.items() if k != "decile_props"})

    # SUPER-centralized
    print("\n=== SUPER-centralized ===")
    scores_super = super_post_process_scores(scores_c.copy(), pop_prob,
                                             user_pop_sensitivity=user_pop_sens,
                                             alpha=SUPER_ALPHA_REWEIGHT,
                                             mode=SUPER_MODE)
    # re-mask (reweight may have lifted train items)
    for u in range(n_users):
        scores_super[u, list(train_pos[u])] = -1e9
    metrics_s = full_rank_eval(scores_super, test, K=10, pop_prob=pop_prob)
    print("SUPER-cent:", {k: round(v,4) for k,v in metrics_s.items() if k != "decile_props"})

    return {"BPR-MF": metrics_c, "SUPER-centralized": metrics_s}

def run_federated():
    print("\n=== Federated FedNCF ===")
    t = time.time()
    model_f = train_federated(train_matrix, n_users, n_items, dim=64,
                              rounds=60, clients_per_round=512, local_epochs=2,
                              lr=0.3, dp_eps=None, max_grad_norm=1.0,
                              device=DEVICE, seed=0, verbose=True)
    print(f"trained in {time.time()-t:.1f}s")
    scores_f = model_f.score_matrix()
    for u in range(n_users):
        scores_f[u, list(train_pos[u])] = -1e9
    metrics_f = full_rank_eval(scores_f, test, K=10, pop_prob=pop_prob)
    print("FedNCF:", {k: round(v,4) for k,v in metrics_f.items() if k != "decile_props"})

    print("\n=== FedSUPER ===")
    scores_fs = super_post_process_scores(scores_f.copy(), pop_prob,
                                          user_pop_sensitivity=user_pop_sens,
                                          alpha=SUPER_ALPHA_REWEIGHT,
                                          mode=SUPER_MODE)
    for u in range(n_users):
        scores_fs[u, list(train_pos[u])] = -1e9
    metrics_fs = full_rank_eval(scores_fs, test, K=10, pop_prob=pop_prob)
    print("FedSUPER:", {k: round(v,4) for k,v in metrics_fs.items() if k != "decile_props"})
    return {"FedNCF": metrics_f, "FedSUPER": metrics_fs}

results = {}
results.update(run_centralized())
results.update(run_federated())

# Save raw metrics
out_path = os.path.join(RESULT_DIR, "metrics_h1.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved -> {out_path}")

# Build FDN-A composite against BPR-MF baseline reference
ref = results["BPR-MF"]
ref_for_norm = {
    "F": max(ref["gini_fairness"], 1e-9),
    "D": max(ref["ild"], 1e-9),
    "N": max(ref["novelty"], 1e-9),
    "A": max(ref["recall@K"] + ref["ndcg@K"], 1e-9),
}
print("\n--- FDN-A composite (normalized vs BPR-MF baseline) ---")
table = []
for name, m in results.items():
    fdna, axes = fdn_a_composite(m, ref_for_norm)
    table.append({"name": name, "FDN-A": round(fdna, 4), **{k: round(v, 4) for k, v in m.items() if k != "decile_props"},
                  **axes})
    print(f"{name:20s} FDN-A={fdna:.4f}  F={axes['F']:.2f} D={axes['D']:.2f} N={axes['N']:.2f} A={axes['A']:.2f}")

# Trajectory
import csv
with open(os.path.join(RESULT_DIR, "trajectory_h1.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=table[0].keys())
    w.writeheader()
    w.writerows(table)
print("saved trajectory_h1.csv")
