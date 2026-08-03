"""H1 v2: Correct SUPER blueprint framework (Pareto partition, dual models, blueprint merge).

Variants:
  1) BPR-MF (centralized, single model)              - accuracy baseline
  2) SUPER-centralized (two BPR-MF models + merge)    - paper SUPER
  3) FedNCF (federated single model)                  - private accuracy baseline
  4) FedSUPER v2 (two FedNCF models + merge)          - our private SUPER
"""
import os, sys, json, time
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from recdata import load_ml1m, leave_one_out_split, build_train_matrix
from metrics import full_rank_eval
from super import (pareto_partition, super_blueprint_merge,
                    user_popularity_inclination, gkpi_score, mask_user_trainpos)
from train import train_centralized, train_federated

RESULT_DIR = os.path.join(ROOT, "experiments", "H1-fedsuper", "results")
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df, n_users, n_items, pop_prob, _, _ = load_ml1m(min_rating=4)
train_df, test = leave_one_out_split(df, n_users, n_items)
train_matrix = build_train_matrix(train_df, n_users, n_items)
# Integer item popularity counts for Pareto partition
pop_count = train_df.groupby('i').size().reindex(range(n_items), fill_value=0).values.astype(np.int32)
pop_prob = pop_prob.astype(np.float32)

print("\n=== Pareto partition (alpha=0.20) ===")
head_idx, tail_idx = pareto_partition(pop_count, alpha=0.20)
print(f"head size: {len(head_idx)} ({len(head_idx)/n_items*100:.1f}%), tail size: {len(tail_idx)}")
print(f"head share of total interactions: {pop_count[head_idx].sum()/pop_count.sum():.4f}")

# Build sub-matrices for each model
train_df_head = train_df[train_df['i'].isin(head_idx)].copy()
train_df_tail = train_df[train_df['i'].isin(tail_idx)].copy()
print(f"head training rows: {len(train_df_head)}, tail training rows: {len(train_df_tail)}")

def mask_trainpos(scores, train_matrix):
    s = scores.copy()
    s[train_matrix > 0] = -np.inf
    return s

def eval_topk(name, scores, train_matrix, method='argsort', reclist=None):
    if method == 'reclist' and reclist is not None:
        m = full_rank_eval(scores, test, K=10, pop_prob=pop_prob,
                            method='reclist', reclist_matrix=reclist,
                            head_idx=head_idx, train_matrix=train_matrix)
    else:
        sc_masked = mask_trainpos(scores, train_matrix)
        m = full_rank_eval(sc_masked, test, K=10, pop_prob=pop_prob,
                            head_idx=head_idx, train_matrix=train_matrix)
    print(f"{name}: " + str({k: round(v,4) for k,v in m.items() if k != 'decile_props'}))
    return m

results = {}

# ---- Variant 1: Centralized BPR-MF (single model on full dataset) ----
print("\n=== 1) Centralized BPR-MF ===")
t = time.time()
m_full = train_centralized(train_matrix, n_users, n_items, dim=64,
                            epochs=15, lr=0.05, batch_users=1024, device=DEVICE, seed=0, verbose=True)
print(f"trained in {time.time()-t:.1f}s")
results["BPR-MF"] = eval_topk("BPR-MF (full)", m_full.score_matrix(), train_matrix)

# ---- Variant 2: Centralized SUPER (two separate models + blueprint merge) ----
print("\n=== 2) Centralized SUPER ===")
print("Training head-only model...")
m_head = train_centralized(train_matrix, n_users, n_items, dim=64,
                            epochs=15, lr=0.05, batch_users=1024, device=DEVICE, seed=0, verbose=True,
                            multi_obj_alpha=None)
print("Training tail-only model...")
m_tail = train_centralized(train_matrix, n_users, n_items, dim=64,
                            epochs=15, lr=0.05, batch_users=1024, device=DEVICE, seed=0, verbose=True)
# Build head-only and tail-only training sub-matrices to actually specialize the two models.
print("\nTraining specialized head-only model on D_pop...")
sm_head = build_train_matrix(train_df_head, n_users, n_items)
m_head_spec = train_centralized(sm_head, n_users, n_items, dim=64,
                                epochs=15, lr=0.05, batch_users=1024, device=DEVICE, seed=0, verbose=True)
print("Training specialized tail-only model on D_tail...")
sm_tail = build_train_matrix(train_df_tail, n_users, n_items)
m_tail_spec = train_centralized(sm_tail, n_users, n_items, dim=64,
                                epochs=15, lr=0.05, batch_users=1024, device=DEVICE, seed=0, verbose=True)

# SUPER-centralized (specialized models + blueprint merge)
pop_scores = m_head_spec.score_matrix()  # Head-model predictions
tail_scores = m_tail_spec.score_matrix()  # Tail-model predictions
# mask train positives from each model's choices
pop_scores_m = mask_trainpos(pop_scores, train_matrix)
tail_scores_m = mask_trainpos(tail_scores, train_matrix)
reclist_super = super_blueprint_merge(pop_scores_m, tail_scores_m, train_matrix,
                                        head_idx, N=10)
results["SUPER-centralized"] = eval_topk("SUPER-centralized", pop_scores_m,
                                          train_matrix, method='reclist',
                                          reclist=reclist_super)
del m_head, m_tail, m_head_spec, m_tail_spec
gc_collect = __import__('gc'); gc_collect.collect(); torch.cuda.empty_cache()

# ---- Variant 3: Federated FedNCF ----
print("\n=== 3) Federated FedNCF ===")
t = time.time()
m_fed = train_federated(train_matrix, n_users, n_items, dim=64,
                        rounds=40, clients_per_round=256, local_epochs=2,
                        lr=0.3, dp_eps=None, max_grad_norm=1.0,
                        device=DEVICE, seed=0, verbose=True)
print(f"trained in {time.time()-t:.1f}s")
results["FedNCF"] = eval_topk("FedNCF", m_fed.score_matrix(), train_matrix)

# ---- Variant 4: Federated FedSUPER (two FedNCF models + merge) ----
print("\n=== 4) Federated FedSUPER ===")
print("Training FedNCF on D_pop...")
m_fed_head = train_federated(sm_head, n_users, n_items, dim=64,
                              rounds=40, clients_per_round=256, local_epochs=2,
                              lr=0.3, dp_eps=None, max_grad_norm=1.0,
                              device=DEVICE, seed=0, verbose=True)
print("Training FedNCF on D_tail...")
m_fed_tail = train_federated(sm_tail, n_users, n_items, dim=64,
                              rounds=40, clients_per_round=256, local_epochs=2,
                              lr=0.3, dp_eps=None, max_grad_norm=1.0,
                              device=DEVICE, seed=0, verbose=True)

pop_scores_f = mask_trainpos(m_fed_head.score_matrix(), train_matrix)
tail_scores_f = mask_trainpos(m_fed_tail.score_matrix(), train_matrix)
reclist_fedsuper = super_blueprint_merge(pop_scores_f, tail_scores_f, train_matrix,
                                          head_idx, N=10)
results["FedSUPER"] = eval_topk("FedSUPER", pop_scores_f, train_matrix,
                                method='reclist', reclist=reclist_fedsuper)

# Save
with open(os.path.join(RESULT_DIR, "metrics_h1_v2.json"), "w") as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n--- H1 v2 summary ---")
print(f"{'name':25s} {'Recall':>8s} {'NDCG':>8s} {'Gini':>8s} {'Cov':>8s} {'Nov':>8s} "
      f"{'APLT':>8s} {'LTC':>8s} {'Ent':>8s} {'RMSE-PC':>8s} {'MRMC':>8s} {'GKPI':>8s}")
for name, m in results.items():
    print(f"{name:25s} {m['recall@K']:8.4f} {m['ndcg@K']:8.4f} {m['gini_fairness']:8.4f} "
          f"{m['coverage']:8.4f} {m['novelty']:8.4f} {m['aplt']:8.4f} {m['ltc']:8.4f} "
          f"{m['entropy']:8.4f} {m['rmse_pc']:8.4f} {m['mrmc']:8.4f} {m['gkpi']:8.4f}")
