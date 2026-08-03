"""H5: Adaptive DP budget (popularity-aware Gaussian noise).

Test: noise scaled by per-item popularity factor (sigma_i = sigma_base * (1 + amp * log1p(pop_i))).
Variants:
  - FedNCF-DP uniform (H2 v2 ref)
  - FedNCF-DP adaptive eps=2 amp=2
  - FedSUPER-DP uniform (H2 v2 ref)
  - FedSUPER-DP adaptive eps=2 amp=2
"""
import os, sys, json, time
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from recdata import load_ml1m, leave_one_out_split, build_train_matrix
from metrics import full_rank_eval
from super import pareto_partition, super_blueprint_merge, mask_user_trainpos
from train import train_federated

RESULT_DIR = os.path.join(ROOT, "experiments", "H5-adaptive-dp", "results")
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df, n_users, n_items, pop_prob, _, _ = load_ml1m(min_rating=4)
train_df, test = leave_one_out_split(df, n_users, n_items)
train_matrix = build_train_matrix(train_df, n_users, n_items)
pop_count = train_df.groupby('i').size().reindex(range(n_items), fill_value=0).values.astype(np.int32)
pop_prob = pop_prob.astype(np.float32)

head_idx, tail_idx = pareto_partition(pop_count, alpha=0.20)
head_set = set(head_idx.tolist())
tail_set = set(range(n_items)) - head_set

train_df_head = train_df[train_df['i'].isin(head_idx)].copy()
train_df_tail = train_df[train_df['i'].isin(tail_idx)].copy()
sm_head = build_train_matrix(train_df_head, n_users, n_items)
sm_tail = build_train_matrix(train_df_tail, n_users, n_items)


def mask_tr(scores):
    s = scores.copy(); s[train_matrix > 0] = -np.inf
    return s


def eval_reclist(name, sc_pop, sc_tail, train_matrix_local):
    sp = mask_tr(sc_pop); stt = mask_tr(sc_tail)
    rec = super_blueprint_merge(sp, stt, train_matrix_local, head_idx, N=10)
    m = full_rank_eval(sp, test, K=10, pop_prob=pop_prob,
                       reclist_matrix=rec, method='reclist',
                       head_idx=head_idx, train_matrix=train_matrix_local)
    print(f"{name}: " + str({k: round(v, 4) for k, v in m.items() if k != 'decile_props'}))
    return m


def eval_standalone(name, sc, tm):
    sc_m = mask_tr(sc)
    m = full_rank_eval(sc_m, test, K=10, pop_prob=pop_prob,
                       head_idx=head_idx, train_matrix=tm)
    print(f"{name}: " + str({k: round(v, 4) for k, v in m.items() if k != 'decile_props'}))
    return m


results = {}
ROUNDS = 25; CLIENTS = 256; LR = 0.3

# FedNCF-DP adaptive eps=2 pop_amp=2
print("\n=== FedNCF-DP adaptive eps=2 amp=2 ===")
t = time.time()
m = train_federated(train_matrix, n_users, n_items, dim=64,
                    rounds=ROUNDS, clients_per_round=CLIENTS, local_epochs=2,
                    lr=LR, dp_eps=2.0, max_grad_norm=1.0,
                    device=DEVICE, seed=0, verbose=True,
                    adaptive_pop_dp=True, pop_amplification=2.0, pop_prob=pop_prob)
print(f"trained in {time.time()-t:.1f}s")
results["FedNCF-DP_adaptive_eps=2_amp=2"] = eval_standalone("FedNCF-DP_adaptive_eps=2_amp=2",
                                                              m.score_matrix(), train_matrix)
del m; import gc; gc.collect(); torch.cuda.empty_cache()

# FedSUPER-DP adaptive (head and tail models both with adaptive noise)
print("\n=== FedSUPER-DP adaptive eps=2 amp=2 ===")
t = time.time()
print("Training adaptive head model...")
m_pop = train_federated(sm_head, n_users, n_items, dim=64,
                        rounds=ROUNDS, clients_per_round=CLIENTS, local_epochs=2,
                        lr=LR, dp_eps=2.0, max_grad_norm=1.0,
                        device=DEVICE, seed=0, verbose=True,
                        adaptive_pop_dp=True, pop_amplification=2.0, pop_prob=pop_prob)
sc_pop = m_pop.score_matrix()
del m_pop; import gc; gc.collect(); torch.cuda.empty_cache()

print("Training adaptive tail model...")
m_tail = train_federated(sm_tail, n_users, n_items, dim=64,
                         rounds=ROUNDS, clients_per_round=CLIENTS, local_epochs=2,
                         lr=LR, dp_eps=2.0, max_grad_norm=1.0,
                         device=DEVICE, seed=0, verbose=True,
                         adaptive_pop_dp=True, pop_amplification=2.0, pop_prob=pop_prob)
sc_tail = m_tail.score_matrix()
del m_tail; import gc; gc.collect(); torch.cuda.empty_cache()
print(f"trained in {time.time()-t:.1f}s")
results["FedSUPER-DP_adaptive_eps=2_amp=2"] = eval_reclist("FedSUPER-DP_adaptive_eps=2_amp=2",
                                                            sc_pop, sc_tail, train_matrix)

with open(os.path.join(RESULT_DIR, "metrics_h5.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n--- H5 summary ---")
print(f"{'name':45s} {'Recall':>8s} {'nDCG':>8s} {'APLT':>8s} {'LTC':>8s} {'RMSE-PC':>8s} {'MRMC':>8s} {'GKPI':>8s}")
for name, m in results.items():
    print(f"{name:45s} {m['recall@K']:8.4f} {m['ndcg@K']:8.4f} {m['aplt']:8.4f} {m['ltc']:8.4f} {m['rmse_pc']:8.4f} {m['mrmc']:8.4f} {m['gkpi']:8.4f}")

print("\nCompare to H2 v2 uniform eps=2: FedSUPER-DP_eps=2 -> Recall=0.0131, GKPI=0.0120")
print("                       : FedNCF-DP_eps=2   -> Recall=0.0260, GKPI=0.0182")
