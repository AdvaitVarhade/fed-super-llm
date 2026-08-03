"""H2 v2: DP noise + correct SUPER blueprint framework.

Build the proper SUPER post-processing over:
  - Two federated models (M_pop, M_tail), each trained with DP at eps.
  - DP applied per-update on local item+user gradients in FedNCF.

Variants:
  - FedNCF-noDP / FedSUPER-noDP baseline (no DP)
  - FedNCF-DP eps={8,4,2} / FedSUPER-DP eps={8,4,2}
"""
import os, sys, json, time, copy
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from recdata import load_ml1m, leave_one_out_split, build_train_matrix
from metrics import full_rank_eval
from super import pareto_partition, super_blueprint_merge, mask_user_trainpos
from train import train_federated

RESULT_DIR = os.path.join(ROOT, "experiments", "H2-dp-helps-fairness", "results")
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df, n_users, n_items, pop_prob, _, _ = load_ml1m(min_rating=4)
train_df, test = leave_one_out_split(df, n_users, n_items)
train_matrix = build_train_matrix(train_df, n_users, n_items)
pop_count = train_df.groupby('i').size().reindex(range(n_items), fill_value=0).values.astype(np.int32)
pop_prob = pop_prob.astype(np.float32)

head_idx, tail_idx = pareto_partition(pop_count, alpha=0.20)
print(f"head size: {len(head_idx)} ({len(head_idx)/n_items*100:.1f}%)")

train_df_head = train_df[train_df['i'].isin(head_idx)].copy()
train_df_tail = train_df[train_df['i'].isin(tail_idx)].copy()
sm_head = build_train_matrix(train_df_head, n_users, n_items)
sm_tail = build_train_matrix(train_df_tail, n_users, n_items)
print(f"head rows: {len(train_df_head)}, tail rows: {len(train_df_tail)}")

ROUNDS = 25
CLIENTS = 256
LR = 0.3
N = 10

def mask_tr(scores):
    s = scores.copy(); s[train_matrix > 0] = -np.inf
    return s

def eval_topk(name, scores_pop, scores_tail):
    sc_pm = mask_tr(scores_pop)
    sc_tm = mask_tr(scores_tail)
    reclist = super_blueprint_merge(sc_pm, sc_tm, train_matrix, head_idx, N=N)
    pop_used_for_eval = sc_pm  # only scores used for ranking purposes; super_blueprint_merge produces reclist
    m = full_rank_eval(sc_pm, test, K=N, pop_prob=pop_prob,
                       reclist_matrix=reclist, method='reclist',
                       head_idx=head_idx, train_matrix=train_matrix)
    print(f"{name}: " + str({k: round(v, 4) for k, v in m.items() if k != 'decile_props'}))
    return m

results = {}

# ----- FedNCF-noDP-only / FedNCF-DP-only baselines: also eval standalone (no SUPER) -----
def train_and_eval_pair(name_prefix, dp_eps):
    print(f"\n=== {name_prefix} (dp_eps={dp_eps}) ===")
    t = time.time()
    print("Training federated head model...")
    m_pop = train_federated(sm_head, n_users, n_items, dim=64,
                            rounds=ROUNDS, clients_per_round=CLIENTS, local_epochs=2,
                            lr=LR, dp_eps=dp_eps, max_grad_norm=1.0,
                            device=DEVICE, seed=0, verbose=True)
    print(f"head trained in {time.time()-t:.1f}s")
    sc_pop = m_pop.score_matrix()

    t = time.time()
    print("Training federated tail model...")
    m_tail = train_federated(sm_tail, n_users, n_items, dim=64,
                              rounds=ROUNDS, clients_per_round=CLIENTS, local_epochs=2,
                              lr=LR, dp_eps=dp_eps, max_grad_norm=1.0,
                              device=DEVICE, seed=0, verbose=True)
    print(f"tail trained in {time.time()-t:.1f}s")
    sc_tail = m_tail.score_matrix()

    # No-SUPER standalone: just take the average of head and tail scores? No - we should train one
    # combined fedNCF for "no SUPER" baseline comparison. For DP we train one FedNCF on full data once.
    # We'll reuse those results separately.
    del m_pop, m_tail
    import gc; gc.collect(); torch.cuda.empty_cache()

    # FedSUPER with DP at this eps
    ress_name = f"FedSUPER-DP_eps={dp_eps}"
    print(f"Evaluating {ress_name}...")
    m_super = eval_topk(ress_name, sc_pop, sc_tail)
    return dict(sc_pop=sc_pop, sc_tail=sc_tail, m_super=m_super)

def train_one_fed(name, dp_eps):
    print(f"\n=== {name} (dp_eps={dp_eps}) ===")
    t = time.time()
    m = train_federated(train_matrix, n_users, n_items, dim=64,
                        rounds=ROUNDS, clients_per_round=CLIENTS, local_epochs=2,
                        lr=LR, dp_eps=dp_eps, max_grad_norm=1.0,
                        device=DEVICE, seed=0, verbose=True)
    print(f"trained in {time.time()-t:.1f}s")
    sc = m.score_matrix()
    sc_m = mask_tr(sc)
    metrics = full_rank_eval(sc_m, test, K=N, pop_prob=pop_prob,
                              head_idx=head_idx, train_matrix=train_matrix)
    print(f"{name}:", {k: round(v, 4) for k, v in metrics.items() if k != 'decile_props'})
    del m; import gc; gc.collect(); torch.cuda.empty_cache()
    return metrics

# ----- Sweep DP eps over FedSUPER and FedNCF -----
for eps in [None, 8.0, 4.0, 2.0]:
    name_no = "FedNCF-noDP" if eps is None else f"FedNCF-DP_eps={eps}"
    results[name_no] = train_one_fed(name_no, eps)
    if eps is not None:
        pair = train_and_eval_pair(f"FedSUPER-DP eps={eps}", dp_eps=eps)
        results[f"FedSUPER-DP_eps={eps}"] = pair["m_super"]
    else:
        pair = train_and_eval_pair("FedSUPER-noDP", dp_eps=None)
        results["FedSUPER-noDP"] = pair["m_super"]

# Save
with open(os.path.join(RESULT_DIR, "metrics_h2_v2.json"), "w") as f:
    json.dump(results, f, indent=2)

# Summary
print("\n--- H2 v2 summary ---")
print(f"{'name':28s} {'Recall':>8s} {'nDCG':>8s} {'APLT':>8s} {'LTC':>8s} {'RMSE-PC':>8s} {'MRMC':>8s} {'GKPI':>8s}")
for name, m in results.items():
    print(f"{name:28s} {m['recall@K']:8.4f} {m['ndcg@K']:8.4f} {m['aplt']:8.4f} {m['ltc']:8.4f} {m['rmse_pc']:8.4f} {m['mrmc']:8.4f} {m['gkpi']:8.4f}")
