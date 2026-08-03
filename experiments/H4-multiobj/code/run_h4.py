"""H4: Multi-objective training during M_pop.

Train M_pop with popularity-dispersion regularizer, M_tail with BPR baseline.
Sweep alpha_popdisp in {0.0, 0.001, 0.005, 0.01, 0.05}.
Apply blueprint merge + (optionally) LLM blending.
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

RESULT_DIR = os.path.join(ROOT, "experiments", "H4-multiobj", "results")
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

# LLM scores cached from H3
LLM_DIR = os.path.join(ROOT, "experiments", "H3-llm-profiling", "results")
item_emb = np.load(os.path.join(LLM_DIR, "item_emb.npz"))["item_emb"].astype(np.float32)
user_emb = np.load(os.path.join(LLM_DIR, "user_emb.npz"))["user_emb"].astype(np.float32)
llm_full = user_emb @ item_emb.T
llm_pop_only = np.zeros(llm_full.shape, dtype=np.float32); llm_pop_only[:, list(head_set)] = llm_full[:, list(head_set)]
llm_tail_only = np.zeros(llm_full.shape, dtype=np.float32); llm_tail_only[:, list(tail_set)] = llm_full[:, list(tail_set)]


def mask_tr(scores):
    s = scores.copy(); s[train_matrix > 0] = -np.inf
    return s


def eval_reclist(name, sc_pop, sc_tail, lam=0.0):
    if lam > 0:
        sc_pop = (1.0 - lam) * sc_pop + lam * llm_pop_only
        sc_tail = (1.0 - lam) * sc_tail + lam * llm_tail_only
    sp = mask_tr(sc_pop); stt = mask_tr(sc_tail)
    rec = super_blueprint_merge(sp, stt, train_matrix, head_idx, N=10)
    m = full_rank_eval(sp, test, K=10, pop_prob=pop_prob,
                       reclist_matrix=rec, method='reclist',
                       head_idx=head_idx, train_matrix=train_matrix)
    print(f"{name}: " + str({k: round(v, 4) for k, v in m.items() if k != 'decile_props'}))
    return m


results = {}

# Sweep alpha_popdisp on M_pop
ALPHAS = [0.0, 0.001, 0.005, 0.01, 0.05]
ROUNDS = 25; CLIENTS = 256; LR = 0.3

for alpha in ALPHAS:
    print(f"\n=== Training FedSUPER (alpha_popdisp={alpha}) ===")
    t = time.time()
    m_pop = train_federated(sm_head, n_users, n_items, dim=64,
                            rounds=ROUNDS, clients_per_round=CLIENTS, local_epochs=2,
                            lr=LR, dp_eps=None, max_grad_norm=1.0,
                            device=DEVICE, seed=0, verbose=True,
                            multi_obj_alpha=alpha, pop_prob=pop_prob)
    print(f"head trained in {time.time()-t:.1f}s")
    sc_pop = m_pop.score_matrix()
    del m_pop; import gc; gc.collect(); torch.cuda.empty_cache()

    # Tail model: BPR baseline (no multi-obj)
    t = time.time()
    m_tail = train_federated(sm_tail, n_users, n_items, dim=64,
                              rounds=ROUNDS, clients_per_round=CLIENTS, local_epochs=2,
                              lr=LR, dp_eps=None, max_grad_norm=1.0,
                              device=DEVICE, seed=0, verbose=True)
    print(f"tail trained in {time.time()-t:.1f}s")
    sc_tail = m_tail.score_matrix()
    del m_tail; import gc; gc.collect(); torch.cuda.empty_cache()

    # Eval without LLM
    results[f"FedSUPER-multi_alpha={alpha}"] = eval_reclist(f"FedSUPER-multi_alpha={alpha}", sc_pop, sc_tail)
    # Eval with LLM lam=0.7 (best from H3)
    results[f"LLM-FedSUPER-multi_alpha={alpha}_lam=0.7"] = eval_reclist(
        f"LLM-FedSUPER-multi_alpha={alpha}_lam=0.7", sc_pop, sc_tail, lam=0.7)

with open(os.path.join(RESULT_DIR, "metrics_h4.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n--- H4 summary ---")
print(f"{'name':45s} {'Recall':>8s} {'nDCG':>8s} {'APLT':>8s} {'LTC':>8s} {'RMSE-PC':>8s} {'MRMC':>8s} {'GKPI':>8s}")
for name, m in results.items():
    print(f"{name:45s} {m['recall@K']:8.4f} {m['ndcg@K']:8.4f} {m['aplt']:8.4f} {m['ltc']:8.4f} {m['rmse_pc']:8.4f} {m['mrmc']:8.4f} {m['gkpi']:8.4f}")
