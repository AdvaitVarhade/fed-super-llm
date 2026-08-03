"""H3 v2: LLM user-profiling for accuracy recovery in the FedSUPER + DP setting.

The correct SUPER framework showed adding DP to FedSUPER hurts accuracy badly
(recall 0.030 -> 0.013 at eps=2) because the noisy backbone models can no longer
discriminate within-head / within-tail candidate pools.

Hypothesis: Adding LLM-derived semantic similarity score-boost to the head-model
predictions (since head-item embeddings are limited to ~2.1% of catalog and LLMs
can discriminate semantic MovieLens genre/title matches for them much more cleanly)
will recover accuracy without hurting SUPER's perfect calibration.

Specifically: we augment the M_pop predicted scores with LLM semantic similarity
(UserProfile.emb * Item.textContent.emb), then run the same blueprint merge.

We sweep over head-model LLM blend weight (lambda_head).
LLM augmentation for tail model is also tested.

Variants tested:
  1) FedSUPER-noDP (ref from H1 v2)
  2) FedSUPER-DP eps=2 (ref from H2 v2, also reproduced here)
  3) LLM-FedSUPER-noDP, lambda_head in {0.3, 0.5, 0.7}
  4) LLM-FedSUPER-DP eps=2, lambda_head in {0.3, 0.5, 0.7}
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

RESULT_DIR = os.path.join(ROOT, "experiments", "H3-llm-profiling", "results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Load data & LLM embeddings (reuse from H3 v1 cache) ----
df, n_users, n_items, pop_prob, u_map, i_map = load_ml1m(min_rating=4)
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
print(f"head size: {len(head_idx)} ({len(head_idx)/n_items*100:.1f}%), head rows: {len(train_df_head)}, tail rows: {len(train_df_tail)}")

# LLM item embeddings + user embeddings
item_emb = np.load(os.path.join(RESULT_DIR, "item_emb.npz"))["item_emb"].astype(np.float32)
user_emb = np.load(os.path.join(RESULT_DIR, "user_emb.npz"))["user_emb"].astype(np.float32)
llm_scores_full = user_emb @ item_emb.T
print("LLM score range:", llm_scores_full.min(), llm_scores_full.max())

# Slice LLM scores by pool
llm_pop = np.zeros(llm_scores_full.shape, dtype=np.float32)
llm_pop[:, list(head_set)] = llm_scores_full[:, list(head_set)]
llm_tail_full = np.zeros(llm_scores_full.shape, dtype=np.float32)
llm_tail_full[:, list(tail_set)] = llm_scores_full[:, list(tail_set)]


def mask_tr(scores):
    s = scores.copy(); s[train_matrix > 0] = -np.inf
    return s


def eval_reclist(name, sc_pop, sc_tail):
    sp = mask_tr(sc_pop); stt = mask_tr(sc_tail)
    rec = super_blueprint_merge(sp, stt, train_matrix, head_idx, N=10)
    m = full_rank_eval(sp, test, K=10, pop_prob=pop_prob,
                       reclist_matrix=rec, method='reclist',
                       head_idx=head_idx, train_matrix=train_matrix)
    print(f"{name}: " + str({k: round(v, 4) for k, v in m.items() if k != 'decile_props'}))
    return m


def train_pair(name, dp_eps, rounds=25, clients=256, lr=0.3):
    print(f"\n--- Training {name} (eps={dp_eps}) ---")
    t = time.time()
    m_pop = train_federated(sm_head, n_users, n_items, dim=64,
                            rounds=rounds, clients_per_round=clients, local_epochs=2,
                            lr=lr, dp_eps=dp_eps, max_grad_norm=1.0,
                            device=DEVICE, seed=0, verbose=True)
    sc_pop = m_pop.score_matrix()
    print(f"head trained in {time.time()-t:.1f}s")
    t = time.time()
    m_tail = train_federated(sm_tail, n_users, n_items, dim=64,
                             rounds=rounds, clients_per_round=clients, local_epochs=2,
                             lr=lr, dp_eps=dp_eps, max_grad_norm=1.0,
                             device=DEVICE, seed=0, verbose=True)
    print(f"tail trained in {time.time()-t:.1f}s")
    return sc_pop, m_tail.score_matrix()


results = {}

# 1) Baselines (no LLM)
sc_pop_noDP, sc_tail_noDP = train_pair("FedSUPER-noDP", dp_eps=None)
results["FedSUPER-noDP"] = eval_reclist("FedSUPER-noDP", sc_pop_noDP, sc_tail_noDP)

sc_pop_dp, sc_tail_dp = train_pair("FedSUPER-DP eps=2", dp_eps=2.0)
results["FedSUPER-DP_eps=2"] = eval_reclist("FedSUPER-DP_eps=2", sc_pop_dp, sc_tail_dp)

# 2) LLM augmentation sweep: blend LLM into the score matrix BEFORE blueprint merge.
# Augmentation strategy: For M_pop scores, blend with LLM similarity (only on head items).
# For M_tail scores, optionally blend (only tail items).
LAMBDAS = [0.3, 0.5, 0.7]
for lam in LAMBDAS:
    sc_pop_aug = (1.0 - lam) * sc_pop_noDP + lam * llm_pop   # blend LLM into head scores
    sc_tail_aug = (1.0 - lam) * sc_tail_noDP + lam * llm_tail_full   # similarly for tail
    results[f"LLM-FedSUPER-noDP_lam={lam}"] = eval_reclist(f"LLM-FedSUPER-noDP_lam={lam}", sc_pop_aug, sc_tail_aug)

for lam in LAMBDAS:
    sc_pop_aug = (1.0 - lam) * sc_pop_dp + lam * llm_pop
    sc_tail_aug = (1.0 - lam) * sc_tail_dp + lam * llm_tail_full
    results[f"LLM-FedSUPER-DP_eps=2_lam={lam}"] = eval_reclist(f"LLM-FedSUPER-DP_eps=2_lam={lam}", sc_pop_aug, sc_tail_aug)

# Save
with open(os.path.join(RESULT_DIR, "metrics_h3_v2.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n--- H3 v2 summary ---")
print(f"{'name':35s} {'Recall':>8s} {'nDCG':>8s} {'APLT':>8s} {'LTC':>8s} {'RMSE-PC':>8s} {'MRMC':>8s} {'GKPI':>8s}")
for name, m in results.items():
    print(f"{name:35s} {m['recall@K']:8.4f} {m['ndcg@K']:8.4f} {m['aplt']:8.4f} {m['ltc']:8.4f} {m['rmse_pc']:8.4f} {m['mrmc']:8.4f} {m['gkpi']:8.4f}")
