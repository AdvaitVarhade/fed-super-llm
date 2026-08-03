"""Sanity test the federated training harness with reduced scale."""
import os, sys, time
import numpy as np
import torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from recdata import load_ml1m, leave_one_out_split, build_train_matrix
from train import train_federated
from metrics import full_rank_eval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df, n_users, n_items, pop_prob, _, _ = load_ml1m(min_rating=4)
train_df, test = leave_one_out_split(df, n_users, n_items)
train_matrix = build_train_matrix(train_df, n_users, n_items)
train_pos = [set(np.where(train_matrix[u] > 0)[0].tolist()) for u in range(n_users)]

t = time.time()
m = train_federated(train_matrix, n_users, n_items, dim=64,
                    rounds=60, clients_per_round=512, local_epochs=2,
                    lr=0.3, dp_eps=None, max_grad_norm=1.0,
                    device=DEVICE, seed=0, verbose=True)
print(f"trained in {time.time()-t:.1f}s")
del m
import gc; gc.collect(); torch.cuda.empty_cache()
print("done")
