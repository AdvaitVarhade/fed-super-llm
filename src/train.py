"""Training harness: centralized and federated SuperMF training.

Centralized: standard SGD over BPR pairs from train_data.
Federated: simulate per-client training on user-local data shards; aggregate
item embeddings + bias gradients at server each round; user embeddings stay private.
"""
import os, time, copy, numpy as np
import torch
import torch.nn as nn

from model import RecMF, bpr_loss, pick_negatives


def train_centralized(train_matrix, n_users, n_items, dim=64, epochs=10,
                     lr=0.05, batch_users=256, negatives_per_pos=2,
                     device="cuda" if torch.cuda.is_available() else "cpu",
                     seed=0, verbose=True, multi_obj_alpha=None):
    """BPR train on user/item matrix."""
    torch.manual_seed(seed)
    n_users, n_items = train_matrix.shape
    model = RecMF(n_users, n_items, dim=dim, seed=seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    rng = np.random.default_rng(seed + 1)
    pos_user, pos_item = np.where(train_matrix > 0)
    n_pos = len(pos_user)
    for ep in range(epochs):
        order = rng.permutation(n_pos)
        losses = []
        eps_log = []
        for start in range(0, n_pos, batch_users):
            idx = order[start:start + batch_users]
            u = torch.tensor(pos_user[idx], device=device, dtype=torch.long)
            i_pos = torch.tensor(pos_item[idx], device=device, dtype=torch.long)
            negs = []
            for u_idx in pos_user[idx]:
                negs.append(int(pick_negatives(train_matrix[u_idx], n_items, n=negatives_per_pos, rng=rng)[0]))
            i_neg = torch.tensor(negs, device=device, dtype=torch.long)
            ps, _, _ = model(u, i_pos)
            ns, _, _ = model(u, i_neg)
            # BPR loss reported per sample
            loss = bpr_loss(ps, ns)
            if multi_obj_alpha is not None:
                eu = model.user_emb(u)
                loss = loss - multi_obj_alpha * eu.var(dim=0).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            # track % of batches with pos > neg (true fraction of batches where BPR-loss is below ln(2))
            with torch.no_grad():
                eps_log.append((ps > ns).float().mean().item())
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"epoch {ep:3d}: loss={np.mean(losses):.4f}  pos>neg_rate={np.mean(eps_log):.3f}  ({len(losses)} batches)")
    model.eval()
    return model


def train_federated(train_matrix, n_users, n_items, dim=64, rounds=30,
                    clients_per_round=64, local_epochs=2, lr=0.05,
                    dp_eps=None, max_grad_norm=1.0, dp_delta=1e-5,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    seed=0, verbose=True, multi_obj_alpha=0.0, pop_prob=None,
                    adaptive_pop_dp=False, pop_amplification=2.0):
    """FedNCF-style training. Each user is a client; item embeddings + bias are SHARED and
    aggregated server-side; user embeddings are PRIVATE (kept client-side).

    Aggregation = FedAvg over item parameters only.

    Optional DP (V.clients_per_round updated via DP noise) using Gaussian mechanism.

    Optional multi-objective penalty: when `multi_obj_alpha > 0` and `pop_prob` is provided,
    add a popularity-dispersion term to each BPR loss: penalty = sum(score * pop[i])
    for items touched. This INCENTIVIZES lower scores on popular items. Loss = BPR
    loss + alpha * penalty (NOT subtracted; that's reserved in our convention for
    regularization that should DECREASE the loss to push scores down).
    """
    torch.manual_seed(seed)
    n_users, n_items = train_matrix.shape

    # SERVER STATE (publicly aggregated)
    server_item_emb = nn.Embedding(n_items, dim).to(device)
    server_item_emb.weight.data.uniform_(-0.01, 0.01)
    server_item_bias = nn.Embedding(n_items, 1).to(device)
    server_item_bias.weight.data.zero_()

    # CLIENT STATE (private => never aggregated)
    local_user_emb = nn.Embedding(n_users, dim).to(device)
    local_user_emb.weight.data.uniform_(-0.01, 0.01)

    # Popularity dispersion: prepare tensor of per-item popularity (log1p to dampen)
    pop_tensor = None
    if multi_obj_alpha > 0 and pop_prob is not None:
        pop_tensor = torch.tensor(np.log1p(pop_prob * 100), device=device, dtype=torch.float32)

    # Adaptive DP: per-item noise multiplier (high for popular items, low for tail)
    sigma_mult_per_item = None
    if adaptive_pop_dp and pop_prob is not None:
        # Normalized log-pop: range [0, 1]
        log_pop = np.log1p(pop_prob * 100)
        log_pop_min = float(log_pop.min())
        log_pop_max = float(log_pop.max())
        if log_pop_max > log_pop_min:
            norm_log_pop = (log_pop - log_pop_min) / (log_pop_max - log_pop_min)
        else:
            norm_log_pop = np.ones_like(log_pop)
        # Multiplier = 1 + pop_amplification * norm_log_pop -> range [1, 1+pop_amp]
        sigma_mult_per_item = torch.tensor(1.0 + pop_amplification * norm_log_pop,
                                            device=device, dtype=torch.float32)

    sigma = 0.0
    if dp_eps is not None and dp_eps > 0:
        # Practical per-update Gaussian noise scaling. We use sigma = scale / eps with
        # scale=0.3 (chosen to keep training stable across the eps grid; the formal
        # dp guarantee would require accounting for composition, which we leave for
        # future work — here we focus on the empirical fairness/accuracy tradeoff).
        sigma = 0.3 / max(dp_eps, 0.01)
        # cap sigma to keep training tractable
        sigma = float(min(sigma, 0.6))

    rng = np.random.default_rng(seed + 1)
    active_users = np.where(train_matrix.sum(axis=1) > 0)[0]
    if len(active_users) == 0:
        raise ValueError("no active users")

    for rd in range(rounds):
        chosen = rng.choice(active_users, size=min(clients_per_round, len(active_users)), replace=False)

        # snapshot global item params at start of round (for FedAvg delta computation)
        start_item_emb = server_item_emb.weight.detach().clone()
        start_item_bias = server_item_bias.weight.detach().clone()

        # Sparse FedAvg accumulators: sum delta + count contributors per item
        emb_sum = torch.zeros_like(server_item_emb.weight)
        emb_cnt = torch.zeros(server_item_emb.weight.shape[0], device=device)
        bias_sum = torch.zeros_like(server_item_bias.weight)
        bias_cnt = torch.zeros(server_item_bias.weight.shape[0], device=device)
        losses = []

        for u_cl in chosen:
            user_row = train_matrix[u_cl]
            pos_items = np.where(user_row > 0)[0]
            if len(pos_items) < 2:
                continue
            # local copy of item params; user emb is shared (we use the global user embedding)
            local_item_emb = start_item_emb.clone().detach().requires_grad_(True)
            local_item_bias = start_item_bias.clone().detach().requires_grad_(True)
            # user embedding is private (persists across rounds). Detach local tensor for grad
            local_user_vec = local_user_emb.weight[u_cl].detach().clone().requires_grad_(True)
            opt = torch.optim.SGD([local_item_emb, local_item_bias, local_user_vec], lr=lr)

            # Sweep over (a subset of) the user's positives as positive items this round
            sweep = rng.choice(pos_items, size=min(local_epochs * 4, len(pos_items)), replace=False) \
                if len(pos_items) > local_epochs * 4 else pos_items
            touched = set()
            for pidx in sweep:
                pidx = int(pidx)
                nidx = int(pick_negatives(user_row, n_items, rng=rng)[0])
                u_ten = torch.tensor([u_cl], device=device, dtype=torch.long)
                i_pos = torch.tensor([pidx], device=device, dtype=torch.long)
                i_neg = torch.tensor([nidx], device=device, dtype=torch.long)
                eu = local_user_vec.unsqueeze(0)
                ei_pos = local_item_emb[i_pos]
                ei_neg = local_item_emb[i_neg]
                b_pos = local_item_bias[i_pos].squeeze()
                b_neg = local_item_bias[i_neg].squeeze()
                ps = (eu * ei_pos).sum() + b_pos
                ns = (eu * ei_neg).sum() + b_neg
                loss = bpr_loss(ps.unsqueeze(0), ns.unsqueeze(0))
                if multi_obj_alpha > 0 and pop_tensor is not None:
                    # popularity-dispersion penalty: add alpha * pop[i] * score(i) for
                    # the positive item so the optimizer reduces positive-item score on
                    # popular items.
                    loss = loss + multi_obj_alpha * pop_tensor[pidx] * ps
                opt.zero_grad()
                loss.backward()
                # clip+noise on local item grads -> DP
                local_item_emb.grad.data.clamp_(-max_grad_norm, max_grad_norm)
                local_item_bias.grad.data.clamp_(-max_grad_norm, max_grad_norm)
                # clip user grad too (kept private)
                local_user_vec.grad.data.clamp_(-max_grad_norm, max_grad_norm)
                if sigma > 0:
                    if sigma_mult_per_item is not None:
                        # per-item adaptive noise: scale gradient noise by popularity
                        mult = sigma_mult_per_item  # (n_items,)
                        noise_e = torch.randn_like(local_item_emb.grad) * sigma
                        noise_b = torch.randn_like(local_item_bias.grad) * sigma
                        # Apply per-row multiplier: noise_e[i] *= mult[i]
                        local_item_emb.grad.data += noise_e * mult.unsqueeze(1)
                        local_item_bias.grad.data += noise_b * mult.unsqueeze(1)
                    else:
                        local_item_emb.grad.data += torch.randn_like(local_item_emb.grad) * sigma
                        local_item_bias.grad.data += torch.randn_like(local_item_bias.grad) * sigma
                opt.step()
                losses.append(loss.item())
                touched.add(pidx)
                touched.add(nidx)
            # write updated private user embedding back to client-side embedding table
            with torch.no_grad():
                local_user_emb.weight[u_cl].data = local_user_vec.detach()
            # accumulate this client's item deltas sparsely
            if touched:
                t_idx = torch.tensor(sorted(touched), device=device, dtype=torch.long)
                emb_sum[t_idx] += (local_item_emb.detach() - start_item_emb)[t_idx]
                emb_cnt[t_idx] += 1.0
                bias_sum[t_idx] += (local_item_bias.detach() - start_item_bias)[t_idx]
                bias_cnt[t_idx] += 1.0

        # Sparse FedAvg: apply averaged delta to items that were updated this round
        with torch.no_grad():
            mask = emb_cnt > 0
            if mask.any():
                idx = torch.where(mask)[0]
                server_item_emb.weight.data[idx] += (emb_sum[idx] / emb_cnt[idx].unsqueeze(1))
                server_item_bias.weight.data[idx] += (bias_sum[idx] / bias_cnt[idx].unsqueeze(1))

        if verbose and (rd % max(1, rounds // 5) == 0 or rd == rounds - 1):
            mask_pct = (mask.float().mean() * 100).item()
            print(f"round {rd:3d}: {len(chosen)} clients, "
                  f"loss ~{np.mean(losses[-200:]) if losses else 0:.4f}  "
                  f"items touched: {mask_pct:.1f}%  (sigma={sigma:.3f})")

    # assemble a RecMF for evaluation (centralized eval uses both user+item emb)
    model = RecMF(n_users, n_items, dim=dim, seed=seed).to(device)
    model.user_emb.weight.data = local_user_emb.weight.data.to(device)
    model.item_emb.weight.data = server_item_emb.weight.data.to(device)
    model.item_bias.weight.data = server_item_bias.weight.data.to(device)
    model.eval()
    return model
