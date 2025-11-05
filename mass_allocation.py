import torch.nn.functional as F
import torch


def alpha_schedule(epoch, T_warm):
    t = min(epoch, max(T_warm, 1))
    return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * t / max(T_warm, 1))))

def u_schedule(epoch, u_min=0.10, u_max=0.30, tau=10.0):
    return u_min + (u_max - u_min) * (1.0 - torch.exp(torch.tensor(-epoch / tau)))


def compute_view_reliability(
        per_view_repr_all,  # list length V; each tensor is (N_clean + N_mix, D)
        y_clean,  # tensor (N_clean,), integer class labels in [0, num_classes-1]
        num_classes: int,
        N_clean: int,
        metric: str = 'euclidean',  # 'euclidean' | 'cosine' | 'mahalanobis'
        T: float = 2.0,  # temperature for softmax on distances
        eps: float = 1e-6,  # numerical stability
):
    """
         Returns:
        z_conf: (V, N_mix)
            Per-view z-scored confidence deficiency, where confidence deficiency c = 1 - max_k p_hat_k.
        z_div:  (V, N_mix)
            Per-view z-scored JS divergence between a view's p_hat and the across-view mean p_bar.
        d_hat:  (V, N_mix)
            Sigmoid(z_div); a convenient [0,1] gate for unknown/open-set control.
    """
    device = per_view_repr_all[0].device
    V = len(per_view_repr_all)

    # Split per-view embeddings into CLEAN and MIX subsets (no grad needed for stats)
    H_clean_list = [H[:N_clean].detach() for H in per_view_repr_all]  # (N_clean, D)
    H_mix_list = [H[N_clean:].detach() for H in per_view_repr_all]  # (N_mix,  D)

    # -------- 1) Build per-view class prototypes (and inverse variances if needed) --------
    protos = []  # list of (K, D)
    inv_vars = []  # list of (K, D)
    for v in range(V):
        Hc = H_clean_list[v]
        mu_v = []
        iv_v = []
        for k in range(num_classes):
            idx = (y_clean == k)
            if idx.any():
                hk = Hc[idx]
                mu = hk.mean(dim=0)
                if metric == 'mahalanobis':
                    var = hk.var(dim=0, unbiased=False) + eps
                    iv = 1.0 / var
                else:
                    iv = None
            else:
                # No samples for class k in this mini-batch
                mu = Hc.mean(dim=0)   # view-wide mean prototype
                iv = None if metric != 'mahalanobis' else torch.ones(Hc.size(1), device=device)
            mu_v.append(mu)
            iv_v.append(iv if iv is not None else torch.tensor(0., device=device))  # 占位
        protos.append(torch.stack(mu_v, dim=0))  # (K, D)

        if metric == 'mahalanobis':
            iv_stack = []
            for iv in iv_v:
                # Replace placeholder scalars by ones(D) to keep shape consistent
                if iv.dim() == 0:
                    iv_stack.append(torch.ones(Hc.size(1), device=device))
                else:
                    iv_stack.append(iv)
            inv_vars.append(torch.stack(iv_stack, dim=0))  # (K, D)
        else:
            inv_vars.append(None)


    # -------- 2) Distances from MIX samples to prototypes -> per-class probabilities --------
    # For each view: dist in R^{N_mix x K}; p_hat = softmax(-dist / T)
    p_hat_list = []
    for v in range(V):
        Hm = H_mix_list[v]        # (N_mix, D)
        Mu = protos[v]            # (K, D)
        if metric == 'euclidean':
            # ||h - mu||_2^2
            # (N_mix, 1, D) - (1, K, D) -> (N_mix, K, D) -> sum_D
            dist = (Hm.unsqueeze(1) - Mu.unsqueeze(0)).pow(2).sum(dim=-1)  # (N_mix, K)
        elif metric == 'cosine':
            h_n = F.normalize(Hm, dim=-1)
            mu_n = F.normalize(Mu, dim=-1)
            # Convert cosine similarity to distance = 1 - cos
            dist = 1.0 - (h_n @ mu_n.t())  # (N_mix, K)
        elif metric == 'mahalanobis':
            # Diagonal Mahalanobis: sum_d ( (h_d - mu_d)^2 * inv_var_d )
            IV = inv_vars[v]  # (K, D)
            diff = Hm.unsqueeze(1) - Mu.unsqueeze(0)      # (N_mix, K, D)
            dist = (diff.pow(2) * IV.unsqueeze(0)).sum(dim=-1)  # (N_mix, K)
        else:
            raise ValueError("metric must be 'euclidean'|'cosine'|'mahalanobis'")

        p_hat = F.softmax(-dist / max(T, 1e-6), dim=1)    # (N_mix, K)
        p_hat_list.append(p_hat)

    # -------- 3) Confidence deficiency per view, then z-score across views  --------
    # confidence gap for a view = 1 - max_k p_hat_k  (larger = less confident)
    conf = torch.stack([1.0 - p.max(dim=1).values for p in p_hat_list], dim=0)  # (V, N_mix)

    # z-score across views for each sample
    conf_mu = conf.mean(dim=0, keepdim=True)
    conf_std = conf.std(dim=0, unbiased=False, keepdim=True) + eps
    z_conf = (conf - conf_mu) / conf_std                                       # (V, N_mix)

    # -------- 4) View disagreement via JS divergence to the across-view mean --------
    p_bar = torch.stack(p_hat_list, dim=0).mean(dim=0)  # (N_mix, K)
    delta = torch.stack([js_divergence(p, p_bar, eps=eps) for p in p_hat_list], dim=0)  # (V, N_mix)
    del_mu = delta.mean(dim=0, keepdim=True)
    del_std = delta.std(dim=0, unbiased=False, keepdim=True) + eps
    z_div = (delta - del_mu) / del_std                                        # (V, N_mix)

    d_hat = torch.sigmoid(z_div)                                              # (V, N_mix)
    return z_conf, z_div, d_hat


def js_divergence(p, q, eps=1e-6):
    """
    p, q: (N, K), 每行和为1
    返回: (N,) 的逐样本 JS
    """
    m = 0.5 * (p + q)
    kl_pm = (p * (p.add(eps).log() - m.add(eps).log())).sum(dim=1)
    kl_qm = (q * (q.add(eps).log() - m.add(eps).log())).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)

def view_adaptive_budget(u_t, u_min, u_max, z_conf, w_c=1.0, w_d=1.0, tau_g=1.0):
    to = lambda x: torch.as_tensor((x[0] if isinstance(x, (list, tuple)) else x),
                                   dtype=z_conf.dtype, device=z_conf.device)
    w_c_t, w_d_t, tau_t = to(w_c), to(w_d), to(tau_g).clamp_min(1e-12)
    u_min_t, u_max_t = to(u_min), to(u_max)
    u_t = to(u_t)
    gate = torch.sigmoid((w_c_t * z_conf ) / tau_t)
    u_v = u_min_t + (u_t - u_min_t) * gate
    return u_v