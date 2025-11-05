import pickle

import torch
from torch.utils.data import Dataset

import numpy as np


class Multi_view_data(Dataset):
    """
    load multi-view data
    """

    def __init__(self, view_number, idx, feature_list, labels):
        super(Multi_view_data, self).__init__()

        self.x = dict()
        for v_num in range(view_number):
            self.x[v_num] = feature_list[v_num][[idx], :].squeeze()
        self.y = labels[idx]

    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.x)):
            data[v_num] = (self.x[v_num][index]).astype(np.float32)
        target = self.y[index]

        return {
            'x': data,
            'y': target,
            'index': index
        }

    def __len__(self):
        return len(self.y)


def mixup_data(x_dict, y, alpha, mode, area_range):
    """
    Perform multi-view mixup/CutMix variants, per view, against a shared permutation.

    Returns:
      x_mix   : {view: mixed tensor, same shape as x_dict[view]}
      y_mix   : {view: paired labels (N,)}  — usually y[perm]
      lambdas : {view: (N,) effective mix coefficient ∈ [0,1]}
      pair_idx: {view: (N,) permutation indices}
      mask_dict: {view: mask used for mixing; None in 'scalar', otherwise
                  shape (N,D) for 2D or (N,1,H,W) broadcastable for 4D.
                  In 'mask-soft', values in {1 outside, λ inside}.
                  In 'mask-hard', values in {0,1}.}
    Notes:
      - scalar: classic elementwise convex combination with per-sample λ~Beta(α,α)
      - mask-hard: CutMix-like hard replace on a random region (binary mask)
      - mask-soft: soft replace inside region: x_mix = M*x_i + (1-M)*x_j, with
                   M = 1 outside; M = λ inside (λ~Beta(α,α) per sample)
      - Effective λ for (mask-soft/hard) is the spatial/featurewise mean of M.
    """
    assert mode in {"scalar", "mask-soft", "mask-hard"}
    device = next(iter(x_dict.values())).device
    N = y.size(0)

    x_mix, y_mix, lambdas, pair_idx, mask_dict = {}, {}, {}, {}, {}
    perm = torch.randperm(N, device=device)

    y_perm = y[perm].to(device)

    for v, x_v in x_dict.items():
        x_v = x_v.to(device)

        pair_idx[v] = perm

        if mode == "scalar":
            # Classic MixUp: xm = λ x_i + (1-λ) x_j  (broadcast λ across non-batch dims)
            lam = torch.distributions.Beta(alpha, alpha).sample((N,)).to(device)
            lam = lam if lam.ndim else lam.expand(N)
            lam_view = lam.view(-1, *([1] * (x_v.ndim - 1)))
            xm = lam_view * x_v + (1 - lam_view) * x_v[perm]
            x_mix[v] = xm
            y_mix[v] = y_perm
            lambdas[v] = lam
            mask_dict[v] = None

        else:
            # Region-based mixing: construct mask M and apply
            if x_v.dim() == 4:
                # (N,C,H,W)
                N_, C, H, W = x_v.shape
                assert N_ == N
                m_sel = torch.zeros((N, 1, H, W), device=device)
                a_min, a_max = area_range
                for n in range(N):
                    area = float(torch.empty(1).uniform_(a_min, a_max))
                    side = (area ** 0.5)
                    h = max(1, int(H * side))
                    w = max(1, int(W * side))
                    cy = torch.randint(0, max(H - h + 1, 1), (1,)).item()
                    cx = torch.randint(0, max(W - w + 1, 1), (1,)).item()
                    m_sel[n, 0, cy:cy + h, cx:cx + w] = 1.0
                m_sel = m_sel.expand(-1, C, -1, -1)
            elif x_v.dim() == 2:
                N_, D = x_v.shape
                assert N_ == N

                lam = torch.distributions.Beta(alpha, alpha).sample((N,)).to(device)
                lambdas[v] = lam  # 保持你的返回接口

                k = (lam * D).round().clamp(0, D).long()  # (N,)
                m_sel = torch.zeros((N, D), device=device)
                for n in range(N):
                    kn = k[n].item()
                    if kn > 0:
                        idx = torch.randperm(D, device=device)[:kn]
                        m_sel[n, idx] = 1.0
            else:
                raise ValueError(f"Unsupported tensor dim {x_v.dim()}; expect (N,D) or (N,C,H,W).")

            M = m_sel

            xm = M * x_v + (1.0 - M) * x_v[perm]

            x_mix[v] = xm
            y_mix[v] = y_perm
            mask_dict[v] = M

    return x_mix, y_mix, lambdas, pair_idx, mask_dict


def count_each_class_num(labels):
    '''
        Count the number of samples in each class
    '''
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict


def generate_partition(labels, ind, ratio=0.1):
    each_class_num = count_each_class_num(labels)
    labeled_each_class_num = {}  ## number of labeled samples for each class
    total_num = round(ratio * len(labels))
    for label in each_class_num.keys():
        labeled_each_class_num[label] = max(round(each_class_num[label] * ratio), 1)  # min is 1

    # index of labeled and unlabeled samples
    p_labeled = []
    p_unlabeled = []
    for idx, label in enumerate(labels):
        if (labeled_each_class_num[label] > 0):
            labeled_each_class_num[label] -= 1
            p_labeled.append(ind[idx])
            total_num -= 1
        else:
            p_unlabeled.append(ind[idx])
    return p_labeled, p_unlabeled
