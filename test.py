import argparse
import warnings
import random
from mass_allocation import *
from utils import build_graphs, load_data
import numpy as np
import torch
from config import load_config
from evaluation_metrics import EvaluationMetrics, compute_all_scores
import copy
from label_utils import reassign_labels, special_train_test_split, generate_partition
from data import Multi_view_data, mixup_data
from models import PUME

np.set_printoptions(threshold=np.inf)
import torch.nn.functional as F


def build_soft_labels_view_avg(y, y_mix, lambdas, u_v, d_hat, alpha_t, num_classes, device, dtype):
    """
    y: (N,)
    y_mix: list length V, 每个元素 (N,)
    lambdas: dict/list length V, 每个元素 (N,)
    u_v, d_hat: (V, N)
    alpha_t: 标量/0-D tensor
    返回:
      y_tilde_avg: (N, K)
    """
    V, N = u_v.size(0), u_v.size(1)
    rho = (1.0 + (
        alpha_t if torch.is_tensor(alpha_t) else torch.tensor(alpha_t, device=device, dtype=dtype))) * 0.5  # 标量
    # (V, N)
    a_v = u_v * (1.0 - d_hat) * rho
    m_unknown = (u_v - a_v).clamp_min(0.0)

    y_tilde_sum = torch.zeros((N, num_classes), device=device, dtype=dtype)
    row = torch.arange(N, device=device)
    yi = y.view(-1).long()

    for v in range(V):
        lam = lambdas[v].to(device).to(dtype)  # (N,)
        yj = y_mix[v].view(-1).long()  # (N,)

        m_yi = lam * (1.0 - u_v[v])  # (N,)
        m_yj = (1.0 - lam) * (1.0 - u_v[v])  # (N,)

        y_tilde = torch.zeros((N, num_classes), device=device, dtype=dtype)
        y_tilde += (m_unknown[v] / num_classes).unsqueeze(1)  # 均匀分配未知质量到 K 类
        y_tilde[row, yi] += m_yi + 0.5 * a_v[v]
        y_tilde[row, yj] += m_yj + 0.5 * a_v[v]
        y_tilde = y_tilde / (y_tilde.sum(1, keepdim=True) + 1e-12)  # 数值归一

        y_tilde_sum += y_tilde

    y_tilde_avg = y_tilde_sum / V
    return y_tilde_avg


def main(args, device):
    def train(model, train_loader, valid_loader, device):
        model = model.to(device)

        optimizer = torch.optim.Adam([
            {'params': (p for n, p in model.named_parameters() if p.requires_grad and 'weight' in n),
             'weight_decay': 1e-2},
            {'params': (p for n, p in model.named_parameters() if p.requires_grad and 'weight' not in n)},
        ], lr=args.lr)
        step_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=23, gamma=0.1)
        epoch_best = -1
        best_valid_ccr = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        T_warm_ratio = 0.30

        w_c = 1.0
        w_d = 1.0

        for epoch in range(1, args.epoch + 1):
            model.train()
            train_loss, correct, num_samples = 0, 0, 0

            # ---- stage-adaptive factors for this epoch
            T_warm = int(T_warm_ratio * args.epoch) if args.epoch > 0 else 1
            alpha_t = alpha_schedule(epoch, T_warm)  # ∈ [0,1]
            u_t = u_schedule(epoch, u_min=args.u_min, u_max=args.u_max, tau=args.tau)  # ∈ [0,1]

            for batch in train_loader:
                x, y, index = batch['x'], batch['y'], batch['index']
                if args.beta != 0:
                    A_train = build_graphs(x, neighbor=args.neighbor)
                    for k in x.keys():
                        A_train[k] = A_train[k].to(device)
                else:
                    A_train = None

                for k in x.keys():
                    x[k] = x[k].to(device)
                y = y.long().to(device)

                # ====== mixup & forward ======
                x_mix, y_mix, lambdas, pair_idx, masks = mixup_data(
                    x, y, alpha=1.0, mode=args.mix_mode, area_range=(0.25, 0.6)
                )

                x_all = {v: torch.cat([x[v], x_mix[v]], dim=0) for v in range(len(x))}
                evidence, Zv = model(x_all, A_train)

                N = y.shape[0]
                num_views = len(x)
                dtype = evidence.dtype

                sup_loss = criterion(evidence[:N], y)  # clean CE

                # ====== (1)(2) View-dependent reliability ======
                per_view_repr_all = [Zv[v] for v in range(num_views)]
                z_conf, z_div, d_hat = compute_view_reliability(
                    per_view_repr_all=per_view_repr_all,
                    y_clean=y,  # 前 N 个为干净样本的标签
                    num_classes=num_classes,
                    N_clean=N,  # clean 样本数
                    metric='euclidean',  # 可选: 'euclidean' | 'cosine' | 'mahalanobis'
                    T=2.0,  # 距离转 softmax 的温度
                    eps=1e-6
                )

                # ====== (3) View-adaptive budget u_t^{(v)} ======
                u_v = view_adaptive_budget(
                    u_t=u_t.item() if torch.is_tensor(u_t) else float(u_t),
                    u_min=args.u_min, u_max=args.u_max,
                    z_conf=z_conf,
                    w_c=w_c, w_d=w_d
                )  # (V, N)

                # ====== (5)(6) Four-mass (DS) assignment (no h): ρ = (1 + α_t)/2; build soft labels and CE ======
                # Build view-averaged soft labels for the mixed samples using
                y_tilde_avg = build_soft_labels_view_avg(
                    y=y, y_mix=y_mix, lambdas=lambdas,
                    u_v=u_v, d_hat=d_hat, alpha_t=alpha_t,
                    num_classes=num_classes, device=device, dtype=dtype
                )
                # Compute CE only on the mixed samples (rows N: end).
                logp_mix = F.log_softmax(evidence[N:], dim=1)
                loss_Oen = -(y_tilde_avg * logp_mix).sum(dim=1).mean()

                # 总损失
                loss = sup_loss + args.alpha * loss_Oen

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                prob = F.softmax(evidence, dim=1)

                num_samples += len(y)
                correct += torch.sum(prob.argmax(dim=-1)[:y.shape[0]].eq(y)).item()

            resp, softmax_scores, valid_truth_label = valid(model, valid_loader, device)

            softmax_ccr, softmax_fpr, softmax_ccrs = EvaluationMetrics.ccr_at_fpr(np.array(valid_truth_label),
                                                                                  softmax_scores)

            valid_ccr = softmax_ccrs[-3]
            if valid_ccr:
                if best_valid_ccr <= valid_ccr:
                    best_valid_ccr = valid_ccr
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epoch_best = epoch

            if (epoch - epoch_best) > args.early_stop:
                break
            print(
                f'Epoch {epoch:3d}: train loss: {train_loss:.4f}')

        if epoch_best != -1:
            model.load_state_dict(best_model_wts)
        test_Z, softmax_scores, test_truth_label = valid(model, test_loader, device)

        test_truth_label = np.array(test_truth_label)
        compute_all_scores(args, test_truth_label, softmax_scores)
        return model

    def valid(model, loader, device):
        model.eval()
        prob = []
        resp = []
        label = []
        with torch.no_grad():
            for batch in loader:
                x, y = batch['x'], batch['y']
                A_train = build_graphs(x, neighbor=args.neighbor)
                #
                for k in x.keys():
                    x[k] = x[k].to(device)
                    A_train[k] = A_train[k].to(device)
                res, _ = model(x, A_train)
                _, Y_pre = torch.max(res, dim=1)
                prob.append(torch.softmax(res, 1).cpu().numpy())
                resp.append(res.cpu().numpy())
                label.append(y.cpu().numpy())
        label = np.concatenate(label)
        prob = np.concatenate(prob)
        resp = np.concatenate(resp)
        return resp, prob, label

    model = PUME(n_feats, n_view, num_classes, args.layer_num, args.thre, args.beta, device, args.hidden_dim)

    print('---------------------------- Experiment ------------------------------')
    print('Number of views:', len(train_data.x), ' views with dims:', [v.shape[-1] for v in train_data.x.values()])
    print('Number of training samples:', len(train_data))
    print('Number of validating samples:', len(valid_data))
    print('Trainable Parameters:')
    for n, p in model.named_parameters():
        print('%-40s' % n, '\t', p.data.shape)
    print('----------------------------------------------------------------------')
    train(model, train_loader, valid_loader, device)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='input batch size for training')
    parser.add_argument('--epoch', type=int, default=100, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--early_stop', type=int, default=50, metavar='N')

    parser.add_argument('--layer_num', type=int, default=10, metavar='N')
    parser.add_argument('--neighbor', type=int, default=10)

    parser.add_argument('--u_min', type=float, default=0.10)
    parser.add_argument('--u_max', type=float, default=0.30)
    parser.add_argument('--tau', type=float, default=10)

    parser.add_argument('--unseen_label_index', type=int, default=-100)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--thre', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--openness', type=float, default=0.1)

    parser.add_argument('--training_rate', type=float, default=0.1)
    parser.add_argument('--valid_rate', type=float, default=0.1)

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate')

    parser.add_argument("--config_path", type=str, default='layer.yaml')
    parser.add_argument("--save_file", type=str, default="res.txt")
    parser.add_argument("--data_path", type=str, default= "./data/")
    parser.add_argument('--save_results', default=True)
    parser.add_argument('--fix_seed', default=True)
    parser.add_argument('--mix_mode', type=str,default="mask-hard")
    args = parser.parse_args()

    args.device = '2'
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)

    dataset_dict = {1: 'esp_game', 2: 'Hdigit', 3: 'MITIndoor', 4: 'NoisyMNIST_15000',
                    5: 'NUSWIDE-OBJ', 6: 'scene15', 7: 'UCI', 8: "AwA", 9: 'VGGFace2'}
    select_dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    config = load_config(args.config_path)
    for ii in select_dataset:

        data = dataset_dict[ii]
        args.data = data
        args.layer_num = config[args.data]
        if data == 'AwA' or data == 'VGGFace2':
            args.batch_size = 512
            args.hidden_dim = 64
        else:
            args.batch_size = 256
            args.hidden_dim = 32

        features, labels = load_data(args.data, args.data_path)

        n_view = len(features)
        n_feats = [x.shape[1] for x in features]
        n = features[0].shape[0]
        n_classes = len(np.unique(labels))

        print(data, n, n_view, n_feats)
        open2 = (1 - args.openness) * (1 - args.openness)
        unseen_num = round((1 - open2 / (2 - open2)) * n_classes)
        args.unseen_num = unseen_num
        print("unseen_num:%d" % unseen_num)
        original_num_classes = len(np.unique(labels))
        seen_labels = list(range(original_num_classes - unseen_num))
        y_true = reassign_labels(labels, seen_labels, args.unseen_label_index)

        train_indices, test_valid_indices = special_train_test_split(y_true, args.unseen_label_index,
                                                                     test_size=1 - args.training_rate)
        valid_indices, test_indices = generate_partition(y_true[test_valid_indices], test_valid_indices,
                                                         args.valid_rate / (1 - args.training_rate))
        num_classes = np.max(y_true) + 1
        NCLASSES = num_classes

        print('data:{}\tseen_labels:{}\tunseen_num:{}\tnum_classes:{}'.format(
            data,
            seen_labels,
            unseen_num,
            num_classes))

        train_data = Multi_view_data(n_view, train_indices, features, y_true)
        valid_data = Multi_view_data(n_view, valid_indices, features, y_true)
        test_data = Multi_view_data(n_view, test_indices, features, y_true)

        train_loader = torch.utils.data.DataLoader(train_data
                                                   , batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_data
                                                   , batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data
                                                  , batch_size=args.batch_size, shuffle=False)
        labels = torch.from_numpy(labels).long().to(device)
        y_true = torch.from_numpy(y_true).to(device)
        train_indices = torch.LongTensor(train_indices).to(device)

        N_mini_batches = len(train_loader)
        print('The number of training images = %d' % N_mini_batches)
        args.num_classes = num_classes
        args.seen_labels = seen_labels

        if args.fix_seed:
            seed = 20
            torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
            torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            main(args, device)
        with open(args.save_file, "a") as f:
            f.write('\n')
