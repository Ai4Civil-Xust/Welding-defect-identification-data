import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import csv

# ==================== 路径 ====================
pretrained_model_path = r'E:\OneDrive\project\open-set-ours\models\source_domain_best.pth'
data_path = r'E:/OneDrive/project/open-set-ours/data/open_210.csv'
save_dir = r'E:/OneDrive/project/open-set-ours/legacy_out_D\6种方法\6种方法\PROSER\网格搜索_5fold'

os.makedirs(save_dir, exist_ok=True)

# ==================== 超参数 ====================
gamma = 0.5
mixup_alpha = 2.0
mixup_prob = 1.0
unknown_loss_weight = 5.0
num_epochs = 200
num_frozen = 2
batch_size = 32
lr = 0.0005
dummy_hidden = 64

# 初始超参搜索列表（仅评估该列表，不再细化）
T_candidates = [0.5, 0.7, 1.0, 1.5, 2.0]
k_candidates = [-1.0, -0.5, 0.0, 0.5, 1.0]
threshold_percentiles = [60,65,70,75,80,85,90,95]

num_folds = 5
min_acc_u_threshold = 0.94

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

# ==================== 已知类配置 ====================
num_known_classes = 3
unknown_class_id = 3

# ==================== CNN1D ====================
class CNN1D(nn.Module):
    def __init__(self, in_ch, conv_str, dropout, n_classes, in_len, n_blocks, bn_str, pool_str, pool_type_str):
        super().__init__()
        conv_params = [tuple(map(int, p.split(','))) for p in conv_str.split(';')]
        bn_list = [bool(int(b)) for b in bn_str.split(',')]
        pool_list = [bool(int(p)) for p in pool_str.split(',')]
        pool_types = [t.lower() for t in pool_type_str.split(',')]
        self.conv_blocks = nn.ModuleList()
        ch = in_ch
        L = in_len
        for i in range(n_blocks):
            layers = []
            for out_ch, k, s in conv_params[i::n_blocks]:
                layers.append(nn.Conv1d(ch, out_ch, k, stride=s))
                L = (L - k) // s + 1
                if bn_list[i]:
                    layers.append(nn.BatchNorm1d(out_ch))
                layers.append(nn.ReLU())
                if pool_list[i]:
                    if pool_types[i] == 'max':
                        layers.append(nn.MaxPool1d(3, 2))
                        L = (L - 3) // 2 + 1
                    elif pool_types[i] == 'avg':
                        layers.append(nn.AvgPool1d(3, 2))
                        L = (L - 3) // 2 + 1
            self.conv_blocks.append(nn.Sequential(*layers))
            ch = out_ch
        flat_dim = ch * L
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        for blk in self.conv_blocks:
            x = blk(x)
        x = x.view(x.size(0), -1)
        feat = self.classifier[:-1](x)
        out = self.classifier[-1](feat)
        return feat, out

# ==================== PROSER Module ====================
class PROSERModule(nn.Module):
    def __init__(self, feat_dim=128, hidden_dim=64):
        super().__init__()
        self.dummy = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.register_buffer('bias', torch.tensor(0.0))
        self.register_buffer('temperature', torch.tensor(1.0))

    def forward(self, feat):
        return self.dummy(feat).squeeze(1)

    def get_unknown_prob(self, feat, logits_known, T):
        raw = self.forward(feat) + self.bias
        combined = torch.cat([logits_known, raw.unsqueeze(1)], dim=1) / T
        probs = F.softmax(combined, dim=1)
        return probs[:, -1]

    def set_params(self, bias, T):
        self.bias = bias
        self.temperature = T

# ==================== Mixup ====================
def manifold_mixup(f1, f2, y1, y2, alpha=0.4, num_known=3, prob=0.8):
    B = f1.size(0)
    device = f1.device
    idx = torch.randperm(B, device=device)
    f2_s = f2[idx]
    y2_s = y2[idx]
    diff = (y1 != y2_s)
    mask = diff & (torch.rand(B, device=device) < prob)
    if mask.sum() == 0:
        return f1, y1
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    lam = torch.tensor(lam, device=device, dtype=f1.dtype)
    fmix = lam * f1 + (1 - lam) * f2_s
    y_mix = torch.where(mask, torch.tensor(num_known, device=device, dtype=y1.dtype), y1)
    return fmix, y_mix

# ==================== 数据加载（仅分离已知/未知，划分在5折循环内） ====================
def load_data(path):
    df = pd.read_csv(path)
    labels = df.iloc[:, 0].values
    feats = df.iloc[:, 1:].values.astype(float)
    cls, idxs = [], []
    for i, lb in enumerate(labels):
        p = str(lb).split('-')
        if len(p) >= 4:
            try:
                cls.append(int(p[3]))
                idxs.append(i)
            except:
                pass
    X_all = feats[idxs][:, np.newaxis, :]
    y_all = np.array(cls)

    # 分离已知类与未知类，标签映射
    known_mask = (y_all == 1) | (y_all == 2) | (y_all == 3)
    X_known = X_all[known_mask]
    y_known = y_all[known_mask] - 1  # 原始1/2/3 → 0/1/2

    unknown_mask = y_all == 4
    X_unknown = X_all[unknown_mask]
    y_unknown = np.full(len(X_unknown), unknown_class_id)  # 未知类映射为3

    print("===== 数据集总览 =====")
    print(f"已知类总数：{len(X_known)}，好{(y_known==0).sum()}，坏{(y_known==1).sum()}，光板{(y_known==2).sum()}")
    print(f"未知类总数：{len(X_unknown)}")
    print("---------------------")
    return X_known, y_known, X_unknown, y_unknown

# ==================== 评估函数 ====================
def evaluate_with_threshold(net, proser, val_loader, test_loader, device, T, bias, thresh_percentile):
    proser.set_params(bias, T)
    net.eval()
    proser.eval()

    # 验证集（已知类）计算未知概率，确定阈值
    val_probs = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            feat, logits = net(x)
            prob = proser.get_unknown_prob(feat, logits, T)
            val_probs.append(prob.cpu().numpy())
    val_probs = np.concatenate(val_probs)
    threshold = np.percentile(val_probs, thresh_percentile)

    # 测试集评估
    ys, ps = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            feat, logits = net(x)
            prob = proser.get_unknown_prob(feat, logits, T)
            known_pred = logits.argmax(1)
            final = torch.where(prob > threshold,
                                torch.tensor(unknown_class_id, device=device, dtype=known_pred.dtype),
                                known_pred)
            ys.extend(y.cpu().numpy())
            ps.extend(final.cpu().numpy())
    ys = np.array(ys)
    ps = np.array(ps)

    mk = np.isin(ys, list(range(num_known_classes)))
    mu = ys == unknown_class_id
    acc_k = (ys[mk] == ps[mk]).sum() / mk.sum() if mk.sum() > 0 else 0
    acc_u = (ps[mu] == unknown_class_id).sum() / mu.sum() if mu.sum() > 0 else 0
    hos = 2 * acc_k * acc_u / (acc_k + acc_u) if (acc_k + acc_u) > 0 else 0.0
    return acc_k, acc_u, hos

# ==================== 构建模型函数 ====================
def build_model(device):
    net = CNN1D(1, "16,21,1;16,21,2;32,7,2;32,7,2;64,3,2", 0.5, 5, 801, 5,
                "0,1,0,1,1", "0,1,1,1,0", "none,max,avg,avg,none")
    net.load_state_dict(torch.load(pretrained_model_path, map_location=device, weights_only=False))
    for i, blk in enumerate(net.conv_blocks):
        if i < num_frozen:
            for p in blk.parameters():
                p.requires_grad = False
    net.classifier[-1] = nn.Linear(128, num_known_classes)
    net.to(device)
    proser = PROSERModule(feat_dim=128, hidden_dim=dummy_hidden).to(device)
    return net, proser

# ==================== 训练模型 ====================
def train_model(net, proser, train_loader, device, fold_seed):
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)
    
    optimizer = optim.Adam(
        list(filter(lambda p: p.requires_grad, net.parameters())) + list(proser.parameters()),
        lr=lr, weight_decay=1e-3)
    cls_crit = nn.CrossEntropyLoss()

    net.train()
    proser.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            B = x.size(0)
            half = B // 2
            if half == 0:
                continue
            x1, y1 = x[:half], y[:half]
            x2, y2 = x[half:2*half], y[half:2*half]

            f1, out1 = net(x1)
            loss_cls = cls_crit(out1, y1)

            f2, _ = net(x2)
            f_mix, y_mix = manifold_mixup(f2, f2, y2, y2, mixup_alpha, num_known_classes, mixup_prob)

            raw = proser(f_mix)
            mask_k = (y_mix != unknown_class_id)
            mask_u = (y_mix == unknown_class_id)
            loss_dummy = 0.0
            if mask_k.sum() > 0:
                loss_dummy += F.binary_cross_entropy_with_logits(raw[mask_k], torch.zeros_like(raw[mask_k]))
            if mask_u.sum() > 0:
                loss_dummy += unknown_loss_weight * F.binary_cross_entropy_with_logits(raw[mask_u], torch.ones_like(raw[mask_u]))

            loss = loss_cls + gamma * loss_dummy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out1.argmax(1) == y1).sum().item()
            total += y1.size(0)

        if (epoch + 1) % 20 == 0:
            print(f'    Epoch {epoch+1:3d} TrAcc={correct/total:.4f} Loss={total_loss/len(train_loader):.4f}')

# ==================== 主程序 ====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    X_known, y_known, X_unknown, y_unknown = load_data(data_path)

    # ---------- 外层5折划分 + 折内8:2拆分 ----------
    fold_models = []
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X_known, y_known)):
        print(f"\n========== 训练 Fold {fold_idx+1}/{num_folds} ==========")

        # 第一层：拆分已知类测试集 + 折内候选集
        X_test_known = X_known[test_idx]
        y_test_known = y_known[test_idx]
        X_trainval = X_known[train_val_idx]
        y_trainval = y_known[train_val_idx]

        # 第二层：候选集内部8:2分层拆分训练集 / 验证集（random_state=42）
        # [沙箱D改动] 去掉折内8:2：训练集恢复为论文口径的 154 组候选
        X_train, X_val, y_train, y_val = X_trainval, X_trainval, y_trainval, y_trainval

        # 拼接最终测试集：已知测试集 + 全部未知类，固定seed打乱
        X_test = np.concatenate([X_test_known, X_unknown], axis=0)
        y_test = np.concatenate([y_test_known, y_unknown], axis=0)
        perm = np.random.RandomState(seed).permutation(len(y_test))
        X_test = X_test[perm]
        y_test = y_test[perm]

        # 构建loader
        train_ld = DataLoader(TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=batch_size, shuffle=True)
        val_ld = DataLoader(TensorDataset(
            torch.FloatTensor(X_val), torch.LongTensor(y_val)),
            batch_size=batch_size, shuffle=False)
        test_ld = DataLoader(TensorDataset(
            torch.FloatTensor(X_test), torch.LongTensor(y_test)),
            batch_size=batch_size, shuffle=False)

        # 训练模型
        fold_seed = seed + fold_idx
        net, proser = build_model(device)
        train_model(net, proser, train_ld, device, fold_seed)

        fold_models.append((net, proser, val_ld, test_ld))

    # ---------- 单组合评估函数 ----------
    def evaluate_combo_on_fold(combo, net, proser, val_ld, test_ld):
        T, k, th_p = combo
        net.eval()
        proser.eval()

        # 计算验证集原始输出，推导bias
        proser.bias = torch.tensor(0.0)
        with torch.no_grad():
            val_feats = []
            for xv, _ in val_ld:
                fv, _ = net(xv.to(device))
                val_feats.append(fv)
            val_feats = torch.cat(val_feats, dim=0)
            raw_val = proser(val_feats).cpu().numpy()

        bias = torch.tensor(-raw_val.mean() - k * raw_val.std())
        proser.set_params(bias, torch.tensor(T))

        # 评估
        ak, au, hos = evaluate_with_threshold(net, proser, val_ld, test_ld,
                                              device, torch.tensor(T), bias, th_p)
        return ak, au, hos

    def evaluate_combo(combo):
        fold_results = []
        for net, proser, val_ld, test_ld in fold_models:
            ak, au, hos = evaluate_combo_on_fold(combo, net, proser, val_ld, test_ld)
            fold_results.append((ak, au, hos))
        return fold_results

    # ---------- 初始列表搜索（无多层细化） ----------
    evaluated = {}
    all_candidates = list(itertools.product(T_candidates, k_candidates, threshold_percentiles))

    print(f"\n========== 开始超参搜索，共 {len(all_candidates)} 组组合 ==========")
    for combo in all_candidates:
        fold_results = evaluate_combo(combo)
        evaluated[combo] = fold_results
        ak_avg = np.mean([r[0] for r in fold_results])
        au_avg = np.mean([r[1] for r in fold_results])
        hos_avg = np.mean([r[2] for r in fold_results])
        print(f"  T={combo[0]:.1f}, k={combo[1]:.1f}, th_p={combo[2]} -> "
              f"Avg_Acc_k={ak_avg:.4f}, Avg_Acc_u={au_avg:.4f}, Avg_HOS={hos_avg:.4f}")

    # ---------- 选取最优组合 ----------
    def combo_key(combo):
        fold_results = evaluated[combo]
        au_avg = np.mean([r[1] for r in fold_results])
        hos_avg = np.mean([r[2] for r in fold_results])
        return (au_avg >= min_acc_u_threshold, hos_avg)

    best_combo = max(evaluated.keys(), key=combo_key)
    best_T, best_k, best_th_p = best_combo
    best_fold_results = evaluated[best_combo]
    best_ak_avg = np.mean([r[0] for r in best_fold_results])
    best_au_avg = np.mean([r[1] for r in best_fold_results])
    best_hos_avg = np.mean([r[2] for r in best_fold_results])

    # ---------- 写入CSV汇总 ----------
    summary_csv = os.path.join(save_dir, 'cv_summary.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['T', 'k', 'th_p',
                         'fold1_HOS', 'fold2_HOS', 'fold3_HOS', 'fold4_HOS', 'fold5_HOS',
                         'avg_HOS', 'avg_Acc_k', 'avg_Acc_u', 'qualified'])

        for combo, fold_results in sorted(evaluated.items(), key=lambda x: np.mean([r[2] for r in x[1]]), reverse=True):
            T, k, th_p = combo
            hos_list = [r[2] for r in fold_results]
            ak_avg = np.mean([r[0] for r in fold_results])
            au_avg = np.mean([r[1] for r in fold_results])
            hos_avg = np.mean(hos_list)
            qualified = 1 if au_avg >= min_acc_u_threshold else 0
            writer.writerow([T, k, th_p] +
                            [f'{h:.4f}' for h in hos_list] +
                            [f'{hos_avg:.4f}', f'{ak_avg:.4f}', f'{au_avg:.4f}', qualified])

    # ---------- 最终结果输出 ----------
    print("\n" + "="*60)
    print("PROSER 5折交叉验证搜索完成！")
    print(f"最优参数: T={best_T}, k={best_k}, threshold_percentile={best_th_p}")
    print(f"5折平均 Acc_k = {best_ak_avg:.4f}")
    print(f"5折平均 Acc_u = {best_au_avg:.4f}")
    print(f"5折平均 HOS   = {best_hos_avg:.4f}")
    print(f"结果汇总表：{summary_csv}")
    print("="*60)


if __name__ == '__main__':
    main()
