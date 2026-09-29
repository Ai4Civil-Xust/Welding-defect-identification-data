import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset  # 补充缺失的导入
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# ===================== 路径配置 =====================
pretrained_model_path = r'E:\OneDrive\project\open-set-ours\models\source_domain_best.pth'
data_path = r'E:/OneDrive/project/open-set-ours/data/open_210.csv'
results_root_dir = r'E:/OneDrive/project/open-set-ours/legacy_out_D\6种方法\6种方法\CAC Loss\deepseek-网格'
grid_search_dir = os.path.join(results_root_dir, "grid_search_5fold")
os.makedirs(grid_search_dir, exist_ok=True)

# ==========【网格搜索参数】==========
lambda_search_list = [0.005, 0.008, 0.012, 0.0167, 0.02, 0.025, 0.03, 0.04, 0.05]

min_acc_u_threshold = 0.94
num_frozen_blocks = 2
num_epochs = 200
cac_margin = 1.0
batch_size = 64
lr = 0.0005
num_folds = 5  # 5折交叉验证
# =================================================

summary_csv_path = os.path.join(grid_search_dir, "grid_search_summary_5fold.csv")
if not os.path.exists(summary_csv_path):
    with open(summary_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "lambda_cac",
            "fold1_HOS", "fold2_HOS", "fold3_HOS", "fold4_HOS", "fold5_HOS",
            "avg_HOS", "avg_Acc_k", "avg_Acc_u", "qualified"
        ])

# 固定随机种子
def fix_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_seed(42)

# ==================== CNN1D ====================
class CNN1D(nn.Module):
    def __init__(self, input_channels, conv_params, dropout_rate, num_classes, input_length, num_conv_blocks,
                 use_bn_list, use_pooling_list, pooling_type_list):
        super(CNN1D, self).__init__()

        if isinstance(conv_params, str):
            conv_params = [tuple(map(int, p.split(','))) for p in conv_params.split(';')]

        use_bn_list = [bool(int(b)) for b in use_bn_list.split(',')]
        use_pooling_list = [bool(int(p)) for p in use_pooling_list.split(',')]
        pooling_type_list = [ptype.lower() for ptype in pooling_type_list.split(',')]

        self.conv_blocks = nn.ModuleList()
        in_channels = input_channels
        cur_len = input_length

        for block_idx in range(num_conv_blocks):
            out_channels, kernel_size, stride = conv_params[block_idx]
            block_layers = []
            block_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride))
            cur_len = (cur_len - kernel_size) // stride + 1

            if use_bn_list[block_idx]:
                block_layers.append(nn.BatchNorm1d(out_channels))

            block_layers.append(nn.ReLU())

            if use_pooling_list[block_idx] and cur_len >= 3:
                pool_type = pooling_type_list[block_idx]
                if pool_type == 'max':
                    block_layers.append(nn.MaxPool1d(kernel_size=3, stride=2))
                    cur_len = (cur_len - 3) // 2 + 1
                elif pool_type == 'avg':
                    block_layers.append(nn.AvgPool1d(kernel_size=3, stride=2))
                    cur_len = (cur_len - 3) // 2 + 1

            self.conv_blocks.append(nn.Sequential(*block_layers))
            in_channels = out_channels

        self.classifier = None
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        feat = torch.flatten(x, 1)
        if self.classifier is None:
            feat_dim = feat.size(1)
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(128, self.num_classes)
            ).to(feat.device)
        embed = self.classifier[:4](feat)
        out = self.classifier(feat)
        return embed, out

    def reset_parameters(self):
        for conv_block in self.conv_blocks:
            for layer in conv_block:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        if self.classifier is not None:
            for layer in self.classifier:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

# ==================== 全局超参 ====================
input_channels = 1
conv_params = "16,21,1;16,21,1;32,7,1;32,7,1;64,3,1"
dropout_rate = 0.5
pretrained_num_classes = 5
input_length = 801
num_conv_blocks = 5
use_bn_list = "0,1,0,1,1"
use_pooling_list = "0,0,0,0,0"
pooling_type_list = "max,max,max,max,max"
new_num_classes = 4

# ==================== 数据加载（分离已知/未知，5折拆分在循环内完成） ====================
def load_data(data_path):
    data = pd.read_csv(data_path)
    assert data.shape[1] == input_length + 1, \
        f"数据列数应为 {input_length + 1}，实际为 {data.shape[1]}"
    labels = data.iloc[:, 0].values
    features = data.iloc[:, 1:].values.astype(float)
    label_classes = []
    valid_idx = []
    for i, label in enumerate(labels):
        parts = str(label).split('-')
        if len(parts) >= 4:
            try:
                lab = int(parts[3])
                label_classes.append(lab)
                valid_idx.append(i)
            except:
                pass
    X_all = features[valid_idx][:, np.newaxis, :]
    y_all = np.array(label_classes)

    # 分离已知类和未知类，并完成标签映射
    known_mask = (y_all == 1) | (y_all == 2) | (y_all == 3)
    X_known = X_all[known_mask]
    y_known = y_all[known_mask] - 1  # 原始1/2/3 → 0/1/2

    unknown_mask = y_all == 4
    X_unknown = X_all[unknown_mask]
    y_unknown = np.full(len(X_unknown), 3)  # 未知类映射为3

    print("===== 数据集总览 =====")
    print(f"已知类总数：{len(X_known)}，好{(y_known==0).sum()}，坏{(y_known==1).sum()}，光板{(y_known==2).sum()}")
    print(f"未知类总数：{len(X_unknown)}")
    return X_known, y_known, X_unknown, y_unknown

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# CAC Loss
class CACLoss(nn.Module):
    def __init__(self, num_classes=3, feat_dim=128, margin=1.0, use_gpu=None):
        super(CACLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.margin = margin
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        self.use_gpu = use_gpu
        self.anchors = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    def forward(self, x, labels):
        anchors_batch = self.anchors[labels]
        attract_loss = torch.sum((x - anchors_batch) ** 2, dim=1).mean()
        anchor_dist = torch.cdist(self.anchors, self.anchors, p=2)
        mask = 1.0 - torch.eye(self.num_classes, device=x.device)
        repel_term = torch.clamp(self.margin - anchor_dist, min=0)
        repel_loss = (repel_term * mask).sum() / max(self.num_classes * (self.num_classes - 1), 1)
        return attract_loss + repel_loss

# 获取特征中心与阈值（训练集上计算，保持CAC原生逻辑）
def get_centers_and_thresholds(model, loader, device):
    model.eval()
    feat_collect = {0:[],1:[],2:[]}
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            feat, _ = model(data)
            fn = feat.cpu().numpy()
            ln = labels.numpy()
            for f, l in zip(fn, ln):
                feat_collect[l].append(f)
    center = {}
    tau = {}
    for c in [0,1,2]:
        arr = np.array(feat_collect[c])
        center[c] = np.mean(arr, axis=0)
        dists = np.sqrt(np.sum((arr - center[c])**2, axis=1))
        tau[c] = np.max(dists)
    return center, tau

# 开放集评估
def openset_evaluate(model, loader, center, tau):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for x, lab in loader:
            x = x.to(device)
            feat, _ = model(x)
            fn = feat.cpu().numpy()
            ln = lab.numpy()
            for ft, gt in zip(fn, ln):
                d0 = np.sqrt(np.sum((ft-center[0])**2))
                d1 = np.sqrt(np.sum((ft-center[1])**2))
                d2 = np.sqrt(np.sum((ft-center[2])**2))
                dists = {0:d0, 1:d1, 2:d2}
                md = min(dists, key=dists.get)
                pred = md if dists[md] < tau[md] else 3
                yt.append(gt)
                yp.append(pred)
    yt = np.array(yt)
    yp = np.array(yp)
    mask_k = np.isin(yt, [0,1,2])
    mask_u = (yt == 3)
    ak = np.sum(yt[mask_k]==yp[mask_k]) / np.sum(mask_k) if np.sum(mask_k)>0 else 0
    au = np.sum(yp[mask_u]==3) / np.sum(mask_u) if np.sum(mask_u)>0 else 0
    hos = 2*ak*au/(ak+au) if (ak+au)!=0 else 0
    return ak, au, hos

# 绘制训练曲线
def plot_curve(train_loss, train_acc, save_dir):
    plt.figure()
    plt.plot(train_loss, label="Train Loss", color='#1f77b4', linewidth=1.5)
    plt.title("Training Loss (CAC)", fontsize=13)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(train_acc, label="Train Accuracy", color='#1f77b4', linewidth=1.5)
    plt.title("Training Accuracy (CAC)", fontsize=13)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=300)
    plt.close()

# 加载预训练权重
def load_pretrained_weights(model, pretrained_path, device):
    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    if isinstance(ckpt, dict) and all(k.startswith('module.') for k in ckpt.keys()):
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    state = model.state_dict()
    valid = {}
    for k, v in ckpt.items():
        if k in state and v.shape == state[k].shape:
            valid[k] = v
    state.update(valid)
    model.load_state_dict(state)
    return model

# ==================== 单折训练与评估 ====================
def train_fold(train_loader, test_loader, device, lambda_val, fold_seed):
    fix_seed(fold_seed)
    net = CNN1D(
        input_channels=input_channels,
        conv_params=conv_params,
        dropout_rate=dropout_rate,
        num_classes=pretrained_num_classes,
        input_length=input_length,
        num_conv_blocks=num_conv_blocks,
        use_bn_list=use_bn_list,
        use_pooling_list=use_pooling_list,
        pooling_type_list=pooling_type_list
    )
    # [沙箱最小修正] 原脚本顺序有误：dummy 已 .to(device)，net 却未迁移
    net = net.to(device)
    dummy = torch.randn(1, input_channels, input_length).to(device)
    _, _ = net(dummy)
    net = load_pretrained_weights(net, pretrained_model_path, device)

    for blk in net.conv_blocks[:num_frozen_blocks]:
        for p in blk.parameters():
            p.requires_grad = False
    net.classifier[-1] = nn.Linear(128, new_num_classes).to(device)
    net = net.to(device)

    ce_fn = nn.CrossEntropyLoss()
    cac_fn = CACLoss(num_classes=3, feat_dim=128, margin=cac_margin).to(device)
    opt_net = torch.optim.Adam(filter(lambda p:p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-3)
    opt_cac = torch.optim.Adam(cac_fn.parameters(), lr=0.5)

    train_loss = []
    train_acc = []

    for epoch in range(num_epochs):
        net.train()
        loss_m = AverageMeter()
        corr = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            feat, out = net(x)
            lce = ce_fn(out, y)
            lcac = cac_fn(feat, y)
            loss = lce + lambda_val * lcac

            opt_net.zero_grad()
            opt_cac.zero_grad()
            loss.backward()
            for p in cac_fn.parameters():
                p.grad /= lambda_val if lambda_val != 0 else 1.0
            opt_net.step()
            opt_cac.step()

            loss_m.update(loss.item(), x.size(0))
            _, pred = torch.max(out, 1)
            total += y.size(0)
            corr += (pred == y).sum().item()

        tr_loss = loss_m.avg
        tr_acc = corr / total
        train_loss.append(tr_loss)
        train_acc.append(tr_acc)

    # 训练集上计算中心与阈值
    center, tau = get_centers_and_thresholds(net, train_loader, device)
    ak, au, hos = openset_evaluate(net, test_loader, center, tau)
    return ak, au, hos, train_loss, train_acc

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    X_known, y_known, X_unknown, y_unknown = load_data(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    # ========== 遍历所有lambda，每个做5折交叉验证 ==========
    evaluated = {}
    for lam in lambda_search_list:
        print(f"\n{'='*60}")
        print(f"5折交叉验证，lambda = {lam:.4f}")
        print(f"{'='*60}")

        fold_hos = []
        fold_ak = []
        fold_au = []

        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X_known, y_known)):
            print(f"\n--- Fold {fold_idx+1} / {num_folds} ---")

            # 第一层：拆分已知类测试集 + 候选集
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
            perm = np.random.RandomState(42).permutation(len(y_test))
            X_test = X_test[perm]
            y_test = y_test[perm]

            # 构建loader
            train_loader = DataLoader(
                TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long()),
                batch_size=batch_size, shuffle=True
            )
            test_loader = DataLoader(
                TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long()),
                batch_size=batch_size, shuffle=False
            )

            # 训练并评估
            fold_seed = 42 + fold_idx
            ak, au, hos, _, _ = train_fold(train_loader, test_loader, device, lam, fold_seed)
            fold_hos.append(hos)
            fold_ak.append(ak)
            fold_au.append(au)

            print(f"    Acc_k={ak:.4f}, Acc_u={au:.4f}, HOS={hos:.4f}")

        # 计算5折平均
        avg_hos = np.mean(fold_hos)
        avg_ak = np.mean(fold_ak)
        avg_au = np.mean(fold_au)
        qualified = 1 if avg_au >= min_acc_u_threshold else 0
        evaluated[lam] = (avg_ak, avg_au, avg_hos, fold_hos)

        print(f"\nλ={lam:.4f} 5折平均: Acc_k={avg_ak:.4f}, Acc_u={avg_au:.4f}, HOS={avg_hos:.4f}")

        # 写入CSV
        with open(summary_csv_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [f"{lam:.4f}"] +
                [f"{h:.4f}" for h in fold_hos] +
                [f"{avg_hos:.4f}", f"{avg_ak:.4f}", f"{avg_au:.4f}", qualified]
            )

    # ========== 选取最优lambda ==========
    def current_best():
        qualified_cands = [lam for lam in evaluated if evaluated[lam][1] >= min_acc_u_threshold]
        if qualified_cands:
            return max(qualified_cands, key=lambda lam: evaluated[lam][2])
        return max(evaluated, key=lambda lam: evaluated[lam][2])

    best_lambda = current_best()
    best_avg_ak, best_avg_au, best_avg_hos, best_fold_hos = evaluated[best_lambda]

    # ========== 最终结果输出 ==========
    print("\n" + "="*60)
    print("5折交叉验证搜索完成！")
    print(f"最优 lambda_cac = {best_lambda:.4f}")
    print(f"5折平均 Acc_k = {best_avg_ak:.4f}")
    print(f"5折平均 Acc_u = {best_avg_au:.4f}")
    print(f"5折平均 HOS   = {best_avg_hos:.4f}")
    print(f"结果汇总表：{summary_csv_path}")
    print("="*60)
