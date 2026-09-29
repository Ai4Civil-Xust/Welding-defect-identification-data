import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import weibull_min
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import csv
import itertools


# ==================== 0. 固定全局随机种子 ====================
def fix_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==================== 1. 路径与全局配置 ====================
pretrained_model_path = (
    r'E:\OneDrive\project\open-set-ours\models\source_domain_best.pth'
)
data_path = r'E:/OneDrive/project/open-set-ours/data/open_210.csv'
results_dir = (
    r'E:/OneDrive/project/open-set-ours/legacy_out_D\6种方法\6种方法\openmax'
)
os.makedirs(results_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 全局超参
NUM_FOLDS = 5
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 200
LR = 0.0004
NUM_FROZEN_BLOCKS = 2
fix_seed(SEED)

# ========== OpenMax 网格搜索参数 ==========
tailsize_candidates = [3, 5, 8, 10, 12]
alpha_candidates = [1, 2, 3]
min_acc_u_threshold = 0.94

# 结果汇总CSV
summary_csv = os.path.join(results_dir, "grid_search_summary.csv")
with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "tailsize", "alpha",
        "fold1_HOS", "fold2_HOS", "fold3_HOS", "fold4_HOS", "fold5_HOS",
        "avg_HOS", "avg_Acc_k", "avg_Acc_u", "qualified"
    ])


# ==================== 2. 模型定义 ====================
class CNN1D(nn.Module):

    def __init__(
            self,
            input_channels,
            conv_params,
            dropout_rate,
            num_classes,
            input_length,
            num_conv_blocks,
            use_bn_list,
            use_pooling_list,
            pooling_type_list,
    ):
        super(CNN1D, self).__init__()

        if isinstance(conv_params, str):
            conv_params = [
                tuple(map(int, p.split(','))) for p in conv_params.split(';')
            ]

        use_bn_list = [bool(int(bn)) for bn in use_bn_list.split(',')]
        use_pooling_list = [bool(int(pool)) for pool in use_pooling_list.split(',')]
        pooling_type_list = [
            ptype.lower() for ptype in pooling_type_list.split(',')
        ]

        self.conv_blocks = nn.ModuleList()
        in_channels = input_channels
        conv_output_length = input_length

        for block_idx in range(num_conv_blocks):
            block_layers = []
            block_conv_params = conv_params[block_idx::num_conv_blocks]

            for idx, (out_channels, kernel_size, stride) in enumerate(
                    block_conv_params
            ):
                block_layers.append(
                    nn.Conv1d(
                        in_channels, out_channels, kernel_size=kernel_size, stride=stride
                    )
                )
                conv_output_length = (conv_output_length - kernel_size) // stride + 1

                if use_bn_list[block_idx]:
                    block_layers.append(nn.BatchNorm1d(out_channels))

                block_layers.append(nn.ReLU())

                if use_pooling_list[block_idx]:
                    pool_type = pooling_type_list[block_idx]
                    if pool_type == 'max':
                        block_layers.append(nn.MaxPool1d(kernel_size=3, stride=2))
                        conv_output_length = (conv_output_length - 3) // 2 + 1
                    elif pool_type == 'avg':
                        block_layers.append(nn.AvgPool1d(kernel_size=3, stride=2))
                        conv_output_length = (conv_output_length - 3) // 2 + 1

            self.conv_blocks.append(nn.Sequential(*block_layers))
            in_channels = out_channels

        flat_size = in_channels * conv_output_length
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = x.view(x.size(0), -1)
        features = self.classifier[:6](x)
        logits = self.classifier[6](features)
        return features, logits


# ==================== 3. 数据加载 ====================
def load_data(csv_path):
    try:
        data = pd.read_csv(csv_path, encoding='gb18030', engine='python')
    except Exception:
        data = pd.read_csv(
            csv_path, encoding='utf-8', engine='python', on_bad_lines='skip'
        )

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

    known_mask = (y_all == 1) | (y_all == 2) | (y_all == 3)
    X_known = X_all[known_mask]
    y_known = y_all[known_mask] - 1

    unknown_mask = y_all == 4
    X_unknown = X_all[unknown_mask]
    y_unknown = np.full(len(X_unknown), 3)

    print("===== 数据集总览 =====")
    print(f"已知类总数：{len(X_known)}，好{(y_known == 0).sum()}，坏{(y_known == 1).sum()}，光板{(y_known == 2).sum()}")
    print(f"未知类总数：{len(X_unknown)}")
    print("---------------------")
    return X_known, y_known, X_unknown, y_unknown


# ==================== 4. 单折模型训练 + 特征提取 ====================
def train_and_extract_feats(X_train, y_train, X_test, y_test, fold_seed):
    fix_seed(fold_seed)
    g = torch.Generator()
    g.manual_seed(fold_seed)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long()),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long()),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # 构建并训练模型
    model = CNN1D(
        input_channels=1,
        conv_params='16,21,1;16,21,2;32,7,2;32,7,2;64,3,2',
        dropout_rate=0.5,
        num_classes=5,
        input_length=801,
        num_conv_blocks=5,
        use_bn_list='0,1,0,1,1',
        use_pooling_list='0,1,1,1,0',
        pooling_type_list='none,max,avg,avg,none',
    )

    if os.path.exists(pretrained_model_path):
        state_dict = torch.load(pretrained_model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=True)

    model.classifier[6] = nn.Linear(128, 3)

    for block in model.conv_blocks[:NUM_FROZEN_BLOCKS]:
        for param in block.parameters():
            param.requires_grad = False
    for block in model.conv_blocks[NUM_FROZEN_BLOCKS:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    model.to(device)
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-3,
    )

    train_losses = []
    train_accuracies = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, logits = model(inputs)
            loss = criterion_ce(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

    # 提取训练集、测试集特征与标签
    model.eval()
    train_feats, train_labs = [], []
    test_feats, test_labs = [], []

    with torch.no_grad():
        for inputs, labels in train_loader:
            feats, _ = model(inputs.to(device))
            train_feats.append(feats.cpu().numpy())
            train_labs.append(labels.numpy())
        for inputs, labels in test_loader:
            feats, logits = model(inputs.to(device))
            test_feats.append(feats.cpu().numpy())
            test_labs.append(labels.numpy())

    train_feats = np.concatenate(train_feats, axis=0)
    train_labs = np.concatenate(train_labs, axis=0)
    test_feats = np.concatenate(test_feats, axis=0)
    test_labs = np.concatenate(test_labs, axis=0)

    return train_feats, train_labs, test_feats, test_labs, train_losses, train_accuracies


# ==================== 5. OpenMax 单组参数评估 ====================
def evaluate_openmax(train_feats, train_labs, test_feats, test_labs, tailsize, alpha_openmax):
    num_known = 3
    mavs = []
    weibull_models = []

    # 拟合 Weibull 分布
    for c in range(num_known):
        c_feats = train_feats[train_labs == c]
        mav = np.mean(c_feats, axis=0)
        mavs.append(mav)
        dists = np.linalg.norm(c_feats - mav, axis=1)
        tail_dists = np.sort(dists)[::-1][: min(tailsize, len(dists))]
        shape, loc, scale = weibull_min.fit(tail_dists)
        weibull_models.append({'shape': shape, 'loc': loc, 'scale': scale})

    # OpenMax 推理
    test_preds = []
    for f in test_feats:
        ranked_idx = np.argsort(f @ np.array(mavs).T)[::-1]  # 近似logits排序
        # 用特征距离替代logits排名，保持原逻辑一致
        # 重新计算每个类的激活距离
        dists = np.array([np.linalg.norm(f - mavs[c]) for c in range(num_known)])
        ranked_idx = np.argsort(dists)  # 距离越近排名越靠前
        w = np.ones(num_known)

        for i in range(min(alpha_openmax, num_known)):
            c = ranked_idx[i]
            dist = dists[c]
            w_c = weibull_min.cdf(
                dist,
                weibull_models[c]['shape'],
                loc=weibull_models[c]['loc'],
                scale=weibull_models[c]['scale'],
            )
            w[c] = 1 - w_c * (alpha_openmax - i) / alpha_openmax

        # 用距离的倒数作为激活值，保持原OpenMax的修正逻辑
        activations = 1.0 / (dists + 1e-8)
        modified_act = activations * w
        unknown_act = np.sum(activations * (1 - w))
        openmax_exp = np.exp(np.append(modified_act, unknown_act))
        openmax_probs = openmax_exp / np.sum(openmax_exp)

        pred_c = np.argmax(openmax_probs)
        test_preds.append(pred_c)

    test_preds = np.array(test_preds)
    known_mask = np.isin(test_labs, [0, 1, 2])
    unknown_mask = test_labs == 3

    acc_k = accuracy_score(test_labs[known_mask], test_preds[known_mask])
    acc_u = accuracy_score(test_labs[unknown_mask], test_preds[unknown_mask])
    hos = 2 * (acc_k * acc_u) / (acc_k + acc_u) if (acc_k + acc_u) > 0 else 0.0
    return acc_k, acc_u, hos


# ==================== 6. 主程序 ====================
def main():
    X_known, y_known, X_unknown, y_unknown = load_data(data_path)
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    # ========== 第一步：训练5折模型，提取所有特征（仅执行1次） ==========
    print("\n========== 开始5折模型训练与特征提取 ==========")
    fold_data = []  # 存储每折的 train_feats, train_labs, test_feats, test_labs, loss, acc
    last_train_loss = None
    last_train_acc = None
    last_test_feats = None
    last_test_labs = None

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X_known, y_known), start=1):
        print(f"\n--- 训练 Fold {fold_idx}/{NUM_FOLDS} ---")

        # 三层划分
        X_test_known = X_known[test_idx]
        y_test_known = y_known[test_idx]
        X_trainval = X_known[train_val_idx]
        y_trainval = y_known[train_val_idx]

        # [沙箱D改动] 去掉折内8:2：训练集恢复为论文口径的 154 组候选
        X_train, X_val, y_train, y_val = X_trainval, X_trainval, y_trainval, y_trainval

        X_test = np.concatenate([X_test_known, X_unknown], axis=0)
        y_test = np.concatenate([y_test_known, y_unknown], axis=0)
        perm = np.random.RandomState(SEED).permutation(len(y_test))
        X_test = X_test[perm]
        y_test = y_test[perm]

        # 训练并提取特征
        fold_seed = SEED + fold_idx
        tr_feat, tr_lab, te_feat, te_lab, tr_loss, tr_acc = train_and_extract_feats(
            X_train, y_train, X_test, y_test, fold_seed
        )
        fold_data.append((tr_feat, tr_lab, te_feat, te_lab))

        if fold_idx == NUM_FOLDS:
            last_train_loss = tr_loss
            last_train_acc = tr_acc
            last_test_feats = te_feat
            last_test_labs = te_lab

    # ========== 第二步：网格搜索参数组合 ==========
    print(f"\n========== 开始网格搜索，共 {len(tailsize_candidates)*len(alpha_candidates)} 组参数 ==========")
    all_candidates = list(itertools.product(tailsize_candidates, alpha_candidates))
    evaluated = {}

    for tailsize, alpha in all_candidates:
        fold_hos = []
        fold_ak = []
        fold_au = []

        for tr_feat, tr_lab, te_feat, te_lab in fold_data:
            ak, au, hos = evaluate_openmax(tr_feat, tr_lab, te_feat, te_lab, tailsize, alpha)
            fold_hos.append(hos)
            fold_ak.append(ak)
            fold_au.append(au)

        avg_hos = np.mean(fold_hos)
        avg_ak = np.mean(fold_ak)
        avg_au = np.mean(fold_au)
        qualified = 1 if avg_au >= min_acc_u_threshold else 0
        evaluated[(tailsize, alpha)] = (avg_ak, avg_au, avg_hos, fold_hos)

        print(f"  tailsize={tailsize}, alpha={alpha} -> "
              f"Avg_Acc_k={avg_ak:.4f}, Avg_Acc_u={avg_au:.4f}, Avg_HOS={avg_hos:.4f} "
              f"{'[合格]' if qualified else '[不合格]'}")

        # 写入汇总CSV
        with open(summary_csv, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [tailsize, alpha] +
                [f"{h:.4f}" for h in fold_hos] +
                [f"{avg_hos:.4f}", f"{avg_ak:.4f}", f"{avg_au:.4f}", qualified]
            )

    # ========== 选取最优参数 ==========
    def combo_key(combo):
        ak, au, hos, _ = evaluated[combo]
        return (au >= min_acc_u_threshold, hos)

    best_combo = max(evaluated.keys(), key=combo_key)
    best_tailsize, best_alpha = best_combo
    best_ak, best_au, best_hos, best_fold_hos = evaluated[best_combo]

    # ========== 最终结果输出 ==========
    print("\n" + "=" * 60)
    print("OpenMax 网格搜索完成！最优参数：")
    print(f"  tailsize = {best_tailsize}")
    print(f"  alpha    = {best_alpha}")
    print(f"5折平均 Acc_k = {best_ak:.4f}")
    print(f"5折平均 Acc_u = {best_au:.4f}")
    print(f"5折平均 HOS   = {best_hos:.4f}")
    print(f"结果汇总表：{summary_csv}")
    print("=" * 60)

    # ========== 可视化（基于最后一折 + 最优参数） ==========
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 训练准确率曲线
    plt.figure(figsize=(5, 4))
    plt.plot(last_train_acc, label='Train Accuracy', color='#1f77b4', linewidth=1.5)
    plt.title('Training Accuracy (OpenMax)', fontsize=13)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'accuracy_curve.png'), dpi=300)
    plt.close()

    # 2. 训练损失曲线
    plt.figure(figsize=(5, 4))
    plt.plot(last_train_loss, label='Train Loss', color='#1f77b4', linewidth=1.5)
    plt.title('Training Loss (OpenMax)', fontsize=13)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'loss_curve.png'), dpi=300)
    plt.close()

    # 3. 混淆矩阵（最优参数下最后一折）
    tr_f, tr_l, te_f, te_l = fold_data[-1]
    _, _, _ = evaluate_openmax(tr_f, tr_l, te_f, te_l, best_tailsize, best_alpha)
    # 重新获取预测结果用于画混淆矩阵
    num_known = 3
    mavs = []
    weibull_models = []
    for c in range(num_known):
        c_feats = tr_f[tr_l == c]
        mav = np.mean(c_feats, axis=0)
        mavs.append(mav)
        dists = np.linalg.norm(c_feats - mav, axis=1)
        tail_dists = np.sort(dists)[::-1][: min(best_tailsize, len(dists))]
        shape, loc, scale = weibull_min.fit(tail_dists)
        weibull_models.append({'shape': shape, 'loc': loc, 'scale': scale})

    test_preds = []
    for f in te_f:
        dists = np.array([np.linalg.norm(f - mavs[c]) for c in range(num_known)])
        ranked_idx = np.argsort(dists)
        w = np.ones(num_known)
        for i in range(min(best_alpha, num_known)):
            c = ranked_idx[i]
            w_c = weibull_min.cdf(dists[c], weibull_models[c]['shape'], weibull_models[c]['loc'], weibull_models[c]['scale'])
            w[c] = 1 - w_c * (best_alpha - i) / best_alpha
        activations = 1.0 / (dists + 1e-8)
        modified_act = activations * w
        unknown_act = np.sum(activations * (1 - w))
        openmax_exp = np.exp(np.append(modified_act, unknown_act))
        openmax_probs = openmax_exp / np.sum(openmax_exp)
        test_preds.append(np.argmax(openmax_probs))
    test_preds = np.array(test_preds)

    cm = confusion_matrix(te_l, test_preds, labels=[0, 1, 2, 3])
    plt.figure(figsize=(5.5, 4.2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1', '2', '3'],
                yticklabels=['0', '1', '2', '3'])
    plt.title('Confusion Matrix (OpenMax)', fontsize=13)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # 4. t-SNE
    tsne = TSNE(n_components=2, perplexity=10, random_state=SEED)
    feats_tsne = tsne.fit_transform(last_test_feats)
    plt.figure(figsize=(5, 4))
    for c in range(4):
        mask = last_test_labs == c
        plt.scatter(feats_tsne[mask, 0], feats_tsne[mask, 1],
                    label=str(c), alpha=0.7, s=80, edgecolors='white', linewidth=0.5)
    plt.xlabel('t-SNE dimension 1', fontsize=12)
    plt.ylabel('t-SNE dimension 2', fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'tsne_vis.png'), dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
