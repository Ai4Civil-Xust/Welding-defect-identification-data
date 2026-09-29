import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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
pretrained_model_path = r'E:\OneDrive\project\open-set-ours\models\source_domain_best.pth'
data_path = r'E:\OneDrive\project\open-set-ours\data\open_210.csv'
save_dir = r'E:\OneDrive\project\open-set-ours\results\sens_figs\6种方法\敏感性分析\中心'

os.makedirs(save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

            for idx, (out_channels, kernel_size, stride) in enumerate(block_conv_params):
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


# ==================== 3. Center Loss 定义 ====================
class CenterLoss(nn.Module):
    def __init__(self, num_classes=3, feat_dim=128):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = (
            torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        )
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long().to(x.device)
        labels_expand = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expand.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e12).sum() / batch_size
        return loss


# ==================== 4. 数据加载 ====================
def load_raw_dataset(data_path):
    try:
        data = pd.read_csv(data_path, encoding='gb18030', engine='python')
    except Exception:
        data = pd.read_csv(data_path, encoding='utf-8', engine='python', on_bad_lines='skip')

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
            except Exception:
                pass

    X_all = features[valid_idx][:, np.newaxis, :]
    y_all = np.array(label_classes)

    known_mask = (y_all == 1) | (y_all == 2) | (y_all == 3)
    X_known = X_all[known_mask]
    y_known = y_all[known_mask] - 1

    unknown_mask = y_all == 4
    X_unknown = X_all[unknown_mask]
    y_unknown = np.full(len(X_unknown), 3)

    return X_known, y_known, X_unknown, y_unknown


# ==================== 5. 模型初始化（冻结前2个卷积块） ====================
def build_and_init_model():
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
        state_dict = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)

    model.classifier[6] = nn.Linear(128, 3)

    for block in model.conv_blocks[:2]:
        for param in block.parameters():
            param.requires_grad = False
    for block in model.conv_blocks[2:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model.to(device)


# ==================== 6. 单折训练与评估 ====================
def evaluate_one_fold(train_x, train_y, val_x, val_y, test_x, test_y, lr_model, lr_center, lambda_cent, target_recall, num_epochs=200):
    fix_seed(42)
    g = torch.Generator().manual_seed(42)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_x).float(), torch.tensor(train_y).long()),
        batch_size=32, shuffle=True, generator=g
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_x).float(), torch.tensor(val_y).long()),
        batch_size=32, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(test_x).float(), torch.tensor(test_y).long()),
        batch_size=32, shuffle=False
    )

    model = build_and_init_model()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(num_classes=3, feat_dim=128).to(device)

    optimizer_model = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_model)
    optimizer_cent = optim.SGD(criterion_cent.parameters(), lr=lr_center)

    # 训练
    for _ in range(num_epochs):
        model.train()
        criterion_cent.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            features, logits = model(inputs)
            loss = criterion_ce(logits, labels) + lambda_cent * criterion_cent(features, labels)

            optimizer_model.zero_grad()
            optimizer_cent.zero_grad()
            loss.backward()
            optimizer_model.step()
            optimizer_cent.step()

    # 训练集特征均值计算类中心
    model.eval()
    criterion_cent.eval()
    train_feats_list, train_labs_list = [], []
    with torch.no_grad():
        for inputs, labels in train_loader:
            feats, _ = model(inputs.to(device))
            train_feats_list.append(feats.cpu())
            train_labs_list.append(labels.cpu())

    train_feats = torch.cat(train_feats_list, dim=0)
    train_labs = torch.cat(train_labs_list, dim=0)
    
    centers = torch.zeros(3, 128)
    for c in range(3):
        mask = train_labs == c
        if torch.sum(mask) > 0:
            centers[c] = train_feats[mask].mean(dim=0)

    # 验证集计算自适应阈值
    val_feats_list, val_labs_list = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            feats, _ = model(inputs.to(device))
            val_feats_list.append(feats.cpu())
            val_labs_list.append(labels.cpu())

    val_feats = torch.cat(val_feats_list, dim=0)
    val_labs = torch.cat(val_labs_list, dim=0)

    # [沙箱改动] γ 机制：τ_c = 训练集第 c 类最大距离 × γ（标定集=训练集，无泄漏）
    GAMMA = float(os.environ.get('ABL_GAMMA', '1.8'))
    tau = {}
    for c in range(3):
        mask = train_labs == c
        if torch.sum(mask) > 0:
            c_feats = train_feats[mask]
            dists = torch.pow(c_feats - centers[c], 2).sum(dim=1).numpy()
            tau[c] = float(dists.max()) * GAMMA
        else:
            tau[c] = 0.0

    # 测试集推理
    test_preds, test_gts = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            feats, _ = model(inputs.to(device))
            feats_cpu = feats.cpu()
            for f in feats_cpu:
                dists = [torch.pow(f - centers[cc], 2).sum().item() for cc in range(3)]
                min_d = min(dists)
                pred_c = np.argmin(dists)
                test_preds.append(pred_c if min_d < tau[pred_c] else 3)
            test_gts.extend(labels.numpy().tolist())

    test_preds = np.array(test_preds)
    test_gts = np.array(test_gts)

    known_mask = (test_gts == 0) | (test_gts == 1) | (test_gts == 2)
    unknown_mask = test_gts == 3

    acc_k = accuracy_score(test_gts[known_mask], test_preds[known_mask]) * 100.0
    acc_u = accuracy_score(test_gts[unknown_mask], test_preds[unknown_mask]) * 100.0
    hos = 2 * (acc_k * acc_u) / (acc_k + acc_u) if (acc_k + acc_u) > 0 else 0.0

    return acc_k, acc_u, hos


# ==================== 7. 五折交叉验证 ====================
def run_5fold_cross_validation(X_known, y_known, X_unknown, y_unknown, lr_model, lr_center, lambda_cent, target_recall):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    kca_list, uca_list, hos_list = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_known, y_known)):
        train_x, train_y = X_known[train_idx], y_known[train_idx]
        val_x, val_y = X_known[val_idx], y_known[val_idx]
        test_x = np.concatenate([val_x, X_unknown], axis=0)
        test_y = np.concatenate([val_y, y_unknown], axis=0)

        kca, uca, hos = evaluate_one_fold(
            train_x, train_y, val_x, val_y, test_x, test_y,
            lr_model, lr_center, lambda_cent, target_recall
        )

        kca_list.append(kca)
        uca_list.append(uca)
        hos_list.append(hos)

    return np.mean(kca_list), np.mean(uca_list), np.mean(hos_list)



# ==================== 9. 绘制 η_center 敏感性分析图 ====================
def plot_eta_center(X_known, y_known, X_unknown, y_unknown):
    # 固定另外两个基准最优参数 + 目标召回率
    best_lr_model = 0.001385620113552
    best_lambda = 0.007673465906479
    best_target_recall = 0.93

    # 5个点全部实际运行，中间为最优值
    eta_values = [0.001, 0.003, 0.005, 0.008, 0.01]
    x_labels = ['0.001', '0.003', '0.005*', '0.008', '0.01']

    # 逐个计算五折平均结果
    kca_res, uca_res, hos_res = [], [], []
    for eta in eta_values:
        k, u, h = run_5fold_cross_validation(
            X_known, y_known, X_unknown, y_unknown,
            lr_model=best_lr_model,
            lr_center=eta,
            lambda_cent=best_lambda,
            target_recall=best_target_recall
        )
        kca_res.append(k)
        uca_res.append(u)
        hos_res.append(h)

    # 绘图样式设置（和η_model完全一致）
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    colors = {
        'KCA': '#4159a8',
        'UCA': '#e85539',
        'HOS': '#00b394'
    }
    bar_width = 0.25
    import json as _json
    _json.dump({'x_labels': x_labels, 'kca': kca_res, 'uca': uca_res, 'hos': hos_res},
               open(r'E:\OneDrive\project\open-set-ours\results\sens_eta_center.json', 'w', encoding='utf-8'),
               ensure_ascii=False, indent=1)
    print('JSON 已保存: sens_eta_center.json')

    x = np.arange(len(x_labels))

    plt.figure(figsize=(8, 6), dpi=300)

    # 分组柱状图
    plt.bar(x - bar_width, kca_res, width=bar_width, color=colors['KCA'], label='KCA ($Acc_k$)')
    plt.bar(x, uca_res, width=bar_width, color=colors['UCA'], label='UCA ($Acc_u$)')
    plt.bar(x + bar_width, hos_res, width=bar_width, color=colors['HOS'], label='HOS')

    # 柱顶数值：字号、偏移和模板完全一致
    offset_y = 0.5
    for i, v in enumerate(kca_res):
        plt.text(i - bar_width, v + offset_y, f'{v:.2f}%', ha='center', va='bottom', fontsize=6.8)
    for i, v in enumerate(uca_res):
        plt.text(i, v + offset_y, f'{v:.2f}%', ha='center', va='bottom', fontsize=6.8)
    for i, v in enumerate(hos_res):
        plt.text(i + bar_width, v + offset_y, f'{v:.2f}%', ha='center', va='bottom', fontsize=6.8)

    # 坐标轴
    plt.xticks(x, x_labels, fontsize=11)
    plt.ylim(0, 110)
    plt.yticks(np.arange(0, 111, 20), fontsize=11)
    plt.ylabel('Accuracy / Score (%)', fontsize=12)
    plt.xlabel('$\eta$_center', fontsize=13)
    plt.tick_params(axis='both', direction='in', pad=6, rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # 图例：和模板完全一致
    plt.legend(
        loc='upper left',
        fontsize=7,
        framealpha=1,
        borderpad=0.3,
        handlelength=1.2,
        labelspacing=0.8
    )

    plt.tight_layout()
    save_path = os.path.join(r'E:\OneDrive\project\open-set-ours\results\sens_figs', save_dir, 'sensitivity_eta_center.png')
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"η_center 敏感性分析图已保存到：{save_path}")





# ==================== 主程序 ====================
if __name__ == '__main__':
    fix_seed(42)
    X_known, y_known, X_unknown, y_unknown = load_raw_dataset(data_path)
    print(f"数据加载完成：已知类{len(X_known)}条，未知类{len(X_unknown)}条")
    print("正在生成 η_model 敏感性分析图...")
    plot_eta_center(X_known, y_known, X_unknown, y_unknown)


