import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# ==================================================
# 1.固定随机种子
# ==================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
NUM_FOLDS = 5
set_seed(SEED)

# ==================================================
# 2.参数配置
# ==================================================
PRETRAIN_MODEL_PATH = r"E:\OneDrive\project\open-set-ours\models\source_domain_best.pth"
DATA_PATH = r'E:/OneDrive/project/open-set-ours/data/open_210.csv'
SAVE_ROOT = r"E:/OneDrive/project/open-set-ours/legacy_out_D\6种方法\6种方法\C2AE"
FIG_SAVE_DIR = os.path.join(SAVE_ROOT, "figures")
RESULT_CSV_PATH = os.path.join(SAVE_ROOT, "c2ae_5fold_result.csv")

# 基础超参
BATCH_SIZE = 32
EPOCHS_AE = 200
LR_AE = 0.0010
FREEZE_BLOCKS = 2
FEAT_DIM = 128
NUM_KNOWN_CLASSES = 3
THRESHOLD_QUANTILE = 0.97  # 训练集重建误差分位数阈值
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(FIG_SAVE_DIR, exist_ok=True)
print("当前设备:", DEVICE)

# 结果CSV表头
with open(RESULT_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "fold_id", "threshold",
        "Acc_k", "Acc_u", "HOS"
    ])

# ==================================================
# 3.CNN1D主干模型
# ==================================================
class CNN1D(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN1D, self).__init__()
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=21, stride=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(16, 16, kernel_size=21, stride=2),
                nn.BatchNorm1d(16),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=7, stride=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(32, 32, kernel_size=7, stride=2),
                nn.BatchNorm1d(32),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=3, stride=2),
                nn.BatchNorm1d(64),
                nn.ReLU()
            )
        ])
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.pool2 = nn.AvgPool1d(kernel_size=3, stride=2)
        self.pool3 = nn.AvgPool1d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_blocks[0](x)
        x = self.conv_blocks[1](x)
        x = self.pool1(x)
        x = self.conv_blocks[2](x)
        x = self.conv_blocks[3](x)
        x = self.pool2(x)
        x = self.conv_blocks[4](x)
        x = self.pool3(x)
        x = self.flatten(x)
        feat = self.classifier[:-1](x)
        out = self.classifier[-1](feat)
        return feat, out

# ==================================================
# 4.C2AE 类条件自编码器
# ==================================================
class C2AE_Autoencoder(nn.Module):
    def __init__(self, feat_dim=128, hidden_dim=79, latent_dim=24):
        super(C2AE_Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

# ==================================================
# 5.数据集类
# ==================================================
class WeldDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (self.X[index], self.y[index])

# ==================================================
# 6.数据加载（仅分离已知/未知，划分在5折循环内完成）
# ==================================================
def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path, header=0, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, header=0, encoding="gbk")
    
    labels = df.iloc[:, 0].astype(str)
    X_all = df.iloc[:, 1:].values.astype(np.float32)

    X_known_list = []
    y_known_list = []
    X_unknown_list = []
    y_unknown_list = []

    for i, label in enumerate(labels):
        parts = label.split("-")
        if len(parts) < 4:
            continue
        cls_raw = int(parts[3])
        feat = X_all[i][np.newaxis, :]  # 形状 (1, 801)，保留通道维度
        if cls_raw == 1:
            X_known_list.append(feat)
            y_known_list.append(0)
        elif cls_raw == 2:
            X_known_list.append(feat)
            y_known_list.append(1)
        elif cls_raw == 3:
            X_known_list.append(feat)
            y_known_list.append(2)
        elif cls_raw == 4:
            X_unknown_list.append(feat)
            y_unknown_list.append(3)

    # 保持三维形状 (N, 1, 801)，匹配1D卷积输入要求
    X_known = np.stack(X_known_list, axis=0)
    y_known = np.array(y_known_list)
    X_unknown = np.stack(X_unknown_list, axis=0)
    y_unknown = np.array(y_unknown_list)

    print("===== 数据集总览 =====")
    print(f"已知类总数：{len(X_known)}，好{(y_known==0).sum()}，坏{(y_known==1).sum()}，光板{(y_known==2).sum()}")
    print(f"未知类总数：{len(X_unknown)}")
    print("---------------------")
    return X_known, y_known, X_unknown, y_unknown

# ==================================================
# 7.加载预训练主干 + 冻结层
# ==================================================
def load_pretrained_backbone():
    model = CNN1D(num_classes=5)
    checkpoint = torch.load(PRETRAIN_MODEL_PATH, map_location=DEVICE, weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    delete_keys = [k for k in list(state_dict.keys()) if "classifier" in k]
    for k in delete_keys:
        del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    model.classifier[-1] = nn.Linear(128, 3)
    return model

def freeze_layers(model, freeze_blocks=2):
    for idx, block in enumerate(model.conv_blocks):
        if idx < freeze_blocks:
            for param in block.parameters():
                param.requires_grad = False
        else:
            for param in block.parameters():
                param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = False
    return model

# ==================================================
# 8.训练3个类条件自编码器 + 训练集校准阈值
# ==================================================
def train_c2ae(backbone, X_train, y_train, fold_save_dir):
    backbone.eval()
    ae_list = []
    criterion_mse = nn.MSELoss()

    for cls_id in range(NUM_KNOWN_CLASSES):
        print(f"\n--- 训练第 {cls_id} 类的自编码器 ---")
        cls_mask = (y_train == cls_id)
        X_cls = X_train[cls_mask]
        ds_cls = WeldDataset(X_cls, y_train[cls_mask])
        loader_cls = DataLoader(ds_cls, batch_size=BATCH_SIZE, shuffle=True)

        ae = C2AE_Autoencoder(feat_dim=FEAT_DIM).to(DEVICE)
        optimizer_ae = optim.Adam(ae.parameters(), lr=LR_AE)

        for epoch in range(EPOCHS_AE):
            total_loss = 0.0
            ae.train()
            for x, _ in loader_cls:
                x = x.to(DEVICE)
                with torch.no_grad():
                    feat, _ = backbone(x)
                recon_feat = ae(feat)
                loss = criterion_mse(recon_feat, feat)

                optimizer_ae.zero_grad()
                loss.backward()
                optimizer_ae.step()
                total_loss += loss.item()
            
            if (epoch+1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS_AE}] 重建损失: {total_loss/len(loader_cls):.6f}")

        ae.eval()
        ae_list.append(ae)
        torch.save(ae.state_dict(), os.path.join(fold_save_dir, f"ae_class_{cls_id}.pth"))

    # 训练集上计算阈值
    all_min_recon_errors = []
    train_all_ds = WeldDataset(X_train, y_train)
    train_all_loader = DataLoader(train_all_ds, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for x, _ in train_all_loader:
            x = x.to(DEVICE)
            feat, _ = backbone(x)
            errors = []
            for ae in ae_list:
                recon = ae(feat)
                err = torch.mean((recon - feat)**2, dim=1)
                errors.append(err.unsqueeze(1))
            errors_mat = torch.cat(errors, dim=1)
            min_err, _ = torch.min(errors_mat, dim=1)
            all_min_recon_errors.extend(min_err.cpu().numpy().tolist())
    
    err_arr = np.array(all_min_recon_errors)
    auto_threshold = np.quantile(err_arr, THRESHOLD_QUANTILE)
    print(f"\n自动阈值（训练集{THRESHOLD_QUANTILE}分位数）= {auto_threshold:.6f}")
    return ae_list, auto_threshold

# ==================================================
# 9.绘制混淆矩阵
# ==================================================
def plot_confusion_matrix(fold_id, y_true_all, y_pred_all, save_path):
    y_pred_map = np.where(np.array(y_pred_all) == 4, 3, np.array(y_pred_all))
    cm = confusion_matrix(y_true_all, y_pred_map, labels=[0,1,2,3])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["good_weld","bad_weld","plate","unknown"],
                yticklabels=["good_weld","bad_weld","plate","unknown"])
    plt.title(f"Fold {fold_id} C2AE OpenSet Confusion Matrix", fontsize=13)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==================================================
# 10.开放集评估
# ==================================================
def c2ae_evaluate(backbone, ae_list, X_test, y_test, threshold, fold_id):
    test_set = WeldDataset(X_test, y_test)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    backbone.eval()
    for ae in ae_list:
        ae.eval()

    y_true = []
    y_pred_open = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            feat, _ = backbone(x)
            errors = []
            for ae in ae_list:
                recon = ae(feat)
                err = torch.mean((recon - feat)**2, dim=1)
                errors.append(err.unsqueeze(1))
            errors_mat = torch.cat(errors, dim=1)
            min_err, pred_cls = torch.min(errors_mat, dim=1)
            final_pred = torch.where(min_err < threshold, pred_cls, torch.full_like(pred_cls, 4))
            y_true.extend(y.cpu().numpy().tolist())
            y_pred_open.extend(final_pred.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred_open = np.array(y_pred_open)

    mask_known = (y_true != 3)
    mask_unknown = (y_true == 3)
    acc_k = np.mean(y_pred_open[mask_known] == y_true[mask_known]) if np.sum(mask_known)>0 else 0.0
    acc_u = np.mean(y_pred_open[mask_unknown] == 4) if np.sum(mask_unknown) > 0 else 0.0

    if (acc_k + acc_u) > 1e-8:
        hos = 2 * acc_k * acc_u / (acc_k + acc_u + 1e-8)
    else:
        hos = 0.0

    # 保存混淆矩阵
    cm_path = os.path.join(FIG_SAVE_DIR, f"fold{fold_id}_c2ae_cm.png")
    plot_confusion_matrix(fold_id, y_true, y_pred_open, cm_path)
    return acc_k, acc_u, hos

# ==================================================
# 11.单折完整流程
# ==================================================
def run_single_fold(X_known, y_known, X_unknown, y_unknown, fold, train_val_idx, test_idx):
    print(f"\n==================== Fold {fold} ====================")
    fold_dir = os.path.join(SAVE_ROOT, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

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
    perm = np.random.RandomState(SEED).permutation(len(y_test))
    X_test = X_test[perm]
    y_test = y_test[perm]

    # 加载并冻结主干
    set_seed(SEED + fold)
    backbone = load_pretrained_backbone()
    backbone = freeze_layers(backbone, freeze_blocks=FREEZE_BLOCKS)
    backbone.to(DEVICE)

    # 训练C2AE自编码器 + 自动生成阈值
    ae_list, auto_threshold = train_c2ae(backbone, X_train, y_train, fold_dir)

    # 开放集测试评估
    ak, au, hos = c2ae_evaluate(backbone, ae_list, X_test, y_test, auto_threshold, fold)
    print(f"Fold{fold} 结果: Acc_k={ak:.4f}, Acc_u={au:.4f}, HOS={hos:.4f}")

    # 写入CSV
    with open(RESULT_CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([fold, auto_threshold, ak, au, hos])
    
    return ak, au, hos, auto_threshold

# ==================================================
# 12.主程序：5折分层交叉验证
# ==================================================
def main():
    X_known, y_known, X_unknown, y_unknown = load_data(DATA_PATH)
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    
    fold_metrics = []  # 存储每折 [Acc_k, Acc_u, HOS]
    fold_thresholds = []

    # 遍历5折，仅对已知类做分层划分
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X_known, y_known), start=1):
        ak, au, hos, auto_thr = run_single_fold(
            X_known, y_known, X_unknown, y_unknown,
            fold=fold_idx,
            train_val_idx=train_val_idx,
            test_idx=test_idx
        )
        fold_metrics.append([ak, au, hos])
        fold_thresholds.append(auto_thr)

    # 计算5折结果的均值与标准差
    fold_metrics = np.array(fold_metrics)
    mean_metrics = fold_metrics.mean(axis=0)
    std_metrics = fold_metrics.std(axis=0)
    mean_threshold = np.mean(fold_thresholds)

    # 格式化输出最终结果
    print("\n" + "="*60)
    print("C2AE 5折交叉验证最终结果（均值 ± 标准差）")
    print("="*60)
    print(f"已知类准确率 KCA (Acc_k) = {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
    print(f"未知类准确率 UCA (Acc_u) = {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
    print(f"开放集调和平均 HOS        = {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
    print(f"平均自动阈值              = {mean_threshold:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
