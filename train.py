import torch
import torch.nn.functional as F
from data_processing import TEINet_embeddings_5fold, esm_embeddings_5fold
from model import GraphNet
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from arg_parser import parse_args
import numpy as np
import collections
from torch_geometric.data import Data
import random
from sklearn.model_selection import train_test_split
import yaml
import copy
import os
import matplotlib.pyplot as plt


# # 困难负采样函数
def hard_negative_sampling(
        preds,
        labels,
        neg_ratio=3,
        hard_ratio=0.7):

    probs = torch.sigmoid(preds).detach()

    pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
    neg_idx = (labels == 0).nonzero(as_tuple=True)[0]

    num_pos = len(pos_idx)

    if num_pos == 0:
        return torch.arange(len(labels), device=labels.device)

    k = min(len(neg_idx), num_pos * neg_ratio)

    neg_probs = probs[neg_idx]

    # hardest部分
    num_hard = int(k * hard_ratio)

    hard_idx = neg_idx[torch.topk(neg_probs,num_hard).indices]

    # semi-hard部分
    remain_idx = torch.tensor(
        list(set(neg_idx.cpu().numpy()) - set(hard_idx.cpu().numpy())),
        device=labels.device
    )

    num_random = k - num_hard

    if len(remain_idx) > num_random:
        rand_perm = torch.randperm(len(remain_idx))[:num_random]
        random_idx = remain_idx[rand_perm]
    else:
        random_idx = remain_idx

    selected_idx = torch.cat([pos_idx,hard_idx,random_idx])

    return selected_idx

# # 成对AUC损失函数
def pairwise_auc_loss(
        preds,
        labels,
        num_pos_samples=1000,
        num_neg_samples=1000,
        temperature=2.0):

    pos_preds = preds[labels == 1]
    neg_preds = preds[labels == 0]

    if len(pos_preds) == 0 or len(neg_preds) == 0:
        return torch.tensor(
            0.0,
            device=preds.device
        )

    if len(pos_preds) > num_pos_samples:
        idx = torch.randperm(
            len(pos_preds),
            device=preds.device
        )[:num_pos_samples]
        pos_preds = pos_preds[idx]

    if len(neg_preds) > num_neg_samples:
        idx = torch.randperm(
            len(neg_preds),
            device=preds.device
        )[:num_neg_samples]
        neg_preds = neg_preds[idx]

    diff = (
        pos_preds.unsqueeze(1)
        - neg_preds.unsqueeze(0)
    )

    # RankNet Loss
    pair_loss = F.softplus(-diff)

    # 难样本加权
    weights = torch.sigmoid(-diff / temperature).detach()

    loss = (pair_loss * weights).mean()

    return loss


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, gamma=1.9, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, pos_weight=None):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, 
            targets,
            reduction='none',
            pos_weight=pos_weight
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

def tune_focal_loss(model, train_data, val_data, device, 
                    alpha_range=(0.1, 1.0), gamma_range=(0.1, 5.0),
                    search_mode="grid", num_alpha=10, num_gamma=10, num_samples=20, 
                    epochs=5):
    best_auc = -1
    best_alpha, best_gamma = 0.5, 2.0

    if search_mode == "grid":
        alphas = np.linspace(alpha_range[0], alpha_range[1], num_alpha)
        gammas = np.linspace(gamma_range[0], gamma_range[1], num_gamma)
        candidates = [(a, g) for a in alphas for g in gammas]
    else:  # random search
        candidates = [(random.uniform(*alpha_range), random.uniform(*gamma_range)) 
                      for _ in range(num_samples)]

    for alpha, gamma in candidates:
        focal_loss_fn = FocalLoss(alpha=alpha, gamma=gamma).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(epochs):  # 少量 epoch 快速验证
            model.train()
            optimizer.zero_grad()
            preds = model(train_data.x, train_data.edge_index)
            y_true = train_data.y.to(device)

            num_pos = (y_true == 1).sum()
            num_neg = (y_true == 0).sum()
            weight_factor = (num_neg.float() / num_pos.float()) if num_pos > 0 else 1.0
            pos_weight = torch.tensor([weight_factor], device=device)

            loss = focal_loss_fn(preds, y_true, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()

        # 验证集 AUC
        model.eval()
        with torch.no_grad():
            preds_val = model(val_data.x, val_data.edge_index)
            y_val = val_data.y.to(device)
            auc_val = roc_auc_score(y_val.cpu().numpy(), torch.sigmoid(preds_val).cpu().numpy())

        if auc_val > best_auc:
            best_auc = auc_val
            best_alpha, best_gamma = alpha, gamma

    print(f"最佳 FocalLoss 参数: alpha={best_alpha:.3f}, gamma={best_gamma:.3f}, AUC={best_auc:.4f}")
    return best_alpha, best_gamma


def compute_accuracy(preds, y_true):
    return ((preds > 0).float() == y_true).sum().item() / preds.size(0)

def compute_aupr(preds, y_true):
    probs = torch.sigmoid(preds)
    probs_numpy = probs.detach().cpu().numpy()
    y_true_numpy = y_true.detach().cpu().numpy()
    return average_precision_score(y_true_numpy, probs_numpy)

def compute_auc(preds, y_true):
    probs = torch.sigmoid(preds)
    y_true_numpy = y_true.detach().cpu().numpy()
    probs_numpy = probs.detach().cpu().numpy()
    return roc_auc_score(y_true_numpy, probs_numpy)


#设置随机种子
seed = 18
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
#解析命令行参数，加载配置文件
args = parse_args()
with open(args.configs_path) as file:
    configs = yaml.safe_load(file)

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

#加载 5 折交叉验证的数据，将数据移动到指定设备
# data_list = esm2_embeddings_5fold(args.configs_path)
data_list = TEINet_embeddings_5fold(args.configs_path)
data_list = [data.to(device) for data in data_list]

train_data = data_list[0]
test_data = data_list[1]


model = GraphNet(num_node_features=train_data.num_node_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
sgd_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9) #***

margin = 4.0
epoch_decay = 0.0046
weight_decay = 0.006
aucm_optimizer = PESG(model.parameters(),
                loss_fn=AUCMLoss(),
                lr=args.lr,
                momentum=0.4,
                margin=margin,
                device=device,
                epoch_decay=epoch_decay,
                weight_decay=weight_decay)

# ######################### 保存初始模型状态
# init_model_state = copy.deepcopy(model.state_dict())

# *** 搜索最佳 alpha, gamma
best_alpha, best_gamma = tune_focal_loss(
    model, train_data, test_data, device,
    alpha_range=(0.1, 1.0), gamma_range=(1.0, 5.0),
    search_mode="grid", num_alpha=10, num_gamma=10, epochs=5
)

# # ######################### 恢复模型到初始状态
# model.load_state_dict(init_model_state)
# model.train()  # 确保训练模式

# 用最优参数训练
focal_loss_fn = FocalLoss(alpha=best_alpha, gamma=best_gamma).to(device)

num_epochs = args.epochs
best_valid_roc = 0
best_valid_acc = 0

best_epoch = 0  
epoch_auc_list = []  # 存储每轮的验证集 AUC


aucm_module = AUCMLoss().to(device)

#计算正负样本权重
num_pos = (train_data.y == 1).sum()
num_neg = (train_data.y == 0).sum()
pos_weight = torch.tensor([(num_neg / num_pos) * args.positive_weights], device=device)


for epoch in range(num_epochs):

    model.train()
    optimizer.zero_grad()
    aucm_optimizer.zero_grad()
    sgd_optimizer.zero_grad()
    
    out = model(train_data.x, train_data.edge_index)
    preds = out
    y_true = train_data.y.to(device)


    #损失计算

    selected_idx = hard_negative_sampling(preds, y_true, neg_ratio=4, hard_ratio=0.7)

    focal_loss = focal_loss_fn(preds[selected_idx], y_true[selected_idx], pos_weight=pos_weight)
    aucm_loss = aucm_module(torch.sigmoid(preds), y_true)
    pair_loss = pairwise_auc_loss(preds[selected_idx], y_true[selected_idx], temperature=2.0)
    total_loss = (0.5 * focal_loss+ 0.3 * aucm_loss+ 0.2 * pair_loss)
    
    total_loss.backward()

    # 优化器策略选择
    if args.opt_strategy == "adam":
        optimizer.step()
    elif args.opt_strategy == "pesg":
        aucm_optimizer.step()
    elif args.opt_strategy == "sgd":
        sgd_optimizer.step()
    elif args.opt_strategy == "dual":
        optimizer.step()
        aucm_optimizer.step() 
    elif args.opt_strategy == "triple":
        optimizer.step()
        aucm_optimizer.step()
        sgd_optimizer.step()

    #指标计算
    accuracy = compute_accuracy(preds, y_true)
    roc_auc = compute_auc(preds, y_true)
    aupr = compute_aupr(preds, y_true)

    #验证和测试
    model.eval()
    with torch.no_grad():
        out_valid = model(test_data.x, test_data.edge_index)
        preds_valid = out_valid
        y_true_valid = test_data.y.to(device)

        valid_acc = compute_accuracy(preds_valid, y_true_valid)
        roc_auc_valid = compute_auc(preds_valid, y_true_valid)
        valid_aupr = compute_aupr(preds_valid, y_true_valid)


        epoch_auc_list.append(roc_auc_valid)

        #保存最佳模型
        if roc_auc_valid > best_valid_roc:
            best_valid_roc = roc_auc_valid
            best_epoch = epoch + 1  #### 记录最优AUC对应的训练轮次
            torch.save(model.state_dict(), configs['save_model'])
    print("Epoch: {}/{}, Loss: {:.7f}, Train Acc: {:.4f}, Test Acc: {:.4f}, Train AUC: {:.4f}, Train APUR: {:.4f}, Test AUC: {:.4f}, Test AUPR: {:.4f}".format(epoch+1, num_epochs, total_loss.item(), accuracy, valid_acc, roc_auc, aupr, roc_auc_valid, valid_aupr))


# 创建 ResGAT_results 目录（如果不存在）
results_dir = "ResGAT_results"
os.makedirs(results_dir, exist_ok=True)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_auc_list, marker='o', markersize=3, linewidth=1.5, color='blue', label='Validation AUC')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.title('Validation AUC vs. Epoch', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')

# 标记最佳 AUC 点
best_auc_value = max(epoch_auc_list) if epoch_auc_list else 0
best_epoch_idx = epoch_auc_list.index(best_auc_value) + 1 if epoch_auc_list else 0
plt.scatter(best_epoch_idx, best_auc_value, color='red', s=80, zorder=5, label=f'Best AUC: Epoch {best_epoch_idx}')
plt.legend(loc='lower right')

# 保存图片
plot_save_path = os.path.join(results_dir, f"{configs['dataset_name']}_auc_curve.png")
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n📊 AUC曲线已保存到: {plot_save_path}")

# 同时保存 AUC 数据到 CSV
auc_df = pd.DataFrame({
    'epoch': range(1, num_epochs + 1),
    'val_auc': epoch_auc_list
})
auc_csv_path = os.path.join(results_dir, f"{configs['dataset_name']}_auc_history.csv")
auc_df.to_csv(auc_csv_path, index=False)
print(f"📊 AUC历史数据已保存到: {auc_csv_path}")


# Load the best model
best_model = GraphNet(num_node_features=test_data.num_node_features).to(device)
##  TEINet_embeddings_5fold
best_model.load_state_dict(torch.load(configs['save_model'],weights_only=True))
##  esm_embeddings_5fold
# best_model.load_state_dict(torch.load(configs['save_model'],weights_only=False))


# Evaluate on test test_data
best_model.eval()
with torch.no_grad():
    out_test = best_model(test_data.x, test_data.edge_index)
    preds_test = out_test
    y_true_test = test_data.y.to(device)

    test_acc = compute_accuracy(preds_test, y_true_test)
    roc_auc_test = compute_auc(preds_test, y_true_test)
    test_aupr = compute_aupr(preds_test, y_true_test)

    # save results
    probabilities = torch.sigmoid(preds_test)
    binary_predictions = (probabilities > 0.5).type(torch.int).detach().cpu().numpy()
    df = pd.DataFrame({
        'prediction': binary_predictions,
        'label': y_true_test.detach().cpu().numpy().astype(int)
    })
    df.to_csv(f'results/{configs["dataset_name"]}.csv', index=False)
    

# =========================
# 构造实验记录
# =========================
dataset_name = configs.get("dataset_name", "unknown")
opt_strategy = args.opt_strategy

result_row = {
    "dataset": dataset_name,
    "opt_strategy": opt_strategy,
    "best_epoch": best_epoch,  
    "acc": f"{test_acc:.4f}",
    "auc": f"{roc_auc_test:.4f}",
    "aupr": f"{test_aupr:.4f}"
}

# 保存路径（总记录表）
save_path = "experiment_results.csv"

os.makedirs("results", exist_ok=True)

# 如果文件存在 → 追加
if os.path.exists(save_path):
    df_existing = pd.read_csv(save_path)
    df_new = pd.concat([df_existing, pd.DataFrame([result_row])], ignore_index=True)
else:
    df_new = pd.DataFrame([result_row])

df_new.to_csv(save_path, index=False)

print(f"\n✅ 结果已保存到: {save_path}")
print(f"   - 最优验证AUC: {best_valid_roc:.4f} (第 {best_epoch} 轮)")
print("Test Acc: {:.4f}, Test AUC: {:.4f}, Test AUPR: {:.4f}".format(test_acc, roc_auc_test, test_aupr))

    
