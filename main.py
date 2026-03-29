# -*- coding:utf-8 -*-
"""
主训练入口（main.py）
负责：
1. 数据加载
2. DataLoader
3. 模型初始化
4. 训练 + 验证
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

# ===== 导入你刚刚写的模块 =====
from model import Model, MMOEDataset, uAUC


# ==========================
# 一、评估函数（多任务）
# ==========================
def evaluate(val_labels, val_preds, userids, target):
    from collections import defaultdict
    from sklearn.metrics import roc_auc_score

    def uAUC(labels, preds, user_id_list):
        user_pred = defaultdict(list)
        user_truth = defaultdict(list)

        for i in range(len(labels)):
            uid = user_id_list[i]
            user_pred[uid].append(preds[i])
            user_truth[uid].append(labels[i])

        total_auc, size = 0.0, 0.0
        for uid in user_pred:
            if len(set(user_truth[uid])) > 1:
                total_auc += roc_auc_score(user_truth[uid], user_pred[uid])
                size += 1

        return total_auc / size if size > 0 else 0

    eval_dict = {}
    weights = {"read_comment":4, "like":3, "click_avatar":2, "forward":1}

    for i, t in enumerate(target):
        eval_dict[t] = uAUC(val_labels[i], val_preds[i], userids)

    score = sum(weights[t]*eval_dict[t] for t in eval_dict) / sum(weights.values())

    print("uAUC:", eval_dict)
    print("Weighted uAUC:", round(score,6))


# ==========================
# 二、Loss（多任务加权）
# ==========================
def multi_loss(preds, targets, weights):
    loss_fn = torch.nn.BCELoss()
    total = 0
    for i in range(len(preds)):
        total += weights[i] * loss_fn(preds[i], targets[i])
    return total / sum(weights)


# ==========================
# 三、训练函数
# ==========================
def train(model, train_loader, val_loader, userids, target, epochs=3):
    device = next(model.parameters()).device

    optimizer = torch.optim.Adagrad(
        model.parameters(),
        lr=0.01,
        weight_decay=1e-5
    )

    weights = [4,3,2,1]  # 多任务权重

    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch+1} =====")

        # ===== train =====
        model.train()
        for x, y in tqdm(train_loader):
            x = {k:v.to(device) for k,v in x.items()}
            y = [t.to(device).unsqueeze(1) for t in y]

            optimizer.zero_grad()
            out = model(x)

            loss = multi_loss(out, y, weights)
            loss.backward()
            optimizer.step()

        print("Train done")

        # ===== eval =====
        model.eval()
        preds = [[] for _ in target]
        labels = [[] for _ in target]

        with torch.no_grad():
            for x, y in val_loader:
                x = {k:v.to(device) for k,v in x.items()}
                out = model(x)

                for i in range(len(target)):
                    preds[i].append(out[i].cpu().numpy())
                    labels[i].append(y[i].numpy())

        preds = [np.concatenate(p) for p in preds]
        labels = [np.concatenate(l) for l in labels]

        evaluate(labels, preds, userids, target)


# ==========================
# 四、主函数
# ==========================
if __name__ == "__main__":

    # ===== 设备 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ===== 特征定义 =====
    target = ["read_comment","like","click_avatar","forward"]
    sparse = ['userid','feedid','authorid','bgm_song_id','bgm_singer_id']
    dense = ['videoplayseconds']

    # ===== 读取数据（已经处理好的）=====
    data = pd.read_pickle('./data/with_hist_trunc.pkl')

    # ===== 简单预处理 =====
    for f in sparse:
        data[f] = data[f].fillna(0).astype(np.int64)

    data[dense] = np.log(data[dense].fillna(0) + 1)

    # ===== 切分 =====
    train_df = data[data['date_'] < 14]
    val_df   = data[data['date_'] == 14]

    # ===== feature size =====
    feature_sizes = {f:int(data[f].max())+1 for f in sparse}

    # ===== DataLoader =====
    train_loader = DataLoader(
        MMOEDataset(train_df, sparse, dense, target, seq_len=30),
        batch_size=512,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        MMOEDataset(val_df, sparse, dense, target, seq_len=30),
        batch_size=512,
        num_workers=4
    )

    # ===== 模型 =====
    model = Model(sparse, dense, feature_sizes).to(device)

    # ===== 用户id（评估用）=====
    userids = val_df['userid'].astype(str).tolist()

    # ===== 训练 =====
    train(model, train_loader, val_loader, userids, target, epochs=3)
