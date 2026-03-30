# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler


from model import Model, MMOEDataset

# ==========================
# uAUC
# ==========================
def uAUC(labels, preds, user_id_list):
    from sklearn.metrics import roc_auc_score

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


def evaluate(val_labels, val_preds, userids, target):
    weights = {"read_comment":4, "like":3, "click_avatar":2, "forward":1}

    eval_dict = {}
    for i, t in enumerate(target):
        eval_dict[t] = uAUC(val_labels[i], val_preds[i], userids)

    score = sum(weights[t]*eval_dict[t] for t in eval_dict) / sum(weights.values())

    print("uAUC:", eval_dict)
    print("Weighted uAUC:", round(score,6))


# ==========================
# Loss
# ==========================


def multi_loss(preds, targets, weights):
    loss_fn = torch.nn.BCEWithLogitsLoss()
    total = 0
    for i in range(len(preds)):
        total += weights[i] * loss_fn(preds[i], targets[i])
    return total / sum(weights)

# ==========================
# Train（🔥优化版）
# ==========================
def train(model, train_loader, val_loader, userids, target, epochs=3):
    device = next(model.parameters()).device

    # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.9
    )


    scaler = GradScaler()  # ⭐ 混合精度

    weights = [4,3,2,1]

    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch+1} =====")

        # ===== train =====
        model.train()
        for x, y in tqdm(train_loader, desc="Training"):
            x = {k:v.to(device, non_blocking=True) for k,v in x.items()}
            y = [t.to(device, non_blocking=True).unsqueeze(1) for t in y]

            optimizer.zero_grad()

            # ⭐ 混合精度
            with autocast("cuda"):
                out = model(x)
                loss = multi_loss(out, y, weights)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print("Train done")

        # ===== eval =====
        model.eval()
        preds = [[] for _ in target]
        labels = [[] for _ in target]

        with torch.no_grad():
            # for x, y in val_loader:
            for x, y in tqdm(val_loader, desc="Evaluating"):
                x = {k:v.to(device, non_blocking=True) for k,v in x.items()}
                out = model(x)

                for i in range(len(target)):
                    # preds[i].append(out[i].cpu().numpy())
                    preds[i].append(torch.sigmoid(out[i]).cpu().numpy())
                    labels[i].append(y[i].numpy())

        preds = [np.concatenate(p) for p in preds]
        labels = [np.concatenate(l) for l in labels]

        evaluate(labels, preds, userids, target)

        # ⭐ 清显存
        torch.cuda.empty_cache()
        scheduler.step()

        print("lr:", scheduler.get_last_lr()[0])


# ==========================
# 主函数
# ==========================
if __name__ == "__main__":
    # 打印时间：
    print("Start time:", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    target = ["read_comment","like","click_avatar","forward"]
    sparse = ['userid','feedid','authorid','bgm_song_id','bgm_singer_id']
    dense = ['videoplayseconds']

    # ===== 新数据 =====
    data = pd.read_pickle('./data/processed/base.pkl')
    hist_seq = np.load('./data/processed/hist_seq.npy')

    # ===== 预处理 =====
    for f in sparse:
        data[f] = data[f].fillna(0).astype(np.int64)

    data[dense] = np.log(data[dense].fillna(0) + 1)

    # ===== 切分 =====
    train_idx = data['date_'] < 14
    val_idx   = data['date_'] == 14

    train_df = data[train_idx]
    val_df   = data[val_idx]

    train_hist = hist_seq[train_idx.values]
    val_hist   = hist_seq[val_idx.values]

    # ===== feature size =====
    feature_sizes = {f:int(data[f].max())+1 for f in sparse}

    # ===== DataLoader（🔥优化）=====
    train_loader = DataLoader(
        MMOEDataset(train_df, train_hist, None, sparse, dense, target),
        batch_size=512,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        MMOEDataset(val_df, val_hist, None, sparse, dense, target),
        batch_size=512,
        num_workers=4,
        pin_memory=True
    )

    # ===== 模型 =====
    model = Model(sparse, dense, feature_sizes).to(device)

    userids = val_df['userid'].astype(str).tolist()

    train(model, train_loader, val_loader, userids, target, epochs=3)
