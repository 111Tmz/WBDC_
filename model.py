# -*- coding:utf-8 -*-
"""
DeepFM + MMOE + Expert Dropout + Feed Embedding + DIN
竞赛工业版（带详细注释）
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================
# 一、评估函数
# ==========================
def uAUC(labels, preds, user_id_list):
    """按用户分组计算AUC"""
    user_pred, user_truth = defaultdict(list), defaultdict(list)

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


# ==========================
# 二、Dataset（支持DIN）
# ==========================
class MMOEDataset(Dataset):
    """
    注意：
    hist_feedid: list[int]，长度固定（padding后）
    """
    def __init__(self, df, sparse, dense, target=None, seq_len=20):
        self.df = df.reset_index(drop=True)
        self.sparse = sparse
        self.dense = dense
        self.target = target
        self.seq_len = seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ===== 普通特征 =====
        x = {f: torch.tensor(row[f], dtype=torch.long) for f in self.sparse}
        x.update({f: torch.tensor(row[f], dtype=torch.float32) for f in self.dense})

        # ===== 用户历史序列（DIN用）=====
        hist = row["hist_feedid"]  # list
        # padding（统一长度）
        if len(hist) < self.seq_len:
            hist = hist + [0]*(self.seq_len - len(hist))

        x["hist_feedid"] = torch.tensor(hist, dtype=torch.long)

        if self.target is None:
            return x

        y = [torch.tensor(row[t], dtype=torch.float32) for t in self.target]
        return x, y


# ==========================
# 三、FM层
# ==========================
class FM(nn.Module):
    """二阶特征交叉"""
    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        return 0.5 * (square_of_sum - sum_of_square)
    # 输出的是(batch, embed_dim)，每个维度都是所有特征在该维度上的二阶交叉


# ==========================
# 四、DIN Attention
# ==========================
class Attention(nn.Module):
    """DIN注意力"""
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim*4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, query, keys):
        # query: (batch, dim)
        # keys: (batch, seq_len, dim)

        seq_len = keys.size(1)
        query = query.unsqueeze(1).expand(-1, seq_len, -1)

        x = torch.cat([query, keys, query-keys, query*keys], dim=-1)

        attn = self.fc(x).squeeze(-1)
        attn = F.softmax(attn, dim=1)

        out = torch.sum(keys * attn.unsqueeze(-1), dim=1)
        return out


class DIN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = Attention(dim)

    def forward(self, hist_emb, target_emb):
        return self.attn(target_emb, hist_emb)


# ==========================
# 五、MMOE（含Expert Dropout）
# ==========================
class MMOELayer(nn.Module):
    def __init__(self, num_tasks, num_experts, input_dim, output_dim, dropout=0.2):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.dropout = dropout

        self.experts = nn.ModuleList([
            nn.Linear(input_dim, output_dim, bias=False)
            for _ in range(num_experts)
        ])

        self.gates = nn.ModuleList([
            nn.Linear(input_dim, num_experts, bias=False)
            for _ in range(num_tasks)
        ])

    def forward(self, x):
        expert_out = torch.stack([e(x) for e in self.experts], dim=2)

        if self.training:
            mask = (torch.rand(x.size(0), self.num_experts, device=x.device)
                    > self.dropout).float()

        outputs = []
        for gate in self.gates:
            gate_out = torch.softmax(gate(x), dim=1)

            if self.training:
                gate_out = gate_out * mask
                gate_out = gate_out / (gate_out.sum(dim=1, keepdim=True)+1e-8)

            gate_out = gate_out.unsqueeze(1)
            out = (expert_out * gate_out).sum(dim=2)
            outputs.append(out)

        return outputs


# ==========================
# 六、主模型
# ==========================
class Model(nn.Module):
    def __init__(self, sparse, dense, feature_sizes):
        super().__init__()

        embed_dim = 16
        self.sparse = sparse
        self.dense = dense

        # ===== Embedding =====
        self.emb = nn.ModuleDict({
            f: nn.Embedding(feature_sizes[f], embed_dim)
            for f in sparse
        })

        # ===== feed embedding（512维）=====
        # 加载预训练embedding
        emb_matrix = np.load('./data/feed_embedding.npy')

        self.feed_emb = nn.Embedding.from_pretrained(
            torch.tensor(emb_matrix, dtype=torch.float32),
            freeze=False   # 🔥 可以训练（推荐）
        )

        # ===== DIN =====
        self.din = DIN(512)

        # ===== FM =====
        self.fm = FM()

        # ===== DNN =====
        input_dim = embed_dim*len(sparse) + len(dense) + 512 + 512

        self.dnn = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # ===== MMOE =====
        self.mmoe = MMOELayer(
            num_tasks=4,
            num_experts=8,   # 🔥 改这里
            input_dim=128,
            output_dim=8,
            dropout=0.3   # 🔥 改这里
        )

        # ===== task =====
        self.out = nn.ModuleList([nn.Linear(8,1) for _ in range(4)])

    def forward(self, x):
        # ===== sparse =====
        embed = torch.stack([self.emb[f](x[f]) for f in self.sparse], dim=1)
        sparse_flat = embed.view(embed.size(0), -1)

        # ===== dense =====
        dense = torch.cat([x[f].unsqueeze(1) for f in self.dense], dim=1)

        # ===== feed embedding =====
        feed = self.feed_emb(x["feedid"])  # (batch,512)

        # ===== DIN =====
        hist = self.feed_emb(x["hist_feedid"])  # (batch,seq,512)
        user_interest = self.din(hist, feed)

        # ===== DNN =====
        dnn_input = torch.cat([sparse_flat, dense, feed, user_interest], dim=1)
        dnn_out = self.dnn(dnn_input)

        # ===== MMOE =====
        mmoe_out = self.mmoe(dnn_out)

        outputs = []
        for i in range(4):
            outputs.append(torch.sigmoid(self.out[i](mmoe_out[i])))

        return outputs
