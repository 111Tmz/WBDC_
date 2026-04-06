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
    def __init__(self, df, hist_seq, seq_len, sparse, dense, target=None):
        self.df = df.reset_index(drop=True)
        self.hist_seq = hist_seq
        self.seq_len = seq_len
        self.sparse = sparse
        self.dense = dense
        self.target = target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ===== 普通特征 =====
        x = {f: torch.tensor(row[f], dtype=torch.long) for f in self.sparse}
        x.update({f: torch.tensor(row[f], dtype=torch.float32) for f in self.dense})

        # ===== 直接用 numpy =====
        x["hist_feedid"] = torch.tensor(self.hist_seq[idx], dtype=torch.long)

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



class Dice(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        # BatchNorm1d 在 (B, C, L) 中，对每个通道 C，在 (B, L) 上计算均值和方差
        # 它规定第1（下标）维就是通道维（dim）
        self.bn = nn.BatchNorm1d(dim, eps=eps)
        self.alpha = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        if x.dim() == 3:
            # (batch, seq_len, dim)
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)
        else:
            x = self.bn(x)

        p = torch.sigmoid(x)
        return p * x + (1 - p) * self.alpha * x


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim*4, 128), # nn.Linear作用在最后一个维度上，输入是4*dim，输出是128
            Dice(128),   # ⭐ 改这里
            nn.Linear(128, 1)
        )

    def forward(self, query, keys, mask):
        '''
        (query, keys, mask)的形状分别是(B, D)，(B, L, D)，(B, L)
        query是候选商品的嵌入表示，keys是用户历史行为序列的嵌入表示，mask是历史行为序列的掩码
        (B,D), (B,L,D), (B,L) -> (B,L)
        '''
        seq_len = keys.size(1) # keys的形状是(B, L, dim)，所以seq_len是L，即序列长度
        query = query.unsqueeze(1).expand(-1, seq_len, -1) 
        # query的形状是(B, D)，通过unsqueeze(1)变成(B, 1, D)，再通过expand(-1, seq_len, -1)变成(B, L, D)，与keys对齐

        x = torch.cat([query, keys, query-keys, query*keys], dim=-1) # shape (B, L, 4*dim)
        attn = self.fc(x).squeeze(-1)# shape (B, L, 1) -> (B, L)

        # mask
        attn = attn.masked_fill(mask == 0, torch.finfo(attn.dtype).min)
        # torch.finfo(attn.dtype).min 是 attn 数据类型的最小值，通常是一个非常大的负数，这样在 softmax 后对应位置的权重接近于0
        # finfo 是一个函数，返回一个对象，这个对象包含了attn数据类型的各种信息，包括最小值、最大值、精度等。这里我们用它来获取attn数据类型的最小值，以便在mask中使用。

        # ❗ 仍然必须用 softmax
        attn = torch.softmax(attn, dim=1) # shape (B, L)，在序列长度维度上进行softmax，得到注意力权重

        out = torch.sum(keys * attn.unsqueeze(-1), dim=1) 
        # keys的形状是(B, L, dim)，attn.unsqueeze(-1)的形状是(B, L, 1)，
        # 通过广播机制，keys * attn.unsqueeze(-1)的形状是(B, L, dim)，然后在序列长度维度上求和，得到(B, dim)

        return out



class DIN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = Attention(dim) 

    def forward(self, hist_emb, target_emb, mask):
        # hist_emb: (B, L, D)，用户历史行为序列的嵌入表示，形状是 (batch_size, seq_len, embed_dim)
        # target_emb: (B, D)，候选商品的嵌入表示，形状是 (batch_size, embed_dim)
        # mask: (B, L)，历史行为序列的掩码，形状是 (batch_size, seq_len)，其中1表示有效位置，0表示无效位置
        return self.attn(target_emb, hist_emb, mask)


# ==========================
# 五、MMOE（含Expert Dropout）
# ==========================


class MMOELayer(nn.Module):
    def __init__(self, num_experts, num_tasks, input_dim, output_dim, expert_units=32, gate_units=32, dropout=0.2):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.dropout = dropout

        self.experts =  nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim,expert_units),
                nn.ReLU(),
                nn.Linear(expert_units,output_dim)
            ) for _ in range(num_experts) 
        ])

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim,gate_units),
                nn.ReLU(),
                nn.Linear(gate_units,num_experts)
            ) for _ in range(num_tasks) 
        ])
        
        
    def forward(self, x):
        expert_out = torch.stack([e(x) for e in self.experts], dim=2)
        # expert_out的形状是(batch_size, output_dim, num_experts)，每个专家的输出在最后一个维度上
        
        mask = torch.ones(x.size(0), self.num_experts, device=x.device)

        if self.training:
            mask = (torch.rand(x.size(0), self.num_experts, device=x.device)
                    > self.dropout).float()
            # mask的形状是(batch_size, num_experts)，每个专家对应一个0/1的掩码，表示是否被drop掉，
            # 这样设计是因为要和gate_out的形状对齐，gate_out的形状是(batch_size, num_experts)
        
        outputs = []
        for gate in self.gates:
            gate_out = torch.softmax(gate(x), dim=1)
            # gate_out的形状是(batch_size, num_experts)，表示每个专家的权重
            gate_out = gate_out * mask
            gate_out = gate_out / (gate_out.sum(dim=1, keepdim=True)+1e-8)
            # 对gate_out进行归一化，确保权重和为1，避免drop掉的专家权重过大
            gate_out = gate_out.unsqueeze(1) 
            # gate_out的形状变为(batch_size, 1, num_experts)，方便和expert_out进行加权求和
            out = (expert_out*gate_out).sum(dim=2)
            # expert_out的形状是(batch_size, output_dim, num_experts)，
            # gate_out的形状是(batch_size, 1, num_experts)，
            # 通过广播机制，expert_out*gate_out的形状是(batch_size, output_dim, num_experts)，
            # 然后在专家维度上求和，得到(batch_size, output_dim)，即每个任务
            outputs.append(out)
        return outputs

# ==========================
# 六、主模型
# ==========================

class Model(nn.Module):
    def __init__(self, sparse, dense, feature_sizes):
        super().__init__()

        embed_dim = 16
        feed_dim = 64   # 🔥 新维度
        self.sparse = sparse
        self.dense = dense

        # ===== sparse embedding =====
        self.emb = nn.ModuleDict({
            f: nn.Embedding(feature_sizes[f], embed_dim)
            for f in sparse
        })

        # ===== LR =====
        self.linear = nn.ModuleDict({
            f: nn.Embedding(feature_sizes[f], 1)
            for f in sparse
        })

        # ===== 原始 feed embedding（512）=====
        emb_matrix = np.load('./data/processed/feed_embedding.npy')
        self.feed_emb_raw = nn.Embedding.from_pretrained(
            torch.tensor(emb_matrix, dtype=torch.float32),
            freeze=False
        )

        # 🔥 ===== 降维层（512 → 64）=====

        emb_dim = emb_matrix.shape[1]

        self.feed_proj = nn.Sequential(
            nn.Linear(emb_dim,128),
            nn.ReLU(),
            nn.Linear(128,64)
        )


        # ===== DIN（改成64维）=====
        self.din = DIN(feed_dim)

        # ===== FM =====
        self.fm = FM()

        # ===== DNN =====
        input_dim = embed_dim * len(sparse) + len(dense) + feed_dim + feed_dim

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
            num_experts=8,
            input_dim=128,
            output_dim=8,
            dropout=0.3
        )

        # ===== 融合层 =====
        self.final = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1 + 1 + 8, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
            for _ in range(4)
        ])

    def forward(self, x):
        # ===== sparse embedding =====
        embed = torch.stack([self.emb[f](x[f]) for f in self.sparse], dim=1)
        sparse_flat = embed.view(embed.size(0), -1)

        # ===== LR =====
        linear_part = torch.sum(
            torch.cat([self.linear[f](x[f]) for f in self.sparse], dim=1),
            dim=1,
            keepdim=True
        )  # (batch,1)

        # ===== FM =====
        fm_part = self.fm(embed).sum(dim=1, keepdim=True)  # (batch,1)

        # ===== dense =====
        dense = torch.cat([x[f].unsqueeze(1) for f in self.dense], dim=1)

        # ===== feed embedding（降维）=====
        feed_raw = self.feed_emb_raw(x["feedid"])      # (batch,512)
        feed = self.feed_proj(feed_raw)                # (batch,64)

        # ===== hist embedding（降维）=====
        hist_raw = self.feed_emb_raw(x["hist_feedid"])  # (batch,seq,512)
        hist = self.feed_proj(hist_raw)                 # (batch,seq,64)
        

        # ===== DIN =====
        mask = (x["hist_feedid"] != 0).float()  # (batch, seq_len)
        user_interest = self.din(hist, feed, mask)            # (batch,64)

        # ===== DNN =====
        dnn_input = torch.cat([sparse_flat, dense, feed, user_interest], dim=1)
        dnn_out = self.dnn(dnn_input)

        # ===== MMOE =====
        mmoe_out = self.mmoe(dnn_out)

        outputs = []
        for i in range(4):
            deep_part = mmoe_out[i]  # (batch,8)

            concat_feat = torch.cat([
                linear_part,
                fm_part,
                deep_part
            ], dim=1)

            logit = self.final[i](concat_feat)
            # outputs.append(torch.sigmoid(logit))
            outputs.append(logit)  # ⭐ 不要 sigmoid

        return outputs
