# -*- coding:utf-8 -*-
"""
🔥 进阶工业版 prepare.py（带冷启动优化）

升级点：
1. embedding 对齐修复（避免错位）
2. 冷启动优化（author + global mean）
3. 更安全的 mapping 体系

输出：
- base.pkl
- hist_seq.npy
- seq_len.npy
- feed_embedding.npy
"""

import pandas as pd
import numpy as np
from collections import defaultdict, deque
from tqdm import tqdm
import os

# =======================
# 🔥 多任务负采样函数（新增）
# =======================
# =======================
# 🔥 用户级 + 多任务负采样（最终版）
# =======================
def negative_sampling_user_weighted(df, weights, neg_ratio=3):
    print("🚀 User-level + multi-task sampling...")

    dfs = []

    for uid, group in df.groupby('userid'):
        group = group.copy()

        # ===== 1️⃣ 多任务打分 =====
        score = np.zeros(len(group))

        for label, w in weights.items():
            if label in group.columns:
                group[label] = group[label].fillna(0)
                score += w * group[label].values

        group["score"] = score

        # ===== 2️⃣ 正负划分 =====
        pos = group[group["score"] > 0]
        neg = group[group["score"] == 0]

        if len(pos) == 0:
            continue

        # ===== 3️⃣ 负样本采样 =====
        neg_keep = min(len(neg), len(pos) * neg_ratio)

        if len(neg) > 0:
            # ⭐ 简单均匀采样（稳定版）
            neg_sample = neg.sample(
                n=neg_keep,
                random_state=42
            )
            sampled = pd.concat([pos, neg_sample])
        else:
            sampled = pos

        dfs.append(sampled)

    df_sampled = pd.concat(dfs)

    # 打乱
    df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"After sampling: {len(df)} -> {len(df_sampled)}")

    return df_sampled

# =======================
# 参数
# =======================
ACTION_PATH = './data/wechat_algo_data1/user_action.csv'
FEED_INFO_PATH = './data/wechat_algo_data1/feed_info.csv'
EMB_PATH = './data/wechat_algo_data1/feed_embeddings.csv'

SAVE_DIR = './data/processed'
MAX_LEN = 150 # 25%的用户历史长度 >= 94

os.makedirs(SAVE_DIR, exist_ok=True)

# =======================
# 1️⃣ 读取数据
# =======================
print("📥 Load data...")

action = pd.read_csv(ACTION_PATH)
feed = pd.read_csv(FEED_INFO_PATH)

# merge
data = action.merge(feed, on='feedid', how='left')

# 排序（DIN 必须）
data = data.sort_values(['userid', 'date_']).reset_index(drop=True)

# =======================
# 🔥 负采样（用户级 + 多任务）
# =======================
print("⚖️ Negative sampling...")

WEIGHTS = {
    "read_comment": 4,
    "like": 3,
    "click_avatar": 2,
    "forward": 1
}

data = negative_sampling_user_weighted(
    data,
    weights=WEIGHTS,
    neg_ratio=10
)

# ⚠️ 重新排序（保证DIN序列正确）
data = data.sort_values(['userid', 'date_']).reset_index(drop=True)


# =======================
# 2️⃣ 特征定义
# =======================
sparse = ['userid','feedid','authorid','bgm_song_id','bgm_singer_id']
dense = ['videoplayseconds']

# =======================
# 3️⃣ 填充缺失
# =======================
print("🧹 Fill NA...")

for f in sparse:
    data[f] = data[f].fillna(0)

data[dense] = data[dense].fillna(0)

# =======================
# 4️⃣ ID 映射（保存 mapping！）
# =======================
print("🔢 ID mapping...")

feature_sizes = {}
mapping_dicts = {}  # 🔥 保存所有 mapping

for f in sparse:
    unique_vals = data[f].unique()
    
    mapping = {v: i+1 for i, v in enumerate(unique_vals)}  # 0留padding
    
    data[f] = data[f].map(mapping)
    
    feature_sizes[f] = len(mapping) + 1
    mapping_dicts[f] = mapping  # 🔥 保存

print("feature_sizes:", feature_sizes)

# =======================
# 5️⃣ 构造历史序列（DIN）
# =======================
print("📜 Build history sequence...")

hist_dict = defaultdict(lambda: deque(maxlen=MAX_LEN))

N = len(data)
hist_seqs = np.zeros((N, MAX_LEN), dtype=np.int32)
seq_lens = np.zeros(N, dtype=np.int16)

for i, (uid, fid) in enumerate(tqdm(zip(data['userid'], data['feedid']), total=N)):
    hist = hist_dict[uid]

    seq = list(hist)
    seq_len = len(seq)

    if seq_len > 0:
        hist_seqs[i, -seq_len:] = seq

    seq_lens[i] = seq_len

    hist.append(fid)

# =======================
# 6️⃣ 读取 embedding
# =======================
print("🧠 Load pretrained embedding...")

emb_df = pd.read_csv(EMB_PATH)

feed_emb_dict = {}

for row in tqdm(emb_df.itertuples(), total=len(emb_df)):
    raw_fid = row.feedid
    emb_str = row[2]

    vec = np.fromstring(emb_str, sep=' ', dtype=np.float32)

    if len(vec) == 0:
        continue

    feed_emb_dict[raw_fid] = vec

EMB_DIM = len(next(iter(feed_emb_dict.values())))
print("Embedding dim:", EMB_DIM)

# =======================
# 🔥 7️⃣ 冷启动增强（author embedding）
# =======================
print("🧩 Build author embedding...")

author_emb_dict = defaultdict(list)

for _, row in feed.iterrows():
    raw_fid = row['feedid']
    author = row['authorid']
    
    if raw_fid in feed_emb_dict:
        author_emb_dict[author].append(feed_emb_dict[raw_fid])

# 求均值
for k in author_emb_dict:
    author_emb_dict[k] = np.mean(author_emb_dict[k], axis=0)

# 全局均值
global_mean_emb = np.mean(list(feed_emb_dict.values()), axis=0)

# =======================
# 8️⃣ 构建 embedding matrix（🔥核心优化）
# =======================
print("🔗 Build embedding matrix (with cold-start)...")

num_items = feature_sizes['feedid']
embedding_matrix = np.random.normal(
    scale=0.01,
    size=(num_items, EMB_DIM)
).astype(np.float32)

embedding_matrix[0] = 0

miss = 0
used_author = 0
used_global = 0

feedid_mapping = mapping_dicts['feedid']

for raw_fid, mapped_idx in tqdm(feedid_mapping.items(), desc="Building embedding"):

    if raw_fid in feed_emb_dict:
        embedding_matrix[mapped_idx] = feed_emb_dict[raw_fid]

    else:
        # 🔥 冷启动策略
        author = feed.loc[feed['feedid'] == raw_fid, 'authorid']
        
        if len(author) > 0:
            author = author.values[0]
        else:
            author = None

        if author in author_emb_dict:
            embedding_matrix[mapped_idx] = author_emb_dict[author]
            used_author += 1
        else:
            embedding_matrix[mapped_idx] = global_mean_emb
            used_global += 1

        miss += 1

print(f"Missing: {miss}, Use author: {used_author}, Use global: {used_global}")

# =======================
# 9️⃣ 保存
# =======================
print("💾 Save...")

data.to_pickle(f'{SAVE_DIR}/base.pkl')

np.save(f'{SAVE_DIR}/hist_seq.npy', hist_seqs)
np.save(f'{SAVE_DIR}/seq_len.npy', seq_lens)
np.save(f'{SAVE_DIR}/feed_embedding.npy', embedding_matrix)

print("✅ DONE")
