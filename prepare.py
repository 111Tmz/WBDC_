# -*- coding:utf-8 -*-
"""
完整版 prepare.py（工业级）

包含：
1. 行为数据 + feed特征 join
2. ID 映射（全部 sparse 特征）
3. 历史序列（DIN）
4. 预训练 embedding 对齐

输出：
- base.pkl（包含所有特征）
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
# 参数
# =======================
ACTION_PATH = './data/wechat_algo_data1/user_action.csv'
FEED_INFO_PATH = './data/wechat_algo_data1/feed_info.csv'
EMB_PATH = './data/wechat_algo_data1/feed_embeddings.csv'

SAVE_DIR = './data/processed'
MAX_LEN = 50

os.makedirs(SAVE_DIR, exist_ok=True)

# =======================
# 1️⃣ 读取数据
# =======================
print("📥 Load data...")

action = pd.read_csv(ACTION_PATH)
feed = pd.read_csv(FEED_INFO_PATH)

# ===== merge =====
data = action.merge(feed, on='feedid', how='left')

# 排序（关键）
data = data.sort_values(['userid', 'date_'])

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
# 4️⃣ ID 映射（全部 sparse）
# =======================
print("🔢 ID mapping...")

feature_sizes = {}

for f in sparse:
    unique_vals = data[f].unique()
    mapping = {v: i+1 for i, v in enumerate(unique_vals)}  # 0留padding
    data[f] = data[f].map(mapping)
    feature_sizes[f] = len(mapping) + 1

print("feature_sizes:", feature_sizes)

# =======================
# 5️⃣ 构造历史序列（只用 feedid）
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
# 6️⃣ 处理 embedding
# =======================
print("🧠 Load pretrained embedding...")

emb_df = pd.read_csv(EMB_PATH)

feed_emb_dict = {}

for row in tqdm(emb_df.itertuples(), total=len(emb_df)):
    fid = row.feedid
    emb_str = row[2]

    vec = np.fromstring(emb_str, sep=' ', dtype=np.float32)

    if len(vec) == 0:
        continue

    feed_emb_dict[fid] = vec

# 自动识别维度
EMB_DIM = len(next(iter(feed_emb_dict.values())))
print("Embedding dim:", EMB_DIM)

# =======================
# 7️⃣ 构建 embedding matrix
# =======================
print("🔗 Build embedding matrix...")

num_items = feature_sizes['feedid']

embedding_matrix = np.random.normal(
    scale=0.01,
    size=(num_items, EMB_DIM)
).astype(np.float32)

embedding_matrix[0] = 0

miss = 0

# ⚠️ 注意：mapping后的 feedid
reverse_map = {v:k for k,v in zip(data['feedid'], data['feedid'])}

for raw_fid, idx in zip(action['feedid'], data['feedid']):
    if raw_fid in feed_emb_dict:
        embedding_matrix[idx] = feed_emb_dict[raw_fid]
    else:
        miss += 1

print("missing embedding:", miss)

# =======================
# 8️⃣ 保存
# =======================
print("💾 Save...")

data.to_pickle(f'{SAVE_DIR}/base.pkl')

np.save(f'{SAVE_DIR}/hist_seq.npy', hist_seqs)
np.save(f'{SAVE_DIR}/seq_len.npy', seq_lens)
np.save(f'{SAVE_DIR}/feed_embedding.npy', embedding_matrix)

print("✅ DONE")
