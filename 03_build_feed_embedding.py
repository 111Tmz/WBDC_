# -*- coding:utf-8 -*-
"""
把 feed_embedding.csv 转成 embedding matrix
"""

import pandas as pd
import numpy as np

# ===== 读取 =====
feed_emb = pd.read_csv('./data/wechat_algo_data1/feed_embedding.csv')

# 格式：
# feedid, embedding (字符串)

# ===== 转成向量 =====
def parse_emb(s):
    return np.array([float(x) for x in s.split()])

feed_emb['emb'] = feed_emb['embedding'].apply(parse_emb)

# ===== 构建矩阵 =====
max_id = feed_emb['feedid'].max()
embed_dim = len(feed_emb['emb'].iloc[0])

embedding_matrix = np.zeros((max_id + 1, embed_dim))

for _, row in feed_emb.iterrows():
    embedding_matrix[row['feedid']] = row['emb']

# 保存
np.save('./data/feed_embedding.npy', embedding_matrix)

print("feed embedding done:", embedding_matrix.shape)
