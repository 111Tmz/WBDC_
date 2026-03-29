# -*- coding:utf-8 -*-
"""
生成用户历史序列 hist_feedid
"""

import pandas as pd
from tqdm import tqdm

# ===== 读取数据 =====
data = pd.read_csv('./data/wechat_algo_data1/user_action.csv')

# 按时间排序（非常关键）
data = data.sort_values(['userid', 'date_'])

# ===== 构建历史序列 =====
hist_dict = {}
hist_list = []

for uid, fid in tqdm(zip(data['userid'], data['feedid']), total=len(data)):
    if uid not in hist_dict:
        hist_dict[uid] = []

    # 当前样本的历史（不包含当前）
    hist_list.append(hist_dict[uid].copy())

    # 更新历史
    hist_dict[uid].append(fid)

data['hist_feedid'] = hist_list

# 保存
data.to_pickle('./data/with_hist.pkl')
print("hist_feedid done")
