# -*- coding:utf-8 -*-
"""
限制历史序列长度
"""

import pandas as pd

SEQ_LEN = 30  # 🔥 你可以改：20~50

data = pd.read_pickle('./data/with_hist.pkl')

def truncate(seq):
    if len(seq) > SEQ_LEN:
        return seq[-SEQ_LEN:]  # 只保留最近行为
    return seq

data['hist_feedid'] = data['hist_feedid'].apply(truncate)

data.to_pickle('./data/with_hist_trunc.pkl')
print("truncate done")
