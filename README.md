```
WBDC/
├── prepare.py                # 🔧 数据预处理脚本（核心：特征工程 + 序列构造 + embedding对齐）
├── model.py                  # 🧠 模型定义（DeepFM + MMOE + DIN）
├── main.py                   # 🚀 训练入口（读取数据 + 训练 + 验证）
│
├── data/
│   ├── wechat_algo_data1/    # 📦 原始数据目录（比赛提供）
│   │   ├── user_action.csv       # 用户行为日志（点击、点赞、转发等）
│   │   ├── feed_info.csv         # feed侧特征（作者、音乐、视频时长等）
│   │   ├── feed_embeddings.csv   # 预训练feed embedding（内容向量）
│
│   └── processed/            # ⚙️ 预处理后数据（prepare.py 生成）
│       ├── base.pkl              # 🧾 主训练数据（所有特征已处理）
│       │                          # 包含：
│       │                          # - sparse特征（已ID化）
│       │                          # - dense特征
│       │                          # - label（多任务目标）
│       │
│       ├── hist_seq.npy          # 📜 用户历史行为序列（DIN输入）
│       │                          # shape: [N, 50]
│       │                          # 每一行 = 当前样本对应的历史feed序列
│       │
│       ├── seq_len.npy           # 📏 序列真实长度（mask用）
│       │                          # shape: [N]
│       │                          # 用于attention时忽略padding
│       │
│       ├── feed_embedding.npy    # 🧠 feed embedding矩阵（预训练）
│                                  # shape: [num_feed+1, emb_dim]
│                                  # index=0 为padding
│                                  # 可直接加载到 nn.Embedding
```

py3.9

uv venv

激活

uv pip install -r requirents.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple

```
pandas==2.3.3
numpy==2.0.2
scikit-learn==1.6.1
tqdm==4.67.3
torch==2.8.0
```
