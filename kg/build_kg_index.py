import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer


nodes_csv: str = "nodes.csv"
index_path: str = "node_index.faiss"
names_path: str = "node_names.pkl"
embs_path: str = "node_embs.npy"
model_name: str = "kamalkraj/BioSimCSE-BioLinkBERT-BASE"

# 读取所有节点名
df = pd.read_csv(nodes_csv)
names = df["name"].tolist()
print(f"Loaded {len(names)} node names from {nodes_csv}")

# 加载模型并向量化
model = SentenceTransformer(model_name)
embs = model.encode(names, convert_to_numpy=True, show_progress_bar=True)
print("Computed embeddings for all nodes.")

# L2 归一化
norms = np.linalg.norm(embs, axis=1, keepdims=True)
embs_norm = embs / (norms + 1e-8)

# 构建 FAISS 索引内积搜索
dim = embs_norm.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embs_norm.astype("float32"))
print(f"Built FAISS index (dim={dim}, n={embs_norm.shape[0]})")

# 持久化
faiss.write_index(index, index_path)
with open(names_path, "wb") as f:
    pickle.dump(names, f)
np.save(embs_path, embs_norm)
print(f"Saved index → {index_path}")
print(f"Saved names → {names_path}")
print(f"Saved normalized embeddings → {embs_path}")


