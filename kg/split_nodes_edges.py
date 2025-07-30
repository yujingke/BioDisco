import pandas as pd
import csv

# 1. 读取原始数据
file_path = "kg.csv"
df = pd.read_csv(
    file_path,
    dtype={
        'x_id': 'string',
        'y_id': 'string'
    },
    low_memory=False
)

# 2. 处理节点数据：分别提取 x 与 y 的节点，然后合并去重
nodes_x = df[['x_id', 'x_name', 'x_source', 'x_type']].rename(
    columns={
        'x_id': 'id:ID',
        'x_name': 'name',
        'x_source': 'source',
        'x_type': ':LABEL'
    }
)
nodes_y = df[['y_id', 'y_name', 'y_source', 'y_type']].rename(
    columns={
        'y_id': 'id:ID',
        'y_name': 'name',
        'y_source': 'source',
        'y_type': ':LABEL'
    }
)
nodes = pd.concat([nodes_x, nodes_y], ignore_index=True).drop_duplicates(subset=['id:ID'])

# 3. 处理关系数据
edges = df[['x_id', 'y_id', 'relation', 'display_relation']].rename(
    columns={
        'x_id': ':START_ID',
        'y_id': ':END_ID',
        'relation': ':TYPE'
        # display_relation 保持原样
    }
)

# 4. 全局替换超长的节点ID（同时更新边数据）
THRESHOLD = 64  # ID 长度上限（可根据需要调整）
id_mapping = {}  # 用于存储需要替换的 id 对应的新短 id
counter = 1

# 为所有节点创建一个 mapping，超长的 id 替换为短 id，记录到 id_mapping
for node_id in nodes['id:ID'].unique():
    if len(node_id) > THRESHOLD:
        id_mapping[node_id] = f"Group_{counter:05d}"
        counter += 1

# 如果存在需要替换的 id，则更新节点和边数据
if id_mapping:
    # 对节点数据：为替换的节点添加 raw_id 列，并更新 id:ID 列
    nodes['raw_id'] = nodes['id:ID'].apply(lambda x: x if x in id_mapping else "")
    nodes['id:ID'] = nodes['id:ID'].apply(lambda x: id_mapping[x] if x in id_mapping else x)
    
    # 对边数据：更新 :START_ID 和 :END_ID 列
    edges[':START_ID'] = edges[':START_ID'].apply(lambda x: id_mapping[x] if x in id_mapping else x)
    edges[':END_ID'] = edges[':END_ID'].apply(lambda x: id_mapping[x] if x in id_mapping else x)

# 5. 导出节点和边 CSV 文件
nodes_path = "nodes.csv"
edges_path = "edges.csv"

nodes.to_csv(nodes_path, index=False, quoting=csv.QUOTE_ALL, quotechar='"')
edges.to_csv(edges_path, index=False, quoting=csv.QUOTE_ALL, quotechar='"')

print("Done! Generated nodes.csv and edges.csv")
