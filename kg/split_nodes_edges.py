import pandas as pd
import csv

file_path = "kg.csv"
df = pd.read_csv(
    file_path,
    dtype={
        'x_id': 'string',
        'y_id': 'string'
    },
    low_memory=False
)

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

edges = df[['x_id', 'y_id', 'relation', 'display_relation']].rename(
    columns={
        'x_id': ':START_ID',
        'y_id': ':END_ID',
        'relation': ':TYPE'

    }
)


THRESHOLD = 64 
id_mapping = {}  
counter = 1

for node_id in nodes['id:ID'].unique():
    if len(node_id) > THRESHOLD:
        id_mapping[node_id] = f"Group_{counter:05d}"
        counter += 1

if id_mapping:

    nodes['raw_id'] = nodes['id:ID'].apply(lambda x: x if x in id_mapping else "")
    nodes['id:ID'] = nodes['id:ID'].apply(lambda x: id_mapping[x] if x in id_mapping else x)
    
    edges[':START_ID'] = edges[':START_ID'].apply(lambda x: id_mapping[x] if x in id_mapping else x)
    edges[':END_ID'] = edges[':END_ID'].apply(lambda x: id_mapping[x] if x in id_mapping else x)


nodes_path = "nodes.csv"
edges_path = "edges.csv"

nodes.to_csv(nodes_path, index=False, quoting=csv.QUOTE_ALL, quotechar='"')
edges.to_csv(edges_path, index=False, quoting=csv.QUOTE_ALL, quotechar='"')

print("Done! Generated nodes.csv and edges.csv")
