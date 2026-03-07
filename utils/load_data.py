import os
import pickle
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def load_and_merge_embeddings(folder_path):
    features_dict = {}

    #ключи должны соответствовать значениям в колонках 'type_entity' dataframe
    mapping = {
        "DNA": "dnabert_DNA.pkl",
        "NucleicAmbigous": "dnabert_NucleicAmbigous.pkl",
        "NucleicMixed": "dnabert_NucleicMixed.pkl",
        "AA": "protein_esm_35M.pkl",
        "RNA": "rna_berta.pkl",
        "SmallMolecule": "sm_chemberta_10M_MTR.pkl"
    }

    for entity_type, file_name in mapping.items():
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                # Загружаем словарь {raw_id: embedding}
                data = pickle.load(f)
                features_dict[entity_type] = data
                print(f"Loaded {entity_type}: {len(data)} entities")
        else:
            print(f"Warning: File {file_name} not found in {folder_path}")

    return features_dict



def df_to_homo_different_emb(df: pd.DataFrame, features_dict: dict):
    #получаем df: node_type->raw_id
    nodes = pd.concat([
        df[["type_entity_1","id_entity_1"]].rename(columns={"type_entity_1":"t","id_entity_1":"id"}),
        df[["type_entity_2","id_entity_2"]].rename(columns={"type_entity_2":"t","id_entity_2":"id"}),
    ], axis=0).drop_duplicates()

    key = list(zip(nodes["t"].astype(str), nodes["id"].astype(int))) #cписок из кортежей (node_type, raw_id)
    node_map = {k:i for i,k in enumerate(key)} #словарь вида {(node_type, raw_id):id_nodes}
    num_nodes = len(node_map) #общее количество узлов

    src = [node_map[(str(t), int(i))] for t,i in zip(df["type_entity_1"], df["id_entity_1"])] #спиcок id_nodes для heads триплетов
    dst = [node_map[(str(t), int(i))] for t,i in zip(df["type_entity_2"], df["id_entity_2"])] #спиcок id_nodes для tales триплетов
    edge_index = torch.tensor([src, dst], dtype=torch.long) #edge_index графа (все ребра) в виде tensor[2,num_edges]

    preds = df["predicate"].astype(str).unique().tolist() #список всех предикатов в графе
    pred2pred_id = {p:i for i,p in enumerate(sorted(preds))} #словарь {pred:pred_id}
    edge_type = torch.tensor([pred2pred_id[p] for p in df["predicate"].astype(str)], dtype=torch.long) #pred_id ребер графа в виде tensor[num_edges]

    edge_index, edge_type = to_undirected(edge_index, edge_type) #дублируем напарвления т.к. граф неориентированный

    #создание словаря вида {node_type:raw_emb_size}
    dims_dict = {}
    for t, emb_dict in features_dict.items():
        sample_emb = next(iter(emb_dict.values()))
        dims_dict[t] = len(sample_emb)

    unique_types = list(features_dict.keys()) #все представленный типы узлов
    type2id = {t: i for i, t in enumerate(unique_types)} #словарь {node_type:node_type_id}

    max_dim = max(dims_dict.values())  #размерность максимального эмбеддинга
    x = torch.zeros(num_nodes, max_dim) #в дальнейшем заполним эмбеддингами
    node_type_tensor = torch.zeros(num_nodes, dtype=torch.long) #сразу создаем пустой тензор

    #итерируемся напрямую по маппингу
    missing_features = 0
    for (t, raw_id), mapped_id in node_map.items():
        dim = dims_dict[t] #размерность эмбеддинга текущей вершины

        #получаем эмбеддинг
        if raw_id in features_dict[t]:
            emb = features_dict[t][raw_id]
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=torch.float)
        else:
            emb = torch.zeros(dim)
            missing_features += 1

        #записываем в x наш эмбеддинг
        x[mapped_id, :dim] = emb
        node_type_tensor[mapped_id] = type2id[t]
    #на случай ошибок
    if missing_features > 0:
        print(f"Warning: {missing_features} nodes are missing features and were zero-padded.")

    #созадем torch-geomertic data
    data = Data(x=x, edge_index=edge_index)
    data.edge_type = edge_type
    data.node_type = node_type_tensor

    return data, node_map, pred2pred_id, type2id, dims_dict