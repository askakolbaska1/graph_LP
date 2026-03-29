import pickle
import os

import pandas as pd


def num_nodes_and_relations(
        df: pd.DataFrame,
        heads: str,
        predicates: str,
        tails: str,
):

    num_nodes = len(pd.concat([df[heads], df[tails]]).unique())
    num_relations = len(df[predicates].unique())

    return num_nodes, num_relations


def merge_dicts_from_folder(
        path: str
):
    combined_dict = {}

    for filename in os.listdir(path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(path, filename)

            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                combined_dict |= data

    return combined_dict