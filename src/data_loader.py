from collections import defaultdict

import torch
import pandas as pd
import numpy as np
from pykeen.triples import TriplesFactory

__all__ = ['df_to_triplets_geometric','df_to_triplets_pykeen']

def df_to_triplets_geometric(
        df: pd.DataFrame,
        heads: str,
        predicates: str,
        tails: str,
        train_size: float = 0.8,
        test_and_val_size: float = 0.1,
        device: str = 'cpu',
):

    node_keys, node_values = pd.factorize(pd.concat([df[heads], df[tails]]))
    rawid2id = {k: v.item() for v, k in zip(node_keys, node_values)}

    pred_keys, pred_values = pd.factorize(df[predicates])
    pred2id = {k: v.item() for v, k in zip(pred_keys, pred_values)}

    df[heads] = df[heads].apply(lambda x: rawid2id[x])
    df[tails] = df[tails].apply(lambda x: rawid2id[x])
    df[predicates] = df[predicates].apply(lambda x: pred2id[x])

    hrt_arr = np.array([df[heads].to_numpy(), df[predicates].to_numpy(), df[tails].to_numpy()])
    hrt_tensor = torch.tensor(hrt_arr, dtype=torch.long).t()


    filtered_dict = defaultdict(set)
    triplets_list = hrt_tensor.cpu().tolist()
    for h, r, t in triplets_list:
        filtered_dict[(h, r)].add(t)


    num_triples = hrt_tensor.shape[0]

    indices = torch.randperm(num_triples)
    train_sizes = int(train_size * num_triples)
    val_sizes = int(test_and_val_size * num_triples)

    test_indices = indices[train_sizes + val_sizes:]
    val_indices = indices[train_sizes: train_sizes + val_sizes]
    train_indices = indices[:train_sizes]

    train_triplets = hrt_tensor[train_indices].to(device)
    val_triplets = hrt_tensor[val_indices].to(device)
    test_triplets = hrt_tensor[test_indices].to(device)

    return train_triplets, val_triplets, test_triplets, filtered_dict, rawid2id, pred2id


def df_to_triplets_pykeen(
        df: pd.DataFrame,
        heads: str,
        predicates: str,
        tails: str,
        train_size: float = 0.8,
        test_and_val_size: float = 0.1,
        create_inverse: bool = False,
        random_state: int = 42
):
    main_data = df.astype(str)
    triples = main_data[[heads, predicates, tails]].values
    triplet_data = TriplesFactory.from_labeled_triples(triples, create_inverse_triples=create_inverse)
    training_set, testing_set, validation_set = triplet_data.split([train_size, test_and_val_size, test_and_val_size], random_state=random_state)

    return training_set, testing_set, validation_set

