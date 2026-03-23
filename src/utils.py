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