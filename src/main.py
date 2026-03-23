import pandas as pd
import torch
import wandb

import yaml

from src import get_model
from .utils import num_nodes_and_relations



def run_experiment(config_path):
    # 1. Загрузка настроек
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config['random_seed'])
    torch.cuda.manual_seed(config['random_seed'])

    run = wandb.init(
        entity="askakolbaska-itmo-university",
        project="bioKG",
        config=config,
    )

    df = pd.read_csv(config['data_params']['input_path'])

    num_nodes, num_relations = num_nodes_and_relations(
        df,
        config['data_params']['heads_col'],
        config['data_params']['predicates_col'],
        config['data_params']['tails_col']
    )

    model = get_model(
        config['model_type'],
        num_nodes=num_nodes,
        num_relations=num_relations,
        **config['model_params']
    )

    processed_data  = model.preprocess(df, **config['data_params'])

    model = model.to(config['data_params']['device'])

    print("[*] Starting training")
    model.train_model(processed_data, **config['train_params'])

    print("[*] Running evaluation...")
    metrics = model.test(processed_data, **config['test_params'])
    run.log(metrics)
    print(f"Final Metrics: {metrics}")

    run.finish()

if __name__ == "__main__":
    run_experiment("configs/distmult_config.yaml")