import pandas as pd
import torch
import mlflow

import yaml
from tqdm import tqdm

from src import get_model, preprocess


def run_experiment(config_path):
    # 1. Загрузка настроек
    with mlflow.start_run():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        mlflow.log_params(config)

        torch.manual_seed(config['random_seed'])
        torch.cuda.manual_seed(config['random_seed'])

        preprocessed_data = preprocess(config['model_type'], **config['data_params'])
        preprocessed_data |= config['model_params']

        model = get_model(
            config['model_type'],
            **preprocessed_data
        )

        model = model.to(config['data_params']['device'])

        print("[*] Starting training")
        for epoch in tqdm(range(config['train_params']['epochs'])):
            loss = model.train_model(preprocessed_data, **config['train_params'])
            mrr = model.test(preprocessed_data, val=True, **config['test_params'])['MRR']
            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("val_MRR", mrr, step=epoch)
            print(f'Loss: {loss:.4f}')
            print(f'MRR: {mrr:.4f}')

        print("[*] Running evaluation...")
        metrics = model.test(preprocessed_data, val=False, **config['test_params'])
        mlflow.log_params(metrics)
        print(f"Final Metrics: {metrics}")


if __name__ == "__main__":
        run_experiment("configs/distmult_config.yaml")