import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base import BaseModel


class MultiModalDistMult(nn.Module, BaseModel):
    def __init__(
            self,
            raw_tensors: dict[str, torch.Tensor], #словарь эмбеддингов из bert объединенный по типам молекул
            output_dim: int,
            num_relations: int
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_types = len(raw_tensors)

        #хранилища сырых эмбеддингов
        self.raw_stores = nn.ModuleDict({
            str(t_id): nn.Embedding.from_pretrained(tensor, freeze=True)
            for t_id, tensor in raw_tensors.items()
        })

        #MLP-проекторы под размерности каждого вида эмбеддингов
        self.projections = nn.ModuleDict()
        for t_id, tensor in raw_tensors.items():
            in_dim = tensor.shape[1]  # Узнаем исходную размерность (например, 1024 для белков)
            self.projections[str(t_id)] = nn.Sequential(
                nn.Linear(in_dim, output_dim * 2),
                nn.ReLU(),
                nn.LayerNorm(output_dim * 2),
                nn.Linear(output_dim * 2, output_dim)
            )

        #эмбеддинги отношений
        self.rel_emb = nn.Embedding(num_relations, output_dim)

        def forward(self, h_local, h_type, r_idx, t_local, t_type):
            batch_size = h_local.size(0)
            device = h_local.device

            #заготовки под финальные векторы размерности D
            h_proj = torch.zeros(batch_size, self.output_dim, device=device)
            t_proj = torch.zeros(batch_size, self.output_dim, device=device)

            #проходим по всем типам данных
            for type_id in range(self.num_types):
                type_str = str(type_id)

                #находим элементы этого типа в батче
                h_mask = (h_type == type_id)
                t_mask = (t_type == type_id)

                #проецируем головы
                if h_mask.any():
                    raw_h = self.raw_stores[type_str](h_local[h_mask]).float()
                    h_proj[h_mask] = self.projections[type_str](raw_h)

                #проецируем хвосты
                if t_mask.any():
                    raw_t = self.raw_stores[type_str](t_local[t_mask]).float()
                    t_proj[t_mask] = self.projections[type_str](raw_t)

            #нормализация
            h_proj = F.normalize(h_proj, p=2, dim=1)
            t_proj = F.normalize(t_proj, p=2, dim=1)

            # эмбеддинги отношений
            r = self.rel_emb(r_idx)

            #считаем скор DistMult
            score = torch.sum(h_proj * r * t_proj, dim=1)
            return score


    def preprocess(self, df, **kwargs):
        pass
