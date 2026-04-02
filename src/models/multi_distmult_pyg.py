import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

from pathlib import Path
from itertools import islice
from collections import defaultdict

from src.base import BaseModel
from src.utils import merge_dicts_from_folder
from src.data_loader import MDMDataset


class MultiModalDistMult(nn.Module, BaseModel):
    def __init__(
            self,
            raw_tensors: dict[str, torch.Tensor], #словарь эмбеддингов из bert объединенный по типам молекул
            hidden_channels: int,
            num_relations: int
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_types = len(raw_tensors)

        #хранилища сырых эмбеддингов
        self.raw_stores = nn.ModuleDict({
            str(t_id): nn.Embedding.from_pretrained(tensor, freeze=True)
            for t_id, tensor in raw_tensors.items()
        })

        #MLP-проекторы под размерности каждого вида эмбеддингов
        self.projections = nn.ModuleDict()
        for t_id, tensor in raw_tensors.items():
            in_dim = tensor.shape[1]  #узнаем исходную размерность эмбеддингов для каждого типа
            self.projections[str(t_id)] = nn.Sequential(
                nn.Linear(in_dim, hidden_channels * 2),
                nn.ReLU(),
                nn.LayerNorm(hidden_channels * 2),
                nn.Linear(hidden_channels * 2, hidden_channels)
            )

        #эмбеддинги отношений
        self.rel_emb = nn.Embedding(num_relations, hidden_channels)

    def forward(self, h_local, h_type, r_idx, t_local, t_type):
        batch_size = h_local.size(0)
        device = h_local.device

        #заготовки под финальные векторы размерности D
        h_proj = torch.zeros(batch_size, self.hidden_channels, device=device)
        t_proj = torch.zeros(batch_size, self.hidden_channels, device=device)

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


    @staticmethod
    def preprocess(**kwargs):
        rawid2enb = merge_dicts_from_folder(
            path=kwargs.get('dicts_folder'),
        )

        node_types = kwargs.get('node_types')
        nodes_folder = kwargs.get('nodes_folder')
        edges = kwargs.get('edges')
        heads = str(kwargs.get('heads_col'))
        predicates = str(kwargs.get('predicates_col'))
        tails = str(kwargs.get('tails_col'))
        train_size = kwargs.get('train_size', 0.8)
        val_size = kwargs.get('val_size', 0.1)
        test_size = kwargs.get('test_size', 0.1)
        device = kwargs.get('device')


        type2id = {name: i for i, name in enumerate(node_types)}

        rawid_to_local = {}
        rawid_to_type = {}
        raw_tensors = {}

        nodes_folder = Path(nodes_folder)

        for t_name in node_types:
            file_path = nodes_folder / f"{t_name}.csv"

            # Читаем колонку с rawid, чтобы узнать тип узлов
            df_nodes = pd.read_csv(file_path, usecols=['id_entity'])
            embeddings_list = []

            for local_idx, raw_id in enumerate(df_nodes['id_entity']):
                #заполняем словари для DataLoader
                rawid_to_local[raw_id] = local_idx
                rawid_to_type[raw_id] = type2id[t_name]

                #берем готовый эмбеддинг из твоего словаря
                emb = rawid2enb[raw_id]
                embeddings_list.append(emb)

            #склеиваем список тензоров - получится тензор размерности [N_nodes_of_this_type, embedding_dim]
            raw_tensors[str(type2id[t_name])] = torch.stack(embeddings_list)

        df_edges = pd.read_csv(edges)

        #мапим предикаты в числа
        pred_keys, pred_values = pd.factorize(df_edges[predicates])
        df_edges[predicates] = pred_keys
        num_relations = len(pred_values)

        #мапим rawid сразу в локальные индексы и типы
        h_local = df_edges[heads].map(rawid_to_local).to_numpy()
        h_type = df_edges[heads].map(rawid_to_type).to_numpy()

        t_local = df_edges[tails].map(rawid_to_local).to_numpy()
        t_type = df_edges[tails].map(rawid_to_type).to_numpy()
        rel_idx = df_edges[predicates].to_numpy()

        #тензоры для DataLoader
        h_local_tensor = torch.tensor(h_local, dtype=torch.long)
        h_type_tensor = torch.tensor(h_type, dtype=torch.long)
        rel_tensor = torch.tensor(rel_idx, dtype=torch.long)
        t_local_tensor = torch.tensor(t_local, dtype=torch.long)
        t_type_tensor = torch.tensor(t_type, dtype=torch.long)

        dataset = MDMDataset(
            h_local_tensor,
            h_type_tensor,
            rel_tensor,
            t_local_tensor,
            t_type_tensor,
        )

        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
        # train_loader = DataLoader(dataset=train_set, batch_size=train_batch_size, shuffle=True)
        # val_loader = DataLoader(dataset=val_set, batch_size=val_test_batch_size, shuffle=True)
        # test_loader = DataLoader(dataset=test_set, batch_size=val_test_batch_size, shuffle=True)



        data = {
            'h_local_tensor': h_local_tensor,
            'h_type_tensor': h_type_tensor,
            'rel_tensor': rel_tensor,
            't_local_tensor': t_local_tensor,
            't_type_tensor': t_type_tensor,
            'train_set': train_set,
            'val_set': val_set,
            'test_set': test_set,
            'raw_tensors': raw_tensors,
            'num_relations': num_relations,
            'device': device,
        }

        return data


    def train_model(self, data: dict, **kwargs):
        lr = kwargs.get('lr', 0.001)
        margin = kwargs.get('margin', 1)
        batch_size = kwargs.get('batch_size', 1024)

        train_set = data['train_set']
        device = data['device']

        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        optimizer = Adam(self.parameters(), lr=lr)
        loss_fn = nn.MarginRankingLoss(margin=margin)

        self.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            #берем реальные триплеты
            h_loc, h_typ, r_id, t_loc, t_typ = [b.to(device) for b in batch]
            batch_size = h_loc.size(0)

            #считаем для них скор
            pos_scores = self.forward(h_loc, h_typ, r_id, t_loc, t_typ)

            #генерируем негативные триплеты внутри батча
            perm = torch.randperm(batch_size, device=device)
            neg_t_loc = t_loc[perm]
            neg_t_typ = t_typ[perm]

            #считаем скор для негативных триплетов
            neg_scores = self.forward(h_loc, h_typ, r_id, neg_t_loc, neg_t_typ)

            #считаем ошибку
            target = torch.ones_like(pos_scores)
            loss = loss_fn(pos_scores, neg_scores, target)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        return avg_loss

    def prepare_evaluation(self, data: dict, **kwargs):
        all_h_local = data['h_local_tensor']
        all_h_type = data['h_type_tensor']
        all_r = data['rel_tensor']
        all_t_local = data['t_local_tensor']
        all_t_type = data['t_type_tensor']
        device = data['device']

        self.eval()

        #проецируем все узлы в общее пространство
        all_embs = []
        entity2eval_id = {}  #маппинг (local_id, type_id) -> индекс в матрице оценки
        eval_id = 0

        with torch.no_grad():
            for type_id in range(self.num_types):
                type_str = str(type_id)
                num_nodes = self.raw_stores[type_str].weight.size(0)

                #достаем и проецируем все узлы одного из типов
                locs = torch.arange(num_nodes, device=device)
                raw = self.raw_stores[type_str](locs).float()
                proj = self.projections[type_str](raw)
                proj = F.normalize(proj, p=2, dim=1)

                all_embs.append(proj)

                #запоминаем, какая строка новой матрицы какому узлу принадлежит
                for local_id in range(num_nodes):
                    entity2eval_id[(local_id, type_id)] = eval_id
                    eval_id += 1

        #матрица всех узлов графа [N_total, D]
        E_all = torch.cat(all_embs, dim=0)

        #создаем словарь для фильтрации (h_eval_id, r) -> set(t_eval_id)
        filter_dict = defaultdict(set)

        #проходимся по всем ребрам графа
        for i in range(len(all_h_local)):
            h_key = (all_h_local[i].item(), all_h_type[i].item())
            t_key = (all_t_local[i].item(), all_t_type[i].item())
            r = all_r[i].item()

            h_eval_id = entity2eval_id[h_key]
            t_eval_id = entity2eval_id[t_key]

            filter_dict[(h_eval_id, r)].add(t_eval_id)

        return E_all, entity2eval_id, filter_dict


    def test(self, data: dict, val = False, **kwargs):
        k_list = kwargs.get('k_list', [1, 5, 10, 50])
        batch_size = kwargs.get('batch_size', 64)
        N_val = kwargs.get('N_val', 10000)
        N_test = kwargs.get('N_test', None)

        val_set = data['val_set']
        test_set = data['test_set']
        device = data['device']

        num_batches = N_val // batch_size
        if val:
            dataloader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)
            cut_loader = islice(dataloader, num_batches)
        else:
            dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
            if N_test is not None:
                cut_loader = islice(dataloader, N_test)
            else:
                cut_loader = dataloader



        mrr = 0
        hits = {k: 0 for k in k_list}
        total_samples = 0


        E_all, entity2eval_id, filter_dict = self.prepare_evaluation(data, **kwargs)

        self.eval()
        with torch.no_grad():
            for i, batch in enumerate(cut_loader):
                h_loc, h_typ, r_idx, t_loc, t_typ = [b.to(device) for b in batch]
                batch_size = h_loc.size(0)
                total_samples += batch_size

                #получаем векторы голов и отношений
                h_eval_ids = [entity2eval_id[(local.item(), typ.item())] for local, typ in zip(h_loc, h_typ)]
                h_proj = E_all[torch.tensor(h_eval_ids, device=device)]
                r_emb = self.rel_emb(r_idx)

                #считаем скоры
                query_emb = h_proj * r_emb
                all_scores = torch.matmul(query_emb, E_all.T) # [batch_size, N_total]

                #фильтруем батчами
                mask_b = []
                mask_idx = []
                target_scores = torch.zeros(batch_size, device=device)

                for i in range(batch_size):
                    h_key = (h_loc[i].item(), h_typ[i].item())
                    t_key = (t_loc[i].item(), t_typ[i].item())
                    r = r_idx[i].item()

                    h_eval_id = entity2eval_id[h_key]
                    target_t_eval_id = entity2eval_id[t_key]

                    #сохраняем целевой скор
                    target_scores[i] = all_scores[i, target_t_eval_id]

                    #создаем маску истинных триплетов
                    true_tails = filter_dict[(h_eval_id, r)]
                    for true_t in true_tails:
                        if true_t != target_t_eval_id:
                            mask_b.append(i)
                            mask_idx.append(true_t)

                #применяем маску
                if mask_b:
                    all_scores[mask_b, mask_idx] = -1e9

                #считаем ранги
                ranks = (all_scores > target_scores.unsqueeze(1)).sum(dim=1) + 1

                #собираем метрики
                mrr += (1.0 / ranks).sum().item()
                for k in k_list:
                    hits[k] += (ranks <= k).sum().item()

                del all_scores, query_emb

            mrr = mrr / total_samples
            hits = {k: v / total_samples for k, v in hits.items()}
            formatted_hits = {k: f"{v:.4f}" for k, v in hits.items()}

            return {'MRR': mrr, 'Hits': formatted_hits}






