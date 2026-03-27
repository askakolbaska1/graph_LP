import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.nn import DistMult
from torch.optim import Adam
from tqdm import tqdm

from src.base import BaseModel
from src.data_loader import df_to_triplets_geometric


class CustomDistMult(DistMult, BaseModel):
    @staticmethod
    def preprocess(**kwargs):
        train_triplets, val_triplets, test_triplets, filtered_dict, rawid2id, pred2id = df_to_triplets_geometric(
            path=kwargs.get('input_path'),
            heads=kwargs.get('heads_col'),
            predicates=kwargs.get('predicates_col'),
            tails=kwargs.get('tails_col'),
            train_size=kwargs.get('train_size', 0.8),
            test_and_val_size=kwargs.get('test_and_val_size', 0.2),
            device=kwargs.get('device'),
        )


        data = {
            'train_triplets': train_triplets,
            'val_triplets': val_triplets,
            'test_triplets': test_triplets,
            'filtered_dict': filtered_dict,
            'rawid2id': rawid2id,
            'pred2id': pred2id,
            'num_nodes': len(rawid2id),
            'num_relations': len(pred2id),
        }


        return data



    def test(self, data: dict, val = False, **kwargs):
        filtered_dict = data['filtered_dict']

        batch_size = kwargs.get('batch_size', 128)
        k_list = kwargs.get('k_list', [1, 5, 10, 50])
        N_val = kwargs.get('N_val', None)
        N_test = kwargs.get('N_test', None)

        self.eval()

        if val:
            head_index = data['val_triplets'][:, 0]
            rel_type = data['val_triplets'][:, 1]
            tail_index = data['val_triplets'][:, 2]
            num_queries = head_index.numel()
            if N_val is not None:
                num_queries = N_val
        else:
            head_index = data['test_triplets'][:, 0]
            rel_type = data['test_triplets'][:, 1]
            tail_index = data['test_triplets'][:, 2]
            num_queries = head_index.numel()
            if N_test is not None:
                num_queries = N_test



        ranks = torch.zeros(num_queries, device=head_index.device)

        #получаем эмбеддинги ВСЕХ узлов графа один раз
        #E_all размерности [num_nodes, embedding_dim]
        with torch.no_grad():
            E_all = self.node_emb.weight.data

            #разбиваем сами запросы (h, r) на батчи, а не хвосты
            for start_idx in tqdm(range(0, num_queries, batch_size)):
                end_idx = min(start_idx + batch_size, num_queries)

                h_batch = head_index[start_idx:end_idx]
                r_batch = rel_type[start_idx:end_idx]
                t_batch = tail_index[start_idx:end_idx] #целевые хвосты

                #получаем эмбеддинги для текущего батча запросов
                h_emb = self.node_emb.weight.data[h_batch]
                r_emb = self.rel_emb.weight.data[r_batch]

                #query_emb: [current_batch_size, embedding_dim]
                query_emb = h_emb * r_emb

                #умножаем батч запросов на транспонированную матрицу всех узлов
                #scores: [current_batch_size, num_nodes]
                scores = torch.matmul(query_emb, E_all.T)

                #фильтрация и маскирование для батча
                if filtered_dict is not None:
                    mask_batch = []
                    mask_idx = []
                    for i_in_batch, (h, r, t) in enumerate(zip(h_batch, r_batch, t_batch)):
                        h, r, t = int(h), int(r), int(t)
                        true_tails = filtered_dict.get((h, r), [])

                        for true_t in true_tails:
                            if true_t != t:
                                mask_batch.append(i_in_batch)
                                mask_idx.append(true_t)

                    if mask_batch:
                        scores[mask_batch, mask_idx] = -float('inf')

                #выделяем целевые скоры [current_batch_size, 1]
                target_scores = scores[torch.arange(len(t_batch)), t_batch].unsqueeze(1)

                #cчитаем ранги для всего батча
                batch_ranks = (scores > target_scores).sum(dim=1) + 1
                ranks[start_idx:end_idx] = batch_ranks

        #итоговый подсчет метрик
        mrr = (1.0 / ranks).float().mean().item()

        hits_at_k = {}
        for k in k_list:
            hits_at_k[k] = (ranks <= k).float().mean().item()

        formatted_hits = {k: f"{v:.4f}" for k, v in hits_at_k.items()}

        return {'MRR': mrr, 'Hits': formatted_hits}




    def get_sns_negatives(self, all_embs, pos_indices, n1, n2):
        """Вспомогательная функция для поиска сложных негативов"""
        # Сэмплируем N1 кандидатов для каждого триплета в батче сразу
        # shape: (num_negatives, n1)
        cand_indices = torch.randint(0, self.num_nodes, (pos_indices.size(0), n1), device=self.node_emb.weight.device)

        # Получаем эмбеддинги: позитивных сущностей и кандидатов
        pos_embs = all_embs[pos_indices].unsqueeze(1)    # (num_negatives, 1, dim)
        cand_embs = all_embs[cand_indices]                # (num_negatives, n1, dim)

        # Считаем расстояние d = ||pos - cand||
        dist = torch.norm(pos_embs - cand_embs, p=2, dim=-1) # (num_negatives, n1)

        # Считаем вероятности P = softmax(1/d)
        probs = torch.softmax(1.0 / (dist + 1e-9), dim=1)

        # Выбираем N2 лучших (самых близких) из N1
        _, top_n2_loc_idx = torch.topk(probs, k=n2, dim=1)

        # Из N2 выбираем по 1 случайному индексу для каждого примера (Exploration)
        rand_selector = torch.randint(0, n2, (pos_indices.size(0),), device=self.node_emb.weight.device)

        # Собираем финальные индексы
        final_loc_idx = top_n2_loc_idx[torch.arange(pos_indices.size(0)), rand_selector]
        return cand_indices[torch.arange(pos_indices.size(0)), final_loc_idx]



    @torch.no_grad()
    def sns_sample(
        self,
        head_index: torch.Tensor,
        rel_type: torch.Tensor,
        tail_index: torch.Tensor,
        n1: int = 50,  # Размер начального пула кандидатов
        n2: int = 5   # Размер пула "сложных" негативов
        ):
        # 1. Получаем текущие эмбеддинги всех сущностей
        # Предполагаем, что они лежат в self.node_emb.weight
        all_embs = self.node_emb.weight
        batch_size = head_index.numel()
        num_negatives = batch_size // 2

        # Клонируем индексы для модификации
        new_head = head_index.clone()
        new_tail = tail_index.clone()


        # 2. Коррептируем головы (первая половина батча)
        new_head[:num_negatives] = self.get_sns_negatives(all_embs, head_index[:num_negatives], n1, n2)

        # 3. Коррептируем хвосты (вторая половина батча)
        new_tail[num_negatives:] = self.get_sns_negatives(all_embs, tail_index[num_negatives:], n1, n2)

        return new_head, rel_type, new_tail



    def loss(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        negative_sampler: str = 'sns'
    ) -> Tensor:

        pos_score = self(head_index, rel_type, tail_index)
        if negative_sampler == 'sns':
            neg_score = self(*self.sns_sample(head_index, rel_type, tail_index))
        elif negative_sampler == 'random':
            neg_score = self(*self.random_sample(head_index, rel_type, tail_index))
        else:
            raise ValueError

        return F.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=self.margin,
        )



    def train_model(self, data: dict, **kwargs):

        batch_size = kwargs.get('batch_size', 1024)
        lr = kwargs.get('lr', 0.001)

        optimizer = Adam(self.parameters(), lr=lr)

        train_loader = DataLoader(
            data['train_triplets'],
            batch_size=batch_size,
            shuffle=True
        )

        self.train()
        total_loss = 0

        for batch in train_loader:
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            optimizer.zero_grad()
            loss = self.loss(h, r, t, negative_sampler=kwargs.get('sampler', 'sns'))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        return avg_loss



