import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.nn import DistMult
from torch.optim import Adam
from tqdm import tqdm

from .base import BaseModel
from .data_loader import df_to_triplets_geometric


class CustomDistMult(DistMult, BaseModel):
    def preprocess(self, df, **kwargs):
        train_triplets, val_triplets, test_triplets, filtered_dict, rawid2id, pred2id = df_to_triplets_geometric(
            df=df,
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
        }


        return data



    def test(self, data: dict, **kwargs):
        """
        data: словарь из preprocess (head_index, rel_type, tail_index)
        kwargs: параметры из конфига (batch_size, k_list, sampling и т.д.)
        """

        head_index = data['test_triplets'][:, 0]
        rel_type = data['test_triplets'][:, 1]
        tail_index = data['test_triplets'][:, 2]
        filtered_dict = data['filtered_dict']

        batch_size = kwargs.get('batch_size', 1024)
        k_list = kwargs.get('k_list', [1, 5, 10, 50])
        N = kwargs.get('N', None)
        sampling = kwargs.get('sampling', False)
        num_negs = kwargs.get('num_negs', 1000)
        log = kwargs.get('log', True)

        self.eval()

        if N is not None:
            head_index = head_index[:N]
            rel_type = rel_type[:N]
            tail_index = tail_index[:N]

        arange = range(head_index.numel())
        arange = tqdm(arange) if log else arange

        mean_ranks, reciprocal_ranks, hits_at_k = [], [], {}
        for k in k_list:
            hits_at_k[k] = []

        for i in arange:
            h, r, t = head_index[i], rel_type[i], tail_index[i]
            scores = []

            if sampling:
                neg_tails = torch.randint(0, self.num_nodes, (num_negs,), device=t.device)
                tail_indices = torch.cat([t.unsqueeze(0), neg_tails])
                target_index = 0
            else:
                tail_indices = torch.arange(self.num_nodes, device=t.device)
                target_index = t

            for ts in tail_indices.split(batch_size):
                scores.append(self(h.expand_as(ts), r.expand_as(ts), ts))

            scores = torch.cat(scores)

            if filtered_dict is not None and not sampling:
                #Получаем все известные истинные хвосты для пары (h, r)
                true_tails = filtered_dict.get((int(h), int(r)), [])

                # Маскируем все известные истинные хвосты, кроме текущего целевого t
                for true_t in true_tails:
                    if true_t != int(t):
                        scores[true_t] = -float('inf') # Убираем из рейтинга


            elif filtered_dict is not None and sampling:
                raise ValueError

            rank = int((scores.argsort(descending=True) == target_index).nonzero().view(-1))

            mean_ranks.append(rank)
            reciprocal_ranks.append(1 / (rank + 1))
            for k in k_list:
                hits_at_k[k].append(rank < k)

        mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
        mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
        for k in k_list:
            hits_at_k[k] = float(torch.tensor(hits_at_k[k], dtype=torch.float).mean())

        formatted_hits = {k: f"{v:.{4}f}" for k, v in hits_at_k.items()}
        result_dict = {'MRR': mrr, 'Hits@': formatted_hits}
        return result_dict



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
        """
        data: словарь из preprocess (head_index, rel_type, tail_index)
        kwargs: параметры из конфига (batch_size, k_list, sampling и т.д.)
        """
        batch_size = kwargs.get('batch_size', 1024)
        epochs = kwargs.get('epochs', 10)
        lr = kwargs.get('lr', 0.001)

        optimizer = Adam(self.parameters(), lr=lr)

        train_loader = DataLoader(
            data['train_triplets'],
            batch_size=batch_size,
            shuffle=True
        )

        for epoch in range(epochs):
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
            print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_loss:.4f}")



