from collections import defaultdict

import torch

@torch.no_grad()
def ranking_metrics_filtered(model, data, all_pos_edge_index, k_list=[1, 3, 10], N=1000,
                             use_sampling=False, num_negatives=1000):
    model.eval()

    z = model(data.x, data.edge_index, data.node_type) #получаем предсказание модели
    num_nodes = z.size(0)

    # 1. Берем только позитивные (истинные) ребра из сплита
    pos_mask = data.edge_label == 1 #маска на истинные ребра
    all_pos_edges = data.edge_label_index[:, pos_mask] #все истинные ребра
    random_indices = torch.randperm(all_pos_edges.size(1))[:N]
    pos_edge_index = all_pos_edges[:, random_indices] #берем рандомные N для оценки
    num_edges = pos_edge_index.size(1) #количество оцениваемых ребер

    #создаем словарь всех истинных связей {src: set(dsts)}
    known_links = defaultdict(set)
    for i in range(all_pos_edge_index.size(1)):
        u = all_pos_edge_index[0, i].item()
        v = all_pos_edge_index[1, i].item()
        known_links[u].add(v)

    ranks = []

    for i in range(num_edges): #итерируемся по ребрам для оценки
        src = pos_edge_index[0, i].item()
        dst = pos_edge_index[1, i].item()
        #все истинные триплеты для head
        known_dsts_set = known_links[src]

        if use_sampling: #оценка с семплированием
            #случайно выбираем num_negatives узлов
            neg_dsts = torch.randint(0, num_nodes, (num_negatives,), device=z.device)

            #на 0-й позиции истинный dst, затем случайные негативы
            candidates = torch.cat([torch.tensor([dst], device=z.device), neg_dsts])

            #считаем скоры для этих кандидатов
            scores = (z[src] * z[candidates]).sum(dim=1)

            #если среди случайно выбранных негативов попались известные истинные
            #соседи, мы должны занулить скор у них
            mask_idx = [j for j, c in enumerate(candidates.tolist()) if c in known_dsts_set and j != 0]
            if mask_idx:
                scores[torch.tensor(mask_idx, dtype=torch.long, device=z.device)] = -float('inf')

            #истинный узел всегда находится на индексе 0
            #находим ранк правильного узла
            _, sorted_idx = torch.sort(scores, descending=True)
            rank = (sorted_idx == 0).nonzero(as_tuple=True)[0].item() + 1

        else:
            #оценка без семплирования
            scores = (z[src] * z).sum(dim=1)

            known_dsts = list(known_dsts_set)
            if dst in known_dsts:
                known_dsts.remove(dst)
            #зануляем скор у известных
            if len(known_dsts) > 0:
                scores[torch.tensor(known_dsts, dtype=torch.long, device=z.device)] = -float('inf')
            # находим ранк правильного узла
            _, sorted_idx = torch.sort(scores, descending=True)
            rank = (sorted_idx == dst).nonzero(as_tuple=True)[0].item() + 1

        ranks.append(rank)

    ranks = torch.tensor(ranks, dtype=torch.float)
    #считаем mrr
    mrr = torch.mean(1.0 / ranks).item()
    #считаем Hits
    hits = {}
    for k in k_list:
        hits_k = torch.mean((ranks <= k).float()).item()
        hits[k] = hits_k

    return mrr, hits