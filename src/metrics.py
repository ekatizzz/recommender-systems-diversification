"""Метрики Recall@K и NDCG@K для случая ОДНОГО целевого товара на пользователя.

rank - позиция целевого товара в выдаче, считая с 0; если товара нет в выдаче (или он вообще не
известен модели) rank = -1.

Recall@K = 1, если target в топ-K, иначе 0 (|relevant| = 1, значит Recall == HitRate).
NDCG@K   = 1 / log2(rank + 2), если rank < K, иначе 0. IDCG = 1/log2(2) = 1 (единственный
           релевантный товар в идеале стоит на первой позиции), поэтому нормировка не меняет значение.
Оба значения усредняются по пользователям
"""
import numpy as np


def target_ranks(recs, targets):
    """recs: (n_users, N) индексы товаров; targets: (n_users,) индекс цели или -1 (неизвестный товар)"""
    match = recs == targets[:, None]
    match &= (targets >= 0)[:, None]
    hit = match.any(axis=1)
    return np.where(hit, match.argmax(axis=1), -1)


def recall_at_k(ranks, k):
    return float(((ranks >= 0) & (ranks < k)).mean())


def ndcg_at_k(ranks, k):
    in_top = (ranks >= 0) & (ranks < k)
    gains = np.zeros(len(ranks))
    gains[in_top] = 1.0 / np.log2(ranks[in_top] + 2)
    return float(gains.mean())
