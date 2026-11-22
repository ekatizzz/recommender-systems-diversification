"""Оценка рекомендаций: Recall@K и NDCG@K.

Протокол
--------
1. Подбор параметров: модель на TRAIN, оценка на VALIDATION (test не используется).
2. Финал: лучшая конфигурация переобучается на TRAIN+VALIDATION (история для фильтрации тоже
   train+val), оценка на TEST один раз.
3. Для каждого пользователя из выдачи исключаются товары из его истории.
4. Целевой товар, которого нет в матрице обучения, засчитывается как промах (hit=0), но
   метрики считаются в двух вариантах и доля холодного старта указывается явно:
     'all'   - все пользователи (честная метрика, потолок Recall = 1 - cold_share);
     'known' - только пользователи, чей целевой товар известен модели.
5. В качестве ориентира считается baseline «популярное».
"""
import itertools
import json
import sys

import numpy as np
import pandas as pd

from als_model import (build_mappings, build_matrix, load_split, save_model,
                       train_als)
from config import (ALS_GRID, ALS_ITERATIONS, EXPERIMENTS_DIR, KS,
                    TUNING_METRIC)
from metrics import ndcg_at_k, recall_at_k, target_ranks


class PopularityModel:
    """Рекомендует самые популярные в train товары (с фильтрацией уже виденных).
    Интерфейс совпадает с implicit-моделью"""

    def __init__(self, user_item_matrix):
        counts = np.asarray(user_item_matrix.astype(bool).sum(axis=0)).ravel()
        self.order = np.argsort(-counts, kind="stable")

    def recommend(self, userids, user_items, N=10, filter_already_liked_items=True):
        out = np.full((len(userids), N), -1, dtype=np.int64)
        for row in range(len(userids)):
            seen = set(user_items[row].indices)
            picked = [i for i in self.order[: N + len(seen)] if i not in seen][:N]
            out[row, : len(picked)] = picked
        return out, None


def get_recommendations(model, user_items, n, batch_size=2000):
    """Top-n индексов товаров для всех пользователей (строки матрицы user_items)"""
    n_users = user_items.shape[0]
    recs = np.empty((n_users, n), dtype=np.int64)
    for start in range(0, n_users, batch_size):
        users = np.arange(start, min(start + batch_size, n_users))
        ids, _ = model.recommend(users, user_items[users], N=n, filter_already_liked_items=True)
        recs[start:start + len(users)] = ids
    return recs


def target_indices(targets_df, mappings):
    """Индекс целевого товара каждого пользователя (в порядке mappings['user_ids']); -1 - товар неизвестен"""
    t = targets_df.set_index("user_id")["item_id"].reindex(mappings["user_ids"])
    assert t.notna().all(), "у части пользователей нет целевой записи"
    return t.map(mappings["item_to_idx"]).fillna(-1).astype(int).to_numpy()


def compute_metrics(recs, targets, ks=KS):
    ranks = target_ranks(recs, targets)
    known = targets >= 0
    res = {"n_users": int(len(targets)), "cold_share": float(1 - known.mean())}
    for k in ks:
        res[f"recall@{k}"] = recall_at_k(ranks, k)
        res[f"ndcg@{k}"] = ndcg_at_k(ranks, k)
        res[f"recall@{k}_known"] = recall_at_k(ranks[known], k)
        res[f"ndcg@{k}_known"] = ndcg_at_k(ranks[known], k)
    return res


def evaluate_model(model, user_items, targets, ks=KS):
    return compute_metrics(get_recommendations(model, user_items, max(ks)), targets, ks)


def tune(train, val):
    maps = build_mappings(train)
    targets = target_indices(val, maps)
    rows = []
    keys = list(ALS_GRID)
    for combo in itertools.product(*ALS_GRID.values()):
        p = dict(zip(keys, combo))
        mat = build_matrix(train, maps, alpha=p["alpha"])
        model = train_als(mat, factors=p["factors"], regularization=p["regularization"])
        m = evaluate_model(model, mat, targets)
        rows.append({**p, **m})
        print(p, f"{TUNING_METRIC}={m[TUNING_METRIC]:.4f}", flush=True)
    res = pd.DataFrame(rows).sort_values(TUNING_METRIC, ascending=False)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(EXPERIMENTS_DIR / "tuning_results_validation.csv", index=False)
    best = {k: float(res.iloc[0][k]) for k in keys}
    best["factors"] = int(best["factors"])
    return best, res


def final_test(train, val, test, best):
    trainval = pd.concat([train, val], ignore_index=True)
    maps = build_mappings(trainval)
    mat = build_matrix(trainval, maps, alpha=best["alpha"])
    targets = target_indices(test, maps)
    model = train_als(mat, factors=best["factors"], regularization=best["regularization"])
    res = {"als": evaluate_model(model, mat, targets),
           "popularity": evaluate_model(PopularityModel(mat), mat, targets)}
    save_model(model, maps, "als_final")
    # сохраняем рекомендации (original item_id) для последующих экспериментов с диверсификацией
    recs = get_recommendations(model, mat, max(KS))
    np.savez_compressed(EXPERIMENTS_DIR / "als_top_recs_test.npz",
                        user_ids=maps["user_ids"], item_ids=maps["item_ids"][recs])
    return res


def main():
    train, val, test = load_split()
    print("== Baseline «популярное» на validation ==")
    maps = build_mappings(train)
    mat = build_matrix(train, maps)
    print(json.dumps(evaluate_model(PopularityModel(mat), mat, target_indices(val, maps)), indent=2))

    print("== Подбор параметров ALS на validation ==")
    best, table = tune(train, val)
    print("Лучшие параметры:", best)
    print(table.head(5).to_string(index=False))

    print("== Финальная оценка на test (train+val -> test) ==")
    res = final_test(train, val, test, best)
    out = {"best_params": best, "iterations": ALS_ITERATIONS, "test": res}
    with open(EXPERIMENTS_DIR / "final_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
