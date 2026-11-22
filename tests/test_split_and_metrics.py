"""Запуск: python -m pytest tests -q   (из корня проекта)  или  python tests/test_split_and_metrics.py"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_preparation import k_core_filter, split_by_time  # noqa: E402
from metrics import ndcg_at_k, recall_at_k, target_ranks  # noqa: E402


def _toy():
    rows = [("u1", "B", 1), ("u1", "C", 2), ("u1", "A", 3), ("u1", "D", 4)]  # пример из задания
    rows += [("u2", "x", 5), ("u2", "y", 5), ("u2", "z", 5)]                 # все ничьи
    return pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"]).assign(rating=5.0)


def test_example_from_task():
    tr, va, te = split_by_time(_toy())
    u = lambda d: d[d.user_id == "u1"].item_id.tolist()
    assert u(tr) == ["B", "C"] and u(va) == ["A"] and u(te) == ["D"]


def test_split_independent_of_row_order():
    df = _toy()
    ref = [d.sort_values(["user_id", "item_id"]).values.tolist() for d in split_by_time(df)]
    for seed in range(10):
        shuffled = df.sample(frac=1, random_state=seed)
        got = [d.sort_values(["user_id", "item_id"]).values.tolist() for d in split_by_time(shuffled)]
        assert got == ref


def test_split_properties():
    tr, va, te = split_by_time(_toy())
    assert len(va) == len(te) == 2 and set(va.user_id) == set(te.user_id) == {"u1", "u2"}
    pairs = lambda d: set(zip(d.user_id, d.item_id))
    assert not (pairs(tr) & pairs(va)) and not (pairs(tr) & pairs(te)) and not (pairs(va) & pairs(te))


def test_k_core_is_iterative():
    # u3 имеет 3 взаимодействия, но товар "rare" встречается один раз -> после его удаления u3 падает ниже порога
    rows = [(u, i, 1, 5.0) for u in ("a", "b", "c") for i in ("p", "q", "r")]
    rows += [("d", "p", 1, 5.0), ("d", "q", 1, 5.0), ("d", "rare", 1, 5.0)]
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp", "rating"])
    out = k_core_filter(df, min_user=3, min_item=2)
    assert "d" not in set(out.user_id) and "rare" not in set(out.item_id)


def test_metrics_hand_computed():
    recs = np.array([[5, 7, 9], [1, 2, 3], [4, 5, 6], [1, 2, 3]])
    targets = np.array([5, 3, 9, -1])           # ранги: 0, 2, нет, неизвестный товар
    ranks = target_ranks(recs, targets)
    assert ranks.tolist() == [0, 2, -1, -1]
    assert recall_at_k(ranks, 1) == 0.25 and recall_at_k(ranks, 3) == 0.5
    assert math.isclose(ndcg_at_k(ranks, 3), (1 + 1 / math.log2(4)) / 4)
    assert math.isclose(ndcg_at_k(ranks, 2), 1 / 4)   # хит на позиции 2 в топ-2 не попал


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("OK", name)
