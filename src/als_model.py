"""ALS (implicit feedback): построение матрицы, обучение, сохранение"""
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from config import ALS_ITERATIONS, MODELS_DIR, PROCESSED_DIR, SEED


def load_split():
    read = lambda n: pd.read_csv(PROCESSED_DIR / f"{n}.csv")
    return read("train"), read("validation"), read("test")


def build_mappings(df):
    """Индексы строятся по отсортированным уникальным ID -> воспроизводимо"""
    user_ids = np.sort(df["user_id"].unique())
    item_ids = np.sort(df["item_id"].unique())
    return {
        "user_ids": user_ids, "item_ids": item_ids,
        "user_to_idx": {u: i for i, u in enumerate(user_ids)},
        "item_to_idx": {it: i for i, it in enumerate(item_ids)},
    }


def build_matrix(df, mappings, alpha=1.0):
    """Матрица пользователи x товары (то, что ожидает implicit >= 0.5).
    Значение = уверенность (confidence) alpha для наблюдаемого взаимодействия.
    Взаимодействия с товарами, которых нет в mappings, пропускаются"""
    u = df["user_id"].map(mappings["user_to_idx"])
    i = df["item_id"].map(mappings["item_to_idx"])
    ok = u.notna() & i.notna()
    u, i = u[ok].astype(int).to_numpy(), i[ok].astype(int).to_numpy()
    shape = (len(mappings["user_ids"]), len(mappings["item_ids"]))
    return csr_matrix((np.full(len(u), alpha, dtype=np.float32), (u, i)), shape=shape)


def train_als(user_item_matrix, factors=64, regularization=0.01, iterations=ALS_ITERATIONS, backend="auto"):
    """backend: 'implicit' (быстрая библиотека), 'numpy' (собственная реализация) или 'auto'
    (implicit, если установлена, иначе numpy)"""
    if backend in ("auto", "implicit"):
        try:
            from implicit.als import AlternatingLeastSquares
            model = AlternatingLeastSquares(
                factors=factors, regularization=regularization, iterations=iterations,
                random_state=SEED, use_gpu=False,  # GPU-версия даёт невоспроизводимые результаты
            )
            model.fit(user_item_matrix, show_progress=False)
            return model
        except ImportError:
            if backend == "implicit":
                raise
            print("implicit не установлен -> используется собственная реализация (als_numpy.py)")
    from als_numpy import NumpyALS
    return NumpyALS(factors, regularization, iterations, SEED).fit(user_item_matrix)


def save_model(model, mappings, name):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / f"{name}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODELS_DIR / f"{name}_id_mappings.pkl", "wb") as f:
        pickle.dump(mappings, f)


if __name__ == "__main__":
    # быстрый запуск с параметрами по умолчанию; полноценный протокол в evaluate.py
    train, _, _ = load_split()
    maps = build_mappings(train)
    mat = build_matrix(train, maps)
    print("Матрица:", mat.shape, "nnz:", mat.nnz)
    save_model(train_als(mat), maps, "als_default")
