"""Собственная реализация ALS для implicit feedback (Hu, Koren, Volinsky, 2008) на numpy/scipy.

Модель: p_ui = 1 для наблюдаемых пар, уверенность c_ui = alpha (в матрице хранится alpha),
для ненаблюдаемых пар c_ui = 1, p_ui = 0. Минимизируется
    sum_{u,i} c_ui (p_ui - x_u^T y_i)^2 + reg (|X|^2 + |Y|^2).
При фиксированных Y оптимум для пользователя u:
    x_u = (Y^T Y + Y^T (C_u - I) Y + reg I)^{-1} Y^T C_u p_u,
причем Y^T (C_u - I) Y считается только по товарам пользователя -> шаг линеен по числу взаимодействий.
Аналогично для товаров. Интерфейс (fit / recommend / user_factors / item_factors) совместим с implicit.
Используется как запасной вариант, если библиотека implicit не установлена, и для проверки ALS
"""
import numpy as np


def _solve_side(M, Y, reg, chunk=128):
    """M: CSR (строки = обновляемые объекты), Y: фиксированные факторы. Возвращает новые факторы строк M"""
    n, f = M.shape[0], Y.shape[1]
    X = np.zeros((n, f), dtype=np.float32)
    YtY = (Y.T @ Y).astype(np.float64)
    eye = reg * np.eye(f)
    nnz = np.diff(M.indptr)
    order = np.argsort(nnz)                      # похожие по длине строки в одном батче -> мало паддинга
    for s in range(0, n, chunk):
        rows = order[s:s + chunk]
        L = int(nnz[rows].max())
        if L == 0:
            continue
        idx = np.zeros((len(rows), L), dtype=np.int64)
        conf = np.zeros((len(rows), L), dtype=np.float64)
        for k, r in enumerate(rows):
            a, b = M.indptr[r], M.indptr[r + 1]
            idx[k, : b - a] = M.indices[a:b]
            conf[k, : b - a] = M.data[a:b]
        Yi = Y[idx].astype(np.float64)            # (B, L, f); паддинг имеет conf=0 и не влияет
        W = (conf - 1.0)[..., None] * Yi
        A = YtY + np.matmul(W.transpose(0, 2, 1), Yi) + eye      # Y^T (C_u - I) Y + Y^T Y + reg I
        rhs = np.matmul(conf[:, None, :], Yi)[:, 0, :]          # Y^T C_u p_u (p=1)
        X[rows] = np.linalg.solve(A, rhs[..., None])[..., 0]
    return X


class NumpyALS:
    def __init__(self, factors=64, regularization=0.01, iterations=15, random_state=42):
        self.factors, self.regularization, self.iterations = factors, regularization, iterations
        self.random_state = random_state

    def fit(self, user_items, show_progress=False):
        C = user_items.tocsr().astype(np.float32)
        Ct = C.T.tocsr()
        rng = np.random.RandomState(self.random_state)
        self.user_factors = (rng.normal(size=(C.shape[0], self.factors)) * 0.01).astype(np.float32)
        self.item_factors = (rng.normal(size=(C.shape[1], self.factors)) * 0.01).astype(np.float32)
        for _ in range(self.iterations):
            self.user_factors = _solve_side(C, self.item_factors, self.regularization)
            self.item_factors = _solve_side(Ct, self.user_factors, self.regularization)
        return self

    def recommend(self, userids, user_items, N=10, filter_already_liked_items=True):
        scores = self.user_factors[userids] @ self.item_factors.T
        if filter_already_liked_items:
            r, c = user_items.nonzero()
            scores[r, c] = -np.inf
        top = np.argpartition(-scores, N - 1, axis=1)[:, :N]
        part = np.take_along_axis(scores, top, 1)
        o = np.argsort(-part, axis=1)
        return np.take_along_axis(top, o, 1), np.take_along_axis(part, o, 1)
