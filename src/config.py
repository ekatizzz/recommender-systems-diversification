"""Единая конфигурация эксперимента"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "ratings_Beauty.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

# ---------- Подготовка данных ----------
# None -> используем все оценки как факт взаимодействия (implicit feedback, как в задании)
# 4.0  -> оставить только оценки >= 4 (прежняя конфигурация проекта)
MIN_RATING = None

# Итеративная фильтрация (k-core): повторяется, пока выборка не стабилизируется
# Стандартный 5-core для Amazon Beauty: 5 и 5
# MIN_USER_INTERACTIONS должен быть >= 3 (train + val + test)
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5

# ---------- Оценка ----------
KS = (10, 20, 50)          # значения K для Recall@K и NDCG@K
TUNING_METRIC = "ndcg@10"  # по какой метрике на validation выбираем параметры
SEED = 42

# ---------- ALS ----------
ALS_ITERATIONS = 15
ALS_GRID = {
    "factors": [32, 64],
    "regularization": [0.01, 0.1, 1.0],
    "alpha": [10.0, 40.0],   # вес наблюдаемых пар (confidence); alpha=1 исключен как вырожденный случай
}
