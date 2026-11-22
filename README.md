# Рекомендательная система на Amazon Beauty (2014)

Курсовая работа: «Исследование методов разнообразия выдачи в рекомендательных системах»
Разработчик: Афанасьева Екатерина

## Данные
Amazon Reviews 2014, категория Beauty, файл `ratings_Beauty.csv` (без заголовка:
`user_id, item_id, rating, timestamp`):
https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv

Положите файл в `data/raw/ratings_Beauty.csv`.

## Запуск
Python 3.12 (на других версиях не проверялось). Команды из корня проекта:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q          # ожидается: 5 passed
cd src
python data_preparation.py         # фильтрация + разбиение -> data/processed/
python evaluate.py                 # baseline, подбор на validation, финал на test -> experiments/
```

Если `implicit` не устанавливается (бывает на Mac с чипом Apple), код автоматически
использует собственную реализацию `src/als_numpy.py`; результаты при этом немного
отличаются от запуска с `implicit`.

## Результаты (Test, 22 363 пользователя)
| Модель | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 |
| --- | --- | --- | --- | --- |
| ALS (factors=64, λ=0.01, α=40) | 0.0564 | 0.0287 | 0.1458 | 0.0481 |
| Популярное | 0.0137 | 0.0061 | 0.0372 | 0.0111 |

Полные метрики: `experiments/final_test_metrics.json`.

## Протокол
- **Разбиение:** для каждого пользователя история сортируется по `(timestamp, item_id)`;
  последний элемент – test, предпоследний – validation, остальные – train.
- **Тай-брейк:** timestamp в Amazon указан с точностью до суток, поэтому у 43–48% пользователей
  на границе разбиения есть ничьи. Без явного правила результат зависел бы от порядка строк в файле.
  При равном timestamp последним считается товар с большим `item_id`.
- **Фильтрация** (`src/config.py`): итеративный 5-core по пользователям и товарам, все оценки считаются
  взаимодействием (implicit feedback). Облегченный вариант (оценки ≥ 4, пользователи ≥ 3) воспроизводится
  параметрами `MIN_RATING=4.0, MIN_USER_INTERACTIONS=3, MIN_ITEM_INTERACTIONS=1`.
- **ALS:** `implicit`, матрица пользователи×товары, значение = confidence `alpha`.
- **Подбор параметров** (factors, regularization, alpha) – только на validation, по `ndcg@10`.
- **Финал:** лучшая конфигурация переобучается на train+validation, оценивается на test один раз.
- **Метрики:** Recall@K и NDCG@K (K=10,20,50), формулы в `src/metrics.py`. Из выдачи убираются уже виденные
  товары. Неизвестный модели целевой товар = промах, но дополнительно считаются метрики `_known` и
  указывается `cold_share`.
- **Baseline «популярное»** считается рядом как нижний ориентир.
