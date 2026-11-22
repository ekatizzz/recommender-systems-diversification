"""Подготовка данных: фильтрация, детерминированное временное разбиение train/val/test"""
import json

import pandas as pd

from config import (MIN_ITEM_INTERACTIONS, MIN_RATING, MIN_USER_INTERACTIONS,
                    PROCESSED_DIR, RAW_PATH)

COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


def k_core_filter(df, min_user, min_item):
    """Итеративно удаляет пользователей и товары с числом взаимодействий меньше порога.
    Одного прохода недостаточно: удаление товаров уменьшает счетчики пользователей и наоборот"""
    while True:
        n_before = len(df)
        user_cnt = df.groupby("user_id")["item_id"].transform("size")
        df = df[user_cnt >= min_user]
        item_cnt = df.groupby("item_id")["user_id"].transform("size")
        df = df[item_cnt >= min_item]
        if len(df) == n_before:
            return df.copy()


def split_by_time(df):
    """Для каждого пользователя: последний элемент -> test, предпоследний -> val, остальные -> train.

    Сортировка полностью детерминирована: (user_id, timestamp, item_id).
    В Amazon timestamp имеет точность до суток, поэтому ничьи очень частые; тай-брейк по item_id
    делает результат независимым от порядка строк в исходном файле
    """
    df = df.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort").reset_index(drop=True)
    pos_from_end = df.groupby("user_id").cumcount(ascending=False)  # 0 = последний
    train = df[pos_from_end >= 2]
    val = df[pos_from_end == 1]
    test = df[pos_from_end == 0]
    return train.copy(), val.copy(), test.copy()


def tie_statistics(df):
    """Доля пользователей, у которых ничья по timestamp на границе разбиения"""
    s = df.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort").reset_index(drop=True)
    pos = s.groupby("user_id").cumcount(ascending=False).to_numpy()  # 0 = последний
    ts = s["timestamp"].to_numpy()
    last, prev, prev2 = ts[pos == 0], ts[pos == 1], ts[pos == 2]
    return {
        "share_test_ts_tied_with_val": float((last == prev).mean()),
        "share_val_ts_tied_with_last_train": float((prev == prev2).mean()),
    }


def prepare_data():
    df = pd.read_csv(RAW_PATH, names=COLUMNS)
    stats = {"raw_interactions": len(df), "raw_users": df.user_id.nunique(), "raw_items": df.item_id.nunique()}

    if MIN_RATING is not None:
        df = df[df["rating"] >= MIN_RATING]
    stats["after_rating_filter"] = len(df)

    # один пользователь - один товар: при повторах оставляем самое позднее взаимодействие
    df = df.sort_values(["user_id", "item_id", "timestamp"]).drop_duplicates(["user_id", "item_id"], keep="last")

    assert MIN_USER_INTERACTIONS >= 3, "нужно минимум 3 взаимодействия: train + val + test"
    df = k_core_filter(df, MIN_USER_INTERACTIONS, MIN_ITEM_INTERACTIONS)
    stats.update(interactions=len(df), users=df.user_id.nunique(), items=df.item_id.nunique())
    stats.update(tie_statistics(df))

    train, val, test = split_by_time(df)
    train_items = set(train.item_id)
    stats.update(
        train=len(train), validation=len(val), test=len(test),
        cold_share_validation=float((~val.item_id.isin(train_items)).mean()),
        cold_share_test_vs_train=float((~test.item_id.isin(train_items)).mean()),
        cold_share_test_vs_train_plus_val=float((~test.item_id.isin(train_items | set(val.item_id))).mean()),
        config=dict(min_rating=MIN_RATING, min_user=MIN_USER_INTERACTIONS, min_item=MIN_ITEM_INTERACTIONS),
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("validation", val), ("test", test)):
        part[COLUMNS].to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    with open(PROCESSED_DIR / "split_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    prepare_data()
