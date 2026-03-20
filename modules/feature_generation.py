from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .index_slicing import get_cols_idx, get_slice


def get_features_df_and_targets(
    df: pd.DataFrame,
    features_ids,
    targets_ids,
    id_column: Union[str, Sequence[str]] = "id",
    date_column: Union[str, Sequence[str]] = "datetime",
    target_column: str = "target",
    rolling_agg: Tuple[str] = ("mean", "std"),
    window: int = 7,
    min_periods: int = 1
):
    df = df.copy()
    # Формируем календарные признаки из признаков времени
    df["minute"] = df[date_column].dt.minute
    df["hour"] = df[date_column].dt.hour
    df["day"] = df[date_column].dt.day
    df["month"] = df[date_column].dt.month
    df["quarter"] = df[date_column].dt.quarter
    df["year"] = df[date_column].dt.year
    date_columns = ["minute", "hour", "day", "month", "quarter", "year"]

    if rolling_agg is not None:
        for agg_func in rolling_agg:
            df[agg_func] = df.groupby(id_column)[target_column].transform(lambda x: x.rolling(window=window, min_periods=1).agg(agg_func))

    # Признаки идентификатора ряда и времени возьмем из таргетных индексов
    features_df_id = get_slice(df, (targets_ids, get_cols_idx(df, id_column)))
    features_time = get_slice(df, (targets_ids, get_cols_idx(df, date_columns)))
    if rolling_agg is not None:
        features_rolling = get_slice(df, (targets_ids, get_cols_idx(df, rolling_agg)))

    # Лаговые признаки возьмем из индексов признаков. Все доступные из истории
    features_lags = get_slice(
        df,
        (features_ids, get_cols_idx(df, target_column)),
    )

    # Объединим все признаки в один массив
    features = np.hstack([features_df_id, features_time, features_lags])
    if rolling_agg is not None:
        features =  np.hstack([features, features_rolling])
    categorical_features_idx = np.arange(
        features_df_id.shape[1] + features_time.shape[1]
    )  # Отметим категориальные признаки для CatBoost

    # Иначе cb.Pool не работает
    features_obj = features.astype(object)
    for j in categorical_features_idx:
        features_obj[:, j] = features_obj[:, j].astype(str)

    # Сформируем таргеты
    targets = get_slice(df, (targets_ids, get_cols_idx(df, target_column)))
    return features_obj, targets, categorical_features_idx
