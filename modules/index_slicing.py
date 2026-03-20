from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd


def features_targets__train_idx(
    id_column: pd.Series,
    series_length: int,
    model_horizon: int,
    history_size: int,
) -> np.ndarray:
    """Создание индексов для формирования обучающей выборки (признаков и таргетов) для многомерных временных рядов.
    Args:
        id_column: Колонка с идентификаторами рядов.
        series_length: Общая длина всех рядов.
        model_horizon: Горизонт прогнозирования модели.
        history_size: Размер окна истории.

    Returns:
        Индексы для формирования признаков и таргетов.

    """
    # Тут возвращаем границы уникальних значений и в конце - индекс конца
    series_start_indices = np.append(
        np.unique(id_column.values, return_index=True)[1], series_length
    )

    features_indices = []
    targets_indices = []
    for i in range(len(series_start_indices) - 1):
        series_start = series_start_indices[i]
        series_end = series_start_indices[i + 1]

        if series_end - series_start < history_size + model_horizon:
            continue  # Пропускаем ряды, которые слишком короткие для формирования окна истории + таргета
    
        # 2d array из индексов окна, каждая след строка - индекс увелич на 1
        sliding_window = np.lib.stride_tricks.sliding_window_view(
            np.arange(series_start, series_end),
            history_size + model_horizon,
        )

        features_indices.append(sliding_window[:, :history_size])
        targets_indices.append(sliding_window[:, history_size:])

    features_indices = np.vstack(features_indices)
    targets_indices = np.vstack(targets_indices)

    return features_indices, targets_indices


def features__test_idx(
    id_column: pd.Series,
    series_length: int,
    model_horizon: int,
    history_size: int,
) -> np.ndarray:
    """Создание индексов для формирования тестовой выборки для многомерных временных рядов.
    Args:
        id_column: Колонка с идентификаторами рядов.
        series_length: Общая длина всех рядов.
        model_horizon: Горизонт прогнозирования модели.
        history_size: Размер окна истории.

    Returns:
        Индексы для формирования признаков.
        Обратите внимание, что для признаков тестовой выборки нужны только последние history индексов.

    """
    series_start_indices = np.append(
        np.unique(id_column.values, return_index=True)[1], series_length
    )

    features_indices = []
    targets_indices = []
    for i in range(len(series_start_indices) - 1):
        series_start = series_start_indices[i]
        series_end = series_start + history_size + model_horizon

        if series_end - series_start < history_size + model_horizon:
            AssertionError(f"Ряд {i} слишком короткий для формирования окна истории + таргета")

        sliding_window = np.lib.stride_tricks.sliding_window_view(
            np.arange(series_start, series_end),
            history_size + model_horizon,
        )

        features_indices.append(sliding_window[:, :history_size])
        targets_indices.append(sliding_window[:, history_size:])

    features_indices = np.vstack(features_indices)
    targets_indices = np.vstack(targets_indices)

    return features_indices, targets_indices


def get_slice(data: pd.DataFrame, k: Tuple[np.ndarray]) -> np.ndarray:
    """Получение среза из DataFrame по индексам строк и колонок.

    Args:
        data: Исходный DataFrame.
        k: Кортеж из двух элементов:
            - Массив индексов строк.
            - Массив индексов колонок или None (если нужны все колонки).

    Returns:
        Массив значений из DataFrame по заданным индексам.

    """
    rows, cols = k
    if cols is None:
        new_data = data.values[rows]
    else:
        new_data = data.iloc[:, cols].values[rows]

    if new_data.ndim == 3:
        new_data = new_data.reshape(new_data.shape[0], -1)

    return new_data


def get_cols_idx(data: pd.DataFrame, columns: Union[str, Sequence[str]]) -> Union[int, np.ndarray]:
    """Получение индексов колонок по их названиям.

    Args:
        data: DataFrame с колонками.
        columns: Название колонки или список названий колонок.

    Returns:
        Индекс колонки или массив индексов колонок.

    """
    if type(columns) is str:
        idx = data.columns.get_loc(columns)
    else:
        idx = data.columns.get_indexer(columns)
    return idx
