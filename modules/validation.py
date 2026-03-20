from typing import Tuple

import numpy as np
import pandas as pd


def expanding_window_validation(
    data: pd.DataFrame,
    model,
    horizon: int,
    history: int,
    start_train_size: int,
    step_size: int,
    id_col: str = "ts_id",
    timestamp_col: str = "timestamp",
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Валидация с расширяющимся окном обучения.

    Args:
    - data: DataFrame с временными рядами.
    - model: не обученная модель для прогнозирования.
    - horizon: длина выходного окна (горизонт прогнозирования).
    - history: длина входного окна (история).
    - start_train_size: начальный размер обучающего набора.
    - step_size: шаг расширения окна обучения.
    - id_col: название столбца с идентификатором ряда.
    - timestamp_col: название столбца с временной меткой.
    - value_col: название столбца с значением ряда.

    Returns: DataFrame с истинными и предсказанными значениями c столбцами
        ['ts_id', 'fold', 'timestamp', 'true_value', 'predicted_value'].

    """
    res_df_list = []

    # Так как ряды выровненные, нам не нужно обрабатывать каждый ряд отдельно.
    # Будем разрезать ряды по времени
    unique_timestamps = data[timestamp_col].sort_values().unique()
    n_timestamps = len(unique_timestamps)

    # Трейн начинается с 0 и идет до start_train_size
    train_start_idx = min(unique_timestamps)
    train_end_idx = train_start_idx + pd.DateOffset(days=start_train_size)

    # Валидация включает history точек из конца трейна и идет до horizon точек после трейна
    val_start_idx = train_end_idx - pd.DateOffset(days=history) # -1?
    val_end_idx = train_end_idx + pd.DateOffset(days=horizon) # +1?

    # Тест включает history точек из конца трейна + валидации и идет до horizon точек после валидации
    test_start_idx = val_end_idx - pd.DateOffset(days=history)
    test_end_idx = val_end_idx + pd.DateOffset(days=horizon)

    # while test_end_idx <= n_timestamps:
    while test_end_idx <= max(unique_timestamps):
        # Чтобы не было лика в test, маскируем то, что мы не знаем на момент предсказания
        data_masked = data.copy()
        data_masked[value_col] = data_masked[value_col].where(
            # data_masked[timestamp_col] < unique_timestamps[test_start_idx + history], np.nan
            data_masked[timestamp_col] < (test_start_idx + pd.DateOffset(days=history)), np.nan
        )

        train_mask = (data_masked[timestamp_col] >= train_start_idx) & (data_masked[timestamp_col] < train_end_idx)
        val_mask = (data_masked[timestamp_col] >= val_start_idx) & (data_masked[timestamp_col] < val_end_idx)
        test_mask = (data_masked[timestamp_col] >= test_start_idx) & (data_masked[timestamp_col] < test_end_idx)

        train_data = data[train_mask]
        val_data = data[val_mask]
        test_data = data_masked[test_mask]

        # Обучаем модель на трейне и валидации и делаем прогноз на тесте

        model.fit(train_data, val_data, id_col=id_col, timestamp_col=timestamp_col, value_col=value_col)
        predictions = model.predict(test_data, id_col=id_col, timestamp_col=timestamp_col, value_col=value_col)
        print(f"Nan count in predictions {predictions['predicted_value'].isna().sum()}")

        # Восстанавливаем истинные значения для теста
        test_data_unmasked = data[test_mask]

        # Обрезаем историю из валидации от теста
        # test_data = test_data[test_data[timestamp_col] >= unique_timestamps[val_end_idx]]
        test_data = test_data[test_data[timestamp_col] >= val_end_idx]
        test_data_unmasked = test_data_unmasked[
            test_data_unmasked[timestamp_col] >= val_end_idx
        ]
        print(test_data[id_col].shape)
        print(test_data.shape)
        print(test_data_unmasked.shape)
        print(np.repeat(len(res_df_list), len(test_data)).shape)
        print(predictions["predicted_value"].shape)

        # Собираем истинные и предсказанные значения
        res_df = pd.DataFrame(
            {
                id_col: test_data[id_col].values,
                "fold": np.repeat(len(res_df_list), len(test_data)),
                timestamp_col: test_data[timestamp_col].values,
                "true_value": test_data_unmasked[value_col].values,
                "predicted_value": predictions["predicted_value"].values,
            }
        )
        res_df_list.append(res_df)

        # Расширяем окно
        train_end_idx = train_end_idx + pd.DateOffset(days=step_size)
        val_start_idx = val_start_idx + pd.DateOffset(days=step_size)
        val_end_idx = val_end_idx + pd.DateOffset(days=step_size)
        test_start_idx = test_start_idx + pd.DateOffset(days=step_size)
        test_end_idx = test_end_idx + pd.DateOffset(days=step_size)

    res_df = pd.concat(res_df_list).reset_index(drop=True)

    return res_df