import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """Load time series data from a CSV file."""
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    return df


def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split dataframe into features (X) and target (y)."""
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y
