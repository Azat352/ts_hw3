import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from src.data import load_data, preprocess
from src.model import build_model


def train(data_path: str, test_size: float = 0.2) -> None:
    df = load_data(data_path)
    X, y = preprocess(df)

    # shuffle=False preserves temporal order and prevents data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    model = build_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Test RMSE: {rmse:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train time series model")
    parser.add_argument("data_path", help="Path to CSV data file")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    train(args.data_path, args.test_size)
