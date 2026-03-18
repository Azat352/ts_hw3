from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator


def build_model() -> BaseEstimator:
    """Return a default regression model."""
    return LinearRegression()
