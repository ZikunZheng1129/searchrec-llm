"""Simple NumPy feature ranker used as a local fallback backend."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.ranking.dataset import apply_feature_standardizer


class NumpyLinearRanker:
    """Deterministic logistic regression trained with NumPy gradient descent."""

    backend = "numpy_linear"

    def __init__(
        self,
        learning_rate: float = 0.05,
        num_epochs: int = 100,
        regularization: float = 0.001,
        seed: int = 42,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.num_epochs = int(num_epochs)
        self.regularization = float(regularization)
        self.seed = int(seed)
        self.feature_columns: list[str] = []
        self.standardizer: dict[str, dict[str, float] | list[str]] = {}
        self.weights = np.zeros(0, dtype=float)
        self.bias = 0.0

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _fit_standardizer(self, train_df: pd.DataFrame) -> None:
        means = train_df[self.feature_columns].astype(float).mean(axis=0)
        stds = (
            train_df[self.feature_columns].astype(float).std(axis=0).replace(0.0, 1.0).fillna(1.0)
        )
        self.standardizer = {
            "feature_columns": list(self.feature_columns),
            "mean": means.to_dict(),
            "std": stds.to_dict(),
        }

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        transformed = apply_feature_standardizer(df, self.standardizer)
        return transformed[self.feature_columns].astype(float).to_numpy(dtype=float)

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_columns: list[str],
        label_column: str = "label",
    ) -> NumpyLinearRanker:
        """Fit the fallback ranker."""
        self.feature_columns = list(feature_columns)
        self._fit_standardizer(train_df)
        x = self._matrix(train_df)
        y = train_df[label_column].astype(float).to_numpy(dtype=float)
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0.0, 0.01, size=x.shape[1]) if x.shape[1] else np.zeros(0)
        self.bias = 0.0

        if x.size == 0:
            return self
        for _ in range(self.num_epochs):
            logits = x @ self.weights + self.bias
            probs = self._sigmoid(logits)
            error = probs - y
            grad_w = (x.T @ error) / len(y) + self.regularization * self.weights
            grad_b = float(error.mean())
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict relevance scores."""
        if not self.feature_columns:
            return np.zeros(len(df), dtype=float)
        logits = self._matrix(df) @ self.weights + self.bias
        return self._sigmoid(logits).astype(float)

    def save(self, path: str | Path) -> None:
        """Save the ranker with pickle."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> NumpyLinearRanker:
        """Load a saved ranker."""
        with Path(path).open("rb") as file:
            loaded = pickle.load(file)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded
