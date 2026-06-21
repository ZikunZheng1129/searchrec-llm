"""Optional LightGBM ranker with a deterministic NumPy fallback."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ranking.dataset import apply_feature_standardizer
from src.ranking.feature_ranker import NumpyLinearRanker


class LightGBMRanker:
    """Train LightGBM when available, otherwise use NumpyLinearRanker."""

    def __init__(
        self,
        backend: str = "auto",
        fallback_backend: str = "numpy_linear",
        learning_rate: float = 0.05,
        num_boost_round: int = 30,
        num_epochs_fallback: int = 100,
        regularization: float = 0.001,
        seed: int = 42,
    ) -> None:
        self.requested_backend = backend
        self.fallback_backend = fallback_backend
        self.learning_rate = float(learning_rate)
        self.num_boost_round = int(num_boost_round)
        self.num_epochs_fallback = int(num_epochs_fallback)
        self.regularization = float(regularization)
        self.seed = int(seed)
        self.backend = "unfit"
        self.model: Any = None
        self.feature_columns: list[str] = []
        self.standardizer: dict[str, Any] = {}

    def _fit_lightgbm(
        self,
        train_df: pd.DataFrame,
        feature_columns: list[str],
        label_column: str,
    ) -> bool:
        if self.requested_backend not in {"auto", "lightgbm"}:
            return False
        try:
            import lightgbm as lgb  # type: ignore[import-not-found]
        except ImportError:
            if self.requested_backend == "lightgbm":
                raise
            return False

        means = train_df[feature_columns].astype(float).mean(axis=0)
        stds = train_df[feature_columns].astype(float).std(axis=0).replace(0.0, 1.0).fillna(1.0)
        self.standardizer = {
            "feature_columns": list(feature_columns),
            "mean": means.to_dict(),
            "std": stds.to_dict(),
        }
        transformed = apply_feature_standardizer(train_df, self.standardizer)
        x = transformed[feature_columns].astype(float).to_numpy(dtype=float)
        y = train_df[label_column].astype(int).to_numpy()
        model = lgb.LGBMClassifier(
            objective="binary",
            learning_rate=self.learning_rate,
            n_estimators=self.num_boost_round,
            random_state=self.seed,
            verbose=-1,
        )
        model.fit(x, y)
        self.model = model
        self.backend = "lightgbm"
        return True

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_columns: list[str],
        group_column: str = "query_id",
        label_column: str = "label",
    ) -> LightGBMRanker:
        """Fit LightGBM or the fallback backend."""
        del group_column
        self.feature_columns = list(feature_columns)
        if self._fit_lightgbm(train_df, self.feature_columns, label_column):
            return self
        if self.fallback_backend != "numpy_linear":
            raise ValueError(f"Unsupported fallback backend: {self.fallback_backend}")
        fallback = NumpyLinearRanker(
            learning_rate=self.learning_rate,
            num_epochs=self.num_epochs_fallback,
            regularization=self.regularization,
            seed=self.seed,
        ).fit(train_df, self.feature_columns, label_column=label_column)
        self.model = fallback
        self.standardizer = fallback.standardizer
        self.backend = fallback.backend
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict relevance scores."""
        if self.backend == "lightgbm":
            transformed = apply_feature_standardizer(df, self.standardizer)
            x = transformed[self.feature_columns].astype(float).to_numpy(dtype=float)
            return self.model.predict_proba(x)[:, 1].astype(float)
        if isinstance(self.model, NumpyLinearRanker):
            return self.model.predict(df)
        return np.zeros(len(df), dtype=float)

    def save(self, path: str | Path) -> None:
        """Save the ranker with pickle."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> LightGBMRanker:
        """Load a saved ranker."""
        with Path(path).open("rb") as file:
            loaded = pickle.load(file)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded
