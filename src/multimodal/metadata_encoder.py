"""Metadata item encoder for Stage 9."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


class MetadataItemEncoder:
    """One-hot and normalized numeric metadata encoder."""

    def __init__(
        self,
        categorical_fields: Sequence[str] | None = None,
        numeric_fields: Sequence[str] | None = None,
        add_log_numeric: Sequence[str] | None = None,
        normalize_numeric: bool = True,
    ) -> None:
        self.categorical_fields = list(categorical_fields or [])
        self.numeric_fields = list(numeric_fields or [])
        self.add_log_numeric = list(add_log_numeric or [])
        self.normalize_numeric = bool(normalize_numeric)
        self.category_values: dict[str, list[str]] = {}
        self.numeric_feature_names: list[str] = []
        self.numeric_mean: dict[str, float] = {}
        self.numeric_std: dict[str, float] = {}
        self.feature_names: list[str] = []

    def _numeric_frame(self, items: pd.DataFrame) -> pd.DataFrame:
        data: dict[str, pd.Series] = {}
        for field in self.numeric_fields:
            values = pd.to_numeric(items[field], errors="coerce") if field in items else 0.0
            data[field] = pd.Series(values, index=items.index, dtype=float).fillna(0.0)
        for field in self.add_log_numeric:
            source = pd.to_numeric(items[field], errors="coerce") if field in items else 0.0
            safe = pd.Series(source, index=items.index, dtype=float).fillna(0.0).clip(lower=0.0)
            data[f"log_{field}"] = np.log1p(safe)
        return pd.DataFrame(data, index=items.index)

    def fit(self, items: pd.DataFrame) -> MetadataItemEncoder:
        """Fit categorical vocabulary and numeric normalization stats."""
        self.category_values = {}
        for field in self.categorical_fields:
            if field in items:
                values = sorted(items[field].dropna().astype(str).unique().tolist())
            else:
                values = []
            self.category_values[field] = values
        numeric = self._numeric_frame(items)
        self.numeric_feature_names = sorted(numeric.columns.tolist())
        for column in self.numeric_feature_names:
            mean = float(numeric[column].mean())
            std = float(numeric[column].std())
            self.numeric_mean[column] = mean
            self.numeric_std[column] = std if std > 0 else 1.0
        self.feature_names = []
        for field in self.categorical_fields:
            self.feature_names.extend([f"{field}={value}" for value in self.category_values[field]])
        self.feature_names.extend(self.numeric_feature_names)
        return self

    def transform(self, items: pd.DataFrame) -> np.ndarray:
        """Encode item metadata into numeric vectors."""
        matrix = np.zeros((len(items), len(self.feature_names)), dtype=float)
        column_offset = 0
        for field in self.categorical_fields:
            values = self.category_values.get(field, [])
            value_to_index = {value: index for index, value in enumerate(values)}
            if field in items:
                series = items[field].fillna("").astype(str)
                for row_index, value in enumerate(series):
                    local_index = value_to_index.get(value)
                    if local_index is not None:
                        matrix[row_index, column_offset + local_index] = 1.0
            column_offset += len(values)
        numeric = self._numeric_frame(items)
        for local_index, column in enumerate(self.numeric_feature_names):
            values = numeric[column].astype(float).to_numpy()
            if self.normalize_numeric:
                values = (values - self.numeric_mean[column]) / self.numeric_std[column]
            matrix[:, column_offset + local_index] = np.nan_to_num(values)
        return matrix

    def fit_transform(self, items: pd.DataFrame) -> np.ndarray:
        """Fit and transform item metadata."""
        return self.fit(items).transform(items)

    def encode_queries(self, query_texts: list[str]) -> np.ndarray:
        """Return zero metadata query vectors aligned to item metadata features."""
        return np.zeros((len(query_texts), len(self.feature_names)), dtype=float)

    def get_feature_names(self) -> list[str]:
        """Return deterministic feature names."""
        return list(self.feature_names)

    def save(self, path: str | Path) -> None:
        """Save encoder with pickle."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> MetadataItemEncoder:
        """Load encoder from pickle."""
        with Path(path).open("rb") as file:
            loaded = pickle.load(file)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded
