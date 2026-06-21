"""Deterministic mixed ranker over relevance and proxy features."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


class MixedRanker:
    """Blend relevance, metadata proxies, diversity, and cold-start signals."""

    backend = "deterministic_mixed"

    def __init__(
        self,
        relevance_weight: float = 0.65,
        authority_weight: float = 0.15,
        conversion_weight: float = 0.10,
        diversity_weight: float = 0.05,
        cold_start_weight: float = 0.05,
        base_score_column: str = "hybrid_score",
        diversity_group_column: str = "category",
    ) -> None:
        self.relevance_weight = float(relevance_weight)
        self.authority_weight = float(authority_weight)
        self.conversion_weight = float(conversion_weight)
        self.diversity_weight = float(diversity_weight)
        self.cold_start_weight = float(cold_start_weight)
        self.base_score_column = base_score_column
        self.diversity_group_column = diversity_group_column

    def fit(self, train_df: pd.DataFrame | None = None) -> MixedRanker:
        """No-op fit for API consistency."""
        del train_df
        return self

    @staticmethod
    def _minmax(values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
        min_value = float(numeric.min())
        max_value = float(numeric.max())
        if max_value == min_value:
            return pd.Series(np.zeros(len(numeric), dtype=float), index=values.index)
        return (numeric - min_value) / (max_value - min_value)

    def _score_group(self, group: pd.DataFrame) -> pd.DataFrame:
        output = group.copy()
        output["relevance_component"] = self._minmax(output.get(self.base_score_column, 0.0))
        output["authority_component"] = self._minmax(output.get("authority_score", 0.0))
        output["conversion_component"] = self._minmax(output.get("conversion_proxy", 0.0))
        output["cold_start_component"] = self._minmax(output.get("cold_start_score", 0.0))

        base_order = output.sort_values(
            [self.base_score_column, "candidate_item_id"],
            ascending=[False, True],
        )
        category_counts: dict[str, int] = {}
        diversity = {}
        for index, row in base_order.iterrows():
            group_value = str(row.get(self.diversity_group_column, ""))
            seen_count = category_counts.get(group_value, 0)
            diversity[index] = 1.0 / (1.0 + seen_count)
            category_counts[group_value] = seen_count + 1
        output["diversity_component"] = pd.Series(diversity, dtype=float)
        output["model_score"] = (
            self.relevance_weight * output["relevance_component"]
            + self.authority_weight * output["authority_component"]
            + self.conversion_weight * output["conversion_component"]
            + self.diversity_weight * output["diversity_component"]
            + self.cold_start_weight * output["cold_start_component"]
        )
        return output

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return mixed scores aligned to df rows."""
        if df.empty:
            return np.zeros(0, dtype=float)
        scored = pd.concat(
            [self._score_group(group) for _, group in df.groupby("query_id", sort=False)]
        ).sort_index()
        return scored["model_score"].to_numpy(dtype=float)

    def rerank(self, df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        """Return reranked candidates with component scores."""
        if df.empty:
            return df.copy()
        scored = pd.concat(
            [self._score_group(group) for _, group in df.groupby("query_id", sort=False)]
        )
        scored = scored.sort_values(
            ["query_id", "model_score", "candidate_item_id"],
            ascending=[True, False, True],
        )
        return scored.groupby("query_id", sort=False).head(int(top_k)).reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        """Save the mixed ranker."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> MixedRanker:
        """Load a saved mixed ranker."""
        with Path(path).open("rb") as file:
            loaded = pickle.load(file)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded
