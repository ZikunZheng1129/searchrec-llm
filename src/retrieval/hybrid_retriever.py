"""Hybrid sparse plus dense retrieval baseline."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import TfidfDenseRetriever


class HybridRetriever:
    """Combine BM25 and TF-IDF dense scores with weighted fusion."""

    method_name = "hybrid"
    index_backend = "bm25+dense_numpy"

    def __init__(
        self,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        score_normalization: str = "minmax",
        bm25_params: dict[str, Any] | None = None,
        dense_params: dict[str, Any] | None = None,
    ) -> None:
        self.bm25_weight = float(bm25_weight)
        self.dense_weight = float(dense_weight)
        self.score_normalization = score_normalization
        self.bm25_retriever = BM25Retriever(**(bm25_params or {}))
        self.dense_retriever = TfidfDenseRetriever(**(dense_params or {}))
        self.item_ids: list[str] = []

    def fit(self, items: pd.DataFrame) -> HybridRetriever:
        """Fit both internal retrievers."""
        self.bm25_retriever.fit(items)
        self.dense_retriever.fit(items)
        self.item_ids = sorted(
            set(self.bm25_retriever.item_ids) | set(self.dense_retriever.item_ids)
        )
        return self

    @staticmethod
    def _minmax(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        values = np.array(list(scores.values()), dtype=float)
        min_value = float(values.min())
        max_value = float(values.max())
        if max_value == min_value:
            return {item_id: 0.0 for item_id in scores}
        return {
            item_id: (score - min_value) / (max_value - min_value)
            for item_id, score in scores.items()
        }

    def _normalize_scores(self, scores: dict[str, float]) -> dict[str, float]:
        if self.score_normalization == "minmax":
            return self._minmax(scores)
        if self.score_normalization in {"none", "raw"}:
            return scores
        raise ValueError(f"Unsupported score_normalization: {self.score_normalization}")

    def search(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        """Search with weighted BM25 and dense score fusion."""
        candidate_count = len(self.item_ids)
        if candidate_count == 0:
            return []

        bm25_results = self.bm25_retriever.search(query, top_k=candidate_count)
        dense_results = self.dense_retriever.search(query, top_k=candidate_count)
        bm25_scores = {row["item_id"]: float(row["score"]) for row in bm25_results}
        dense_scores = {row["item_id"]: float(row["score"]) for row in dense_results}
        normalized_bm25 = self._normalize_scores(bm25_scores)
        normalized_dense = self._normalize_scores(dense_scores)

        combined = []
        for item_id in self.item_ids:
            bm25_score = bm25_scores.get(item_id, 0.0)
            dense_score = dense_scores.get(item_id, 0.0)
            score = self.bm25_weight * normalized_bm25.get(
                item_id, 0.0
            ) + self.dense_weight * normalized_dense.get(item_id, 0.0)
            combined.append(
                {
                    "item_id": item_id,
                    "score": float(score),
                    "bm25_score": float(bm25_score),
                    "dense_score": float(dense_score),
                }
            )

        limit = max(0, min(int(top_k), len(combined)))
        ranked = sorted(combined, key=lambda row: (-row["score"], row["item_id"]))[:limit]
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        return ranked

    def batch_search(self, queries: list[str], top_k: int = 50) -> dict[str, list[dict[str, Any]]]:
        """Search a batch of query strings."""
        return {query: self.search(query, top_k=top_k) for query in queries}

    def save(self, path: str | Path) -> None:
        """Persist the hybrid index with pickle."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> HybridRetriever:
        """Load a persisted hybrid index."""
        with Path(path).open("rb") as file:
            loaded = pickle.load(file)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded
