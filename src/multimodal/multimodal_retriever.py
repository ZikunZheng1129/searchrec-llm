"""Vector retrieval over local multimodal item embeddings."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class MultimodalRetriever:
    """Exact vector retriever for Stage 9 item embeddings."""

    def __init__(self) -> None:
        self.item_ids: list[str] = []
        self.embeddings = np.zeros((0, 0), dtype=float)
        self.query_encoder: Any = None

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms

    def fit(
        self,
        items: pd.DataFrame,
        embeddings: np.ndarray,
        query_encoder: Any,
    ) -> MultimodalRetriever:
        """Store item vectors and query encoder."""
        if "item_id" not in items.columns:
            raise ValueError("items must contain item_id")
        self.item_ids = items["item_id"].astype(str).tolist()
        self.embeddings = self._normalize(np.asarray(embeddings, dtype=float))
        self.query_encoder = query_encoder
        return self

    def search(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        """Search top-k item vectors for a query."""
        if self.embeddings.size == 0 or not self.item_ids:
            return []
        query_vector = np.asarray(self.query_encoder.encode_queries([query]), dtype=float)
        if query_vector.size == 0:
            scores = np.zeros(len(self.item_ids), dtype=float)
        else:
            query_vector = self._normalize(query_vector)
            scores = np.asarray(query_vector @ self.embeddings.T).ravel()
        limit = max(0, min(int(top_k), len(self.item_ids)))
        ranked = sorted(
            ((item_id, float(scores[index])) for index, item_id in enumerate(self.item_ids)),
            key=lambda row: (-row[1], row[0]),
        )[:limit]
        return [
            {"item_id": item_id, "score": score, "rank": rank}
            for rank, (item_id, score) in enumerate(ranked, start=1)
        ]

    def batch_search(self, queries: list[str], top_k: int = 50) -> dict[str, list[dict[str, Any]]]:
        """Search multiple queries."""
        return {query: self.search(query, top_k=top_k) for query in queries}

    def save(self, path: str | Path) -> None:
        """Save retriever with pickle."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> MultimodalRetriever:
        """Load retriever with pickle."""
        with Path(path).open("rb") as file:
            loaded = pickle.load(file)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded
