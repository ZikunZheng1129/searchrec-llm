"""FAISS-style vector retrieval wrapper with NumPy fallback."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.retrieval.dense_retriever import TfidfDenseRetriever
from src.retrieval.text_utils import DEFAULT_ITEM_TEXT_FIELDS


class FaissRetriever:
    """A FAISS-compatible retriever that safely falls back to exact NumPy search."""

    method_name = "faiss"

    def __init__(
        self,
        backend: str = "auto",
        metric: str = "cosine",
        fallback_to_numpy: bool = True,
        max_features: int = 5000,
        normalize: bool = True,
        item_text_fields: Sequence[str] | None = None,
    ) -> None:
        self.requested_backend = backend
        self.backend = "uninitialized"
        self.metric = metric
        self.fallback_to_numpy = bool(fallback_to_numpy)
        self.dense_retriever = TfidfDenseRetriever(
            max_features=max_features,
            normalize=normalize,
            item_text_fields=item_text_fields or DEFAULT_ITEM_TEXT_FIELDS,
        )
        self._faiss_index: Any = None

    def fit(self, items: pd.DataFrame) -> FaissRetriever:
        """Fit the underlying vectorizer and initialize the search backend."""
        self.dense_retriever.fit(items)
        self._initialize_backend()
        return self

    def _initialize_backend(self) -> None:
        if self.requested_backend == "numpy":
            self.backend = "numpy"
            self._faiss_index = None
            return

        try:
            import faiss  # type: ignore[import-not-found]
        except ImportError:
            if not self.fallback_to_numpy:
                raise ImportError("faiss is not installed and fallback_to_numpy is false") from None
            self.backend = "numpy"
            self._faiss_index = None
            return

        if self.metric != "cosine":
            raise ValueError("Stage 3 FaissRetriever currently supports metric='cosine' only")

        item_vectors = np.asarray(self.dense_retriever.item_vectors, dtype=np.float32)
        index = faiss.IndexFlatIP(item_vectors.shape[1])
        if len(item_vectors):
            index.add(item_vectors)
        self._faiss_index = index
        self.backend = "faiss"

    def search(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        """Search the vector index with FAISS or exact NumPy fallback."""
        item_ids = self.dense_retriever.item_ids
        limit = max(0, min(int(top_k), len(item_ids)))
        if limit == 0:
            return []

        if self.backend == "faiss" and self._faiss_index is not None:
            query_vector = np.asarray(
                self.dense_retriever.encode_queries([query]),
                dtype=np.float32,
            )
            scores, indices = self._faiss_index.search(query_vector, limit)
            pairs = []
            for position, score in enumerate(scores[0]):
                index = int(indices[0][position])
                if index >= 0:
                    pairs.append((item_ids[index], float(score)))
            ranked = sorted(pairs, key=lambda row: (-row[1], row[0]))[:limit]
        else:
            scores = self.dense_retriever._score_query(query)
            ranked = sorted(
                ((item_id, scores[index]) for index, item_id in enumerate(item_ids)),
                key=lambda row: (-row[1], row[0]),
            )[:limit]

        return [
            {"item_id": item_id, "score": float(score), "rank": rank}
            for rank, (item_id, score) in enumerate(ranked, start=1)
        ]

    def batch_search(self, queries: list[str], top_k: int = 50) -> dict[str, list[dict[str, Any]]]:
        """Search a batch of query strings."""
        return {query: self.search(query, top_k=top_k) for query in queries}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_faiss_index"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if self.backend == "faiss":
            self._initialize_backend()

    def save(self, path: str | Path) -> None:
        """Persist the FAISS-style index metadata and dense vectors."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> FaissRetriever:
        """Load a persisted FAISS-style index."""
        with Path(path).open("rb") as file:
            loaded = pickle.load(file)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded
