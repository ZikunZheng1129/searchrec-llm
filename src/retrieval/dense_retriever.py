"""Lightweight TF-IDF dense retrieval baseline."""

from __future__ import annotations

import math
import pickle
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.retrieval.text_utils import DEFAULT_ITEM_TEXT_FIELDS, build_item_text, tokenize


class TfidfDenseRetriever:
    """A deterministic TF-IDF retriever using NumPy cosine similarity."""

    method_name = "dense"
    index_backend = "numpy"

    def __init__(
        self,
        max_features: int = 5000,
        normalize: bool = True,
        item_text_fields: Sequence[str] | None = None,
    ) -> None:
        self.max_features = int(max_features)
        self.normalize = bool(normalize)
        self.item_text_fields = list(item_text_fields or DEFAULT_ITEM_TEXT_FIELDS)
        self.item_ids: list[str] = []
        self.vocabulary: dict[str, int] = {}
        self.feature_names: list[str] = []
        self.idf: dict[str, float] = {}
        self.item_vectors = np.zeros((0, 0), dtype=float)

    def fit(self, items: pd.DataFrame) -> TfidfDenseRetriever:
        """Build a local TF-IDF item vector index."""
        if "item_id" not in items.columns:
            raise ValueError("items must contain an item_id column")

        sorted_items = items.copy()
        sorted_items["item_id"] = sorted_items["item_id"].astype(str)
        sorted_items = sorted_items.sort_values("item_id").reset_index(drop=True)
        self.item_ids = sorted_items["item_id"].tolist()
        documents = [
            tokenize(build_item_text(row, self.item_text_fields))
            for _, row in sorted_items.iterrows()
        ]
        self._fit_vocabulary(documents)
        self.item_vectors = self._vectorize_tokenized_documents(documents)
        return self

    def _fit_vocabulary(self, documents: list[list[str]]) -> None:
        document_frequency: Counter[str] = Counter()
        for tokens in documents:
            document_frequency.update(set(tokens))

        sorted_terms = sorted(document_frequency.items(), key=lambda row: (-row[1], row[0]))
        if self.max_features > 0:
            sorted_terms = sorted_terms[: self.max_features]
        self.feature_names = sorted(term for term, _ in sorted_terms)
        self.vocabulary = {term: index for index, term in enumerate(self.feature_names)}

        num_docs = max(1, len(documents))
        self.idf = {
            term: math.log((1.0 + num_docs) / (1.0 + document_frequency[term])) + 1.0
            for term in self.feature_names
        }

    def _vectorize_tokenized_documents(self, documents: list[list[str]]) -> np.ndarray:
        matrix = np.zeros((len(documents), len(self.feature_names)), dtype=float)
        if not self.vocabulary:
            return matrix

        for row_index, tokens in enumerate(documents):
            counts = Counter(token for token in tokens if token in self.vocabulary)
            token_count = sum(counts.values())
            if token_count == 0:
                continue
            for token, count in counts.items():
                column_index = self.vocabulary[token]
                tf = count / token_count
                matrix[row_index, column_index] = tf * self.idf[token]

        return self._l2_normalize(matrix) if self.normalize else matrix

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """Encode query strings into TF-IDF vectors."""
        documents = [tokenize(query) for query in queries]
        return self._vectorize_tokenized_documents(documents)

    def _score_query(self, query: str) -> np.ndarray:
        if self.item_vectors.size == 0:
            return np.zeros(0, dtype=float)
        query_vector = self.encode_queries([query])
        if query_vector.size == 0:
            return np.zeros(len(self.item_ids), dtype=float)
        return np.asarray(query_vector @ self.item_vectors.T).ravel()

    def search(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        """Search for top-k items with cosine similarity."""
        scores = self._score_query(query)
        limit = max(0, min(int(top_k), len(self.item_ids)))
        ranked = sorted(
            ((item_id, scores[index]) for index, item_id in enumerate(self.item_ids)),
            key=lambda row: (-row[1], row[0]),
        )[:limit]
        return [
            {"item_id": item_id, "score": float(score), "rank": rank}
            for rank, (item_id, score) in enumerate(ranked, start=1)
        ]

    def batch_search(self, queries: list[str], top_k: int = 50) -> dict[str, list[dict[str, Any]]]:
        """Search a batch of query strings."""
        return {query: self.search(query, top_k=top_k) for query in queries}

    def save(self, path: str | Path) -> None:
        """Persist the dense index with pickle."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> TfidfDenseRetriever:
        """Load a persisted dense index."""
        with Path(path).open("rb") as file:
            loaded = pickle.load(file)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded
