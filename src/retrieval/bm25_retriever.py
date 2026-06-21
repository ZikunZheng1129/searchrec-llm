"""BM25 sparse retrieval baseline."""

from __future__ import annotations

import math
import pickle
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.retrieval.text_utils import DEFAULT_ITEM_TEXT_FIELDS, build_item_text, tokenize


class BM25Retriever:
    """A lightweight BM25 retriever implemented with Python and pandas."""

    method_name = "bm25"

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        item_text_fields: Sequence[str] | None = None,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.item_text_fields = list(item_text_fields or DEFAULT_ITEM_TEXT_FIELDS)
        self.item_ids: list[str] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_length = 0.0
        self.doc_term_counts: list[Counter[str]] = []
        self.idf: dict[str, float] = {}

    def fit(self, items: pd.DataFrame) -> BM25Retriever:
        """Build a BM25 index from item metadata."""
        if "item_id" not in items.columns:
            raise ValueError("items must contain an item_id column")

        sorted_items = items.copy()
        sorted_items["item_id"] = sorted_items["item_id"].astype(str)
        sorted_items = sorted_items.sort_values("item_id").reset_index(drop=True)
        self.item_ids = sorted_items["item_id"].tolist()
        self.doc_term_counts = []
        self.doc_lengths = []

        document_frequency: Counter[str] = Counter()
        for _, row in sorted_items.iterrows():
            tokens = tokenize(build_item_text(row, self.item_text_fields))
            term_counts = Counter(tokens)
            self.doc_term_counts.append(term_counts)
            self.doc_lengths.append(len(tokens))
            document_frequency.update(term_counts.keys())

        num_docs = len(self.item_ids)
        self.avg_doc_length = sum(self.doc_lengths) / num_docs if num_docs else 0.0
        self.idf = {
            term: math.log(1.0 + (num_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }
        return self

    def _score_query(self, query: str) -> list[float]:
        query_terms = Counter(tokenize(query))
        if not self.item_ids:
            return []
        if not query_terms or self.avg_doc_length <= 0:
            return [0.0 for _ in self.item_ids]

        scores = []
        for index, term_counts in enumerate(self.doc_term_counts):
            doc_length = self.doc_lengths[index]
            score = 0.0
            for term, query_frequency in query_terms.items():
                term_frequency = term_counts.get(term, 0)
                if term_frequency == 0:
                    continue
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * doc_length / self.avg_doc_length
                )
                score += (
                    self.idf.get(term, 0.0)
                    * query_frequency
                    * term_frequency
                    * (self.k1 + 1.0)
                    / denominator
                )
            scores.append(float(score))
        return scores

    def search(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        """Search for the top-k items for a query."""
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
        """Persist the BM25 index with pickle."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> BM25Retriever:
        """Load a persisted BM25 index."""
        with Path(path).open("rb") as file:
            loaded = pickle.load(file)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded
