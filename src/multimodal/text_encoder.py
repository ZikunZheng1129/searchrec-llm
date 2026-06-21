"""Lightweight text item encoder for Stage 9."""

from __future__ import annotations

import math
import pickle
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.retrieval.text_utils import DEFAULT_ITEM_TEXT_FIELDS, build_item_text, tokenize


class TextItemEncoder:
    """Deterministic TF-IDF encoder for item text and query text."""

    def __init__(
        self,
        fields: Sequence[str] | None = None,
        max_features: int = 5000,
        normalize: bool = True,
    ) -> None:
        self.fields = list(fields or DEFAULT_ITEM_TEXT_FIELDS)
        self.max_features = int(max_features)
        self.normalize = bool(normalize)
        self.vocabulary: dict[str, int] = {}
        self.feature_names: list[str] = []
        self.idf: dict[str, float] = {}

    def _item_documents(self, items: pd.DataFrame) -> list[list[str]]:
        return [tokenize(build_item_text(row, self.fields)) for _, row in items.iterrows()]

    def fit(self, items: pd.DataFrame) -> TextItemEncoder:
        """Fit vocabulary and IDF from item metadata."""
        documents = self._item_documents(items)
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
        return self

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms

    def _vectorize(self, documents: list[list[str]]) -> np.ndarray:
        matrix = np.zeros((len(documents), len(self.feature_names)), dtype=float)
        if not self.vocabulary:
            return matrix
        for row_index, tokens in enumerate(documents):
            counts = Counter(token for token in tokens if token in self.vocabulary)
            token_count = sum(counts.values())
            if token_count == 0:
                continue
            for token, count in counts.items():
                matrix[row_index, self.vocabulary[token]] = (count / token_count) * self.idf[token]
        return self._l2_normalize(matrix) if self.normalize else matrix

    def transform(self, items: pd.DataFrame) -> np.ndarray:
        """Encode item rows into TF-IDF vectors."""
        return self._vectorize(self._item_documents(items))

    def fit_transform(self, items: pd.DataFrame) -> np.ndarray:
        """Fit and encode item rows."""
        return self.fit(items).transform(items)

    def encode_queries(self, query_texts: list[str]) -> np.ndarray:
        """Encode query strings with the item vocabulary."""
        return self._vectorize([tokenize(query_text) for query_text in query_texts])

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
    def load(cls, path: str | Path) -> TextItemEncoder:
        """Load encoder from pickle."""
        with Path(path).open("rb") as file:
            loaded = pickle.load(file)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded
