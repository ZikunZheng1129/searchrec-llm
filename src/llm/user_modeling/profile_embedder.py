"""Lightweight bag-of-words profile text embeddings."""

from __future__ import annotations

import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np


class ProfileEmbedder:
    """Deterministic bag-of-words profile embedder."""

    def __init__(self, max_features: int = 512, normalize: bool = True) -> None:
        self.max_features = int(max_features)
        self.normalize = bool(normalize)
        self.vocabulary: dict[str, int] = {}
        self.feature_names: list[str] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", str(text).lower())

    def fit(self, profile_texts: list[str]) -> ProfileEmbedder:
        """Fit a deterministic vocabulary."""
        counts: Counter[str] = Counter()
        for text in profile_texts:
            counts.update(set(self._tokenize(text)))
        sorted_terms = sorted(counts.items(), key=lambda row: (-row[1], row[0]))
        if self.max_features > 0:
            sorted_terms = sorted_terms[: self.max_features]
        self.feature_names = sorted(term for term, _ in sorted_terms)
        self.vocabulary = {term: index for index, term in enumerate(self.feature_names)}
        return self

    def transform(self, profile_texts: list[str]) -> np.ndarray:
        """Transform profile texts into vectors."""
        matrix = np.zeros((len(profile_texts), len(self.feature_names)), dtype=float)
        if not self.vocabulary:
            return matrix
        for row_index, text in enumerate(profile_texts):
            for token in self._tokenize(text):
                column = self.vocabulary.get(token)
                if column is not None:
                    matrix[row_index, column] += 1.0
        if self.normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            matrix = matrix / norms
        return matrix

    def fit_transform(self, profile_texts: list[str]) -> np.ndarray:
        """Fit and transform profile texts."""
        return self.fit(profile_texts).transform(profile_texts)

    def get_feature_names(self) -> list[str]:
        """Return vocabulary feature names."""
        return list(self.feature_names)

    def save(self, path: str | Path) -> None:
        """Save the embedder."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> ProfileEmbedder:
        """Load a saved embedder."""
        with Path(path).open("rb") as file:
            loaded = pickle.load(file)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded
