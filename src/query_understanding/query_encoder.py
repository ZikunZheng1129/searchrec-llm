"""Lightweight query encoder baseline."""

from __future__ import annotations

import re

import numpy as np


class BagOfWordsQueryEncoder:
    """A deterministic bag-of-words query encoder."""

    def __init__(self) -> None:
        self._feature_names: list[str] = []
        self._vocabulary: dict[str, int] = {}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def fit(self, texts: list[str]) -> BagOfWordsQueryEncoder:
        """Build a deterministic vocabulary from query texts."""
        vocabulary = sorted({token for text in texts for token in self._tokenize(text)})
        self._feature_names = vocabulary
        self._vocabulary = {token: index for index, token in enumerate(vocabulary)}
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """Encode texts as bag-of-words count vectors."""
        matrix = np.zeros((len(texts), len(self._feature_names)), dtype=float)
        if not self._vocabulary:
            return matrix

        for row_index, text in enumerate(texts):
            for token in self._tokenize(text):
                column_index = self._vocabulary.get(token)
                if column_index is not None:
                    matrix[row_index, column_index] += 1.0

        return matrix

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit the vocabulary and encode texts."""
        return self.fit(texts).transform(texts)

    def get_feature_names(self) -> list[str]:
        """Return the fitted vocabulary in deterministic order."""
        return list(self._feature_names)
