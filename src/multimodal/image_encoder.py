"""Optional image feature interfaces for Stage 9."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd


class ImageItemEncoder:
    """Base optional image encoder interface."""

    def fit(self, items: pd.DataFrame) -> ImageItemEncoder:
        """Fit image encoder state."""
        del items
        return self

    def transform(self, items: pd.DataFrame) -> np.ndarray | None:
        """Return image vectors, or None when unavailable."""
        del items
        return None

    def fit_transform(self, items: pd.DataFrame) -> np.ndarray | None:
        """Fit and transform image features."""
        return self.fit(items).transform(items)

    def image_available_rate(self, items: pd.DataFrame) -> float:
        """Return fraction of items with usable image features."""
        del items
        return 0.0


class PrecomputedImageFeatureEncoder(ImageItemEncoder):
    """Read precomputed image embeddings from an item metadata column."""

    def __init__(
        self, image_embedding_column: str = "image_embedding", normalize: bool = True
    ) -> None:
        self.image_embedding_column = image_embedding_column
        self.normalize = bool(normalize)
        self.embedding_dim = 0

    @staticmethod
    def _coerce_vector(value: Any) -> list[float] | None:
        if isinstance(value, np.ndarray):
            vector = value.astype(float).tolist()
        elif isinstance(value, (list, tuple)):
            vector = [float(item) for item in value]
        else:
            return None
        return vector if vector else None

    def fit(self, items: pd.DataFrame) -> PrecomputedImageFeatureEncoder:
        """Infer embedding dimension from the first valid vector."""
        self.embedding_dim = 0
        if self.image_embedding_column not in items.columns:
            return self
        for value in items[self.image_embedding_column]:
            vector = self._coerce_vector(value)
            if vector is not None:
                self.embedding_dim = len(vector)
                break
        return self

    def transform(self, items: pd.DataFrame) -> np.ndarray | None:
        """Return precomputed image vectors if the column exists."""
        if self.image_embedding_column not in items.columns or self.embedding_dim == 0:
            return None
        matrix = np.zeros((len(items), self.embedding_dim), dtype=float)
        any_valid = False
        for row_index, value in enumerate(items[self.image_embedding_column]):
            vector = self._coerce_vector(value)
            if vector is not None and len(vector) == self.embedding_dim:
                matrix[row_index] = np.asarray(vector, dtype=float)
                any_valid = True
        if not any_valid:
            return None
        if self.normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            matrix = matrix / norms
        return matrix

    def image_available_rate(self, items: pd.DataFrame) -> float:
        """Return fraction of rows with valid precomputed image vectors."""
        if self.image_embedding_column not in items.columns:
            return 0.0
        valid = [
            self._coerce_vector(value) is not None for value in items[self.image_embedding_column]
        ]
        return float(np.mean(valid)) if valid else 0.0


class TestStubImageEncoder(ImageItemEncoder):
    """Deterministic hash-based image vector stub for tests only."""

    __test__ = False

    def __init__(self, allow_test_stub: bool = False, dim: int = 8, normalize: bool = True) -> None:
        if not allow_test_stub:
            raise ValueError("TestStubImageEncoder requires allow_test_stub=true")
        self.allow_test_stub = bool(allow_test_stub)
        self.dim = int(dim)
        self.normalize = bool(normalize)

    def transform(self, items: pd.DataFrame) -> np.ndarray:
        """Produce deterministic test-only vectors from item IDs."""
        vectors = []
        for item_id in items.get("item_id", pd.Series(range(len(items)))).astype(str):
            digest = hashlib.sha256(item_id.encode("utf-8")).digest()
            values = np.frombuffer(digest[: self.dim], dtype=np.uint8).astype(float) / 255.0
            vectors.append(values)
        matrix = np.vstack(vectors) if vectors else np.zeros((0, self.dim), dtype=float)
        if self.normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            matrix = matrix / norms
        return matrix

    def image_available_rate(self, items: pd.DataFrame) -> float:
        """Test stub availability is complete for provided rows."""
        return 1.0 if len(items) else 0.0
