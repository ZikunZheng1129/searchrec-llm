"""Feature fusion for local multimodal item representations."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.multimodal.image_encoder import PrecomputedImageFeatureEncoder, TestStubImageEncoder
from src.multimodal.metadata_encoder import MetadataItemEncoder
from src.multimodal.text_encoder import TextItemEncoder


class MultimodalFusionEncoder:
    """Concatenate weighted text, metadata, and optional image vectors."""

    def __init__(
        self,
        text_encoder: TextItemEncoder | None = None,
        metadata_encoder: MetadataItemEncoder | None = None,
        image_encoder: Any | None = None,
        strategy: str = "concat",
        normalize_output: bool = True,
        text_weight: float = 0.75,
        metadata_weight: float = 0.25,
        image_weight: float = 0.0,
    ) -> None:
        self.text_encoder = text_encoder
        self.metadata_encoder = metadata_encoder
        self.image_encoder = image_encoder
        self.strategy = strategy
        self.normalize_output = bool(normalize_output)
        self.text_weight = float(text_weight)
        self.metadata_weight = float(metadata_weight)
        self.image_weight = float(image_weight)
        self.image_available_rate_ = 0.0
        self.component_dims: dict[str, int] = {"text": 0, "metadata": 0, "image": 0}

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms

    def fit(self, items: pd.DataFrame) -> MultimodalFusionEncoder:
        """Fit component encoders."""
        if self.strategy not in {"concat", "weighted_concat"}:
            raise ValueError(f"Unsupported fusion strategy: {self.strategy}")
        if self.text_encoder is not None:
            self.text_encoder.fit(items)
            self.component_dims["text"] = len(self.text_encoder.get_feature_names())
        if self.metadata_encoder is not None:
            self.metadata_encoder.fit(items)
            self.component_dims["metadata"] = len(self.metadata_encoder.get_feature_names())
        if self.image_encoder is not None:
            self.image_encoder.fit(items)
            image_vectors = self.image_encoder.transform(items)
            self.component_dims["image"] = (
                0 if image_vectors is None else int(image_vectors.shape[1])
            )
            self.image_available_rate_ = float(self.image_encoder.image_available_rate(items))
        return self

    def _components(self, items: pd.DataFrame) -> list[np.ndarray]:
        components: list[np.ndarray] = []
        if self.text_encoder is not None:
            components.append(self.text_weight * self.text_encoder.transform(items))
        if self.metadata_encoder is not None:
            components.append(self.metadata_weight * self.metadata_encoder.transform(items))
        if self.image_encoder is not None:
            image_vectors = self.image_encoder.transform(items)
            if image_vectors is not None and image_vectors.size:
                components.append(self.image_weight * image_vectors)
        return components

    def transform(self, items: pd.DataFrame) -> np.ndarray:
        """Encode fused item vectors."""
        components = self._components(items)
        matrix = np.concatenate(components, axis=1) if components else np.zeros((len(items), 0))
        return self._normalize(matrix) if self.normalize_output and matrix.size else matrix

    def fit_transform(self, items: pd.DataFrame) -> np.ndarray:
        """Fit and encode fused item vectors."""
        return self.fit(items).transform(items)

    def encode_queries(self, query_texts: list[str]) -> np.ndarray:
        """Encode queries into the same fused dimension as item vectors."""
        components: list[np.ndarray] = []
        if self.text_encoder is not None:
            components.append(self.text_weight * self.text_encoder.encode_queries(query_texts))
        if self.metadata_encoder is not None:
            components.append(
                self.metadata_weight * self.metadata_encoder.encode_queries(query_texts)
            )
        if self.component_dims.get("image", 0) > 0:
            components.append(
                np.zeros((len(query_texts), self.component_dims["image"]), dtype=float)
            )
        matrix = (
            np.concatenate(components, axis=1) if components else np.zeros((len(query_texts), 0))
        )
        return self._normalize(matrix) if self.normalize_output and matrix.size else matrix

    def save(self, path: str | Path) -> None:
        """Save the fusion encoder with pickle."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> MultimodalFusionEncoder:
        """Load a fusion encoder from pickle."""
        with Path(path).open("rb") as file:
            loaded = pickle.load(file)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded


def build_image_encoder_from_config(config: dict[str, Any]) -> Any | None:
    """Build optional image encoder from config."""
    image_config = config.get("image_encoder", {})
    if not bool(image_config.get("enabled", False)):
        return None
    if bool(image_config.get("allow_test_stub", False)):
        return TestStubImageEncoder(
            allow_test_stub=True,
            dim=int(image_config.get("test_stub_dim", 8)),
            normalize=bool(image_config.get("normalize", True)),
        )
    return PrecomputedImageFeatureEncoder(
        image_embedding_column=str(image_config.get("image_embedding_column", "image_embedding")),
        normalize=bool(image_config.get("normalize", True)),
    )


def build_encoder_from_config(config: dict[str, Any]) -> Any:
    """Create the configured Stage 9 item encoder."""
    method = str(config.get("multimodal", {}).get("method"))
    if method == "text_only":
        text_config = config.get("text_encoder", {})
        return TextItemEncoder(
            fields=text_config.get("fields"),
            max_features=int(text_config.get("max_features", 5000)),
            normalize=bool(text_config.get("normalize", True)),
        )
    if method == "metadata_only":
        metadata_config = config.get("metadata_encoder", {})
        return MetadataItemEncoder(
            categorical_fields=metadata_config.get("categorical_fields"),
            numeric_fields=metadata_config.get("numeric_fields"),
            add_log_numeric=metadata_config.get("add_log_numeric"),
            normalize_numeric=bool(metadata_config.get("normalize_numeric", True)),
        )
    if method in {"text_metadata_fusion", "multimodal_fusion"}:
        text_config = config.get("text_encoder", {})
        metadata_config = config.get("metadata_encoder", {})
        fusion_config = config.get("fusion", {})
        image_encoder = (
            build_image_encoder_from_config(config)
            if method == "multimodal_fusion"
            and bool(config.get("multimodal", {}).get("include_image_if_available", True))
            else None
        )
        return MultimodalFusionEncoder(
            text_encoder=TextItemEncoder(
                fields=text_config.get("fields"),
                max_features=int(text_config.get("max_features", 5000)),
                normalize=bool(text_config.get("normalize", True)),
            ),
            metadata_encoder=MetadataItemEncoder(
                categorical_fields=metadata_config.get("categorical_fields"),
                numeric_fields=metadata_config.get("numeric_fields"),
                add_log_numeric=metadata_config.get("add_log_numeric"),
                normalize_numeric=bool(metadata_config.get("normalize_numeric", True)),
            ),
            image_encoder=image_encoder,
            strategy=str(fusion_config.get("strategy", "concat")),
            normalize_output=bool(fusion_config.get("normalize_output", True)),
            text_weight=float(fusion_config.get("text_weight", 0.75)),
            metadata_weight=float(fusion_config.get("metadata_weight", 0.25)),
            image_weight=float(fusion_config.get("image_weight", 0.0)),
        )
    raise ValueError(f"Unsupported multimodal.method: {method}")
