"""BM25 retrieval followed by lightweight metadata-overlap reranking."""

from __future__ import annotations

import math
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.text_utils import DEFAULT_ITEM_TEXT_FIELDS, tokenize


class OverlapRerankRetriever:
    """Rerank BM25 candidates with general text-overlap and quality signals."""

    method_name = "overlap_rerank_bm25"

    def __init__(
        self,
        candidate_pool_size: int = 100,
        item_text_fields: Sequence[str] | None = None,
        bm25_params: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.candidate_pool_size = int(candidate_pool_size)
        self.item_text_fields = list(item_text_fields or DEFAULT_ITEM_TEXT_FIELDS)
        self.bm25_params = dict(bm25_params or {})
        self.weights = {
            "bm25": 1.0,
            "title_overlap": 0.0,
            "category_overlap": 0.0,
            "brand_overlap": 0.0,
            "description_overlap": 0.0,
            "rating": 0.0,
            "popularity": 0.0,
            **{str(key): float(value) for key, value in (weights or {}).items()},
        }
        self.base_retriever = BM25Retriever(
            k1=float(self.bm25_params.get("k1", 1.5)),
            b=float(self.bm25_params.get("b", 0.75)),
            item_text_fields=self.item_text_fields,
            field_weights=self.bm25_params.get("field_weights"),
        )
        self.items_by_id: dict[str, dict[str, Any]] = {}
        self.max_log_rating_count = 1.0
        self.backend = "bm25+overlap_rerank"

    def fit(self, items: pd.DataFrame) -> OverlapRerankRetriever:
        """Fit the underlying BM25 retriever and cache item metadata."""
        self.base_retriever.fit(items)
        frame = items.copy()
        frame["item_id"] = frame["item_id"].astype(str)
        self.items_by_id = {str(row["item_id"]): row.to_dict() for _, row in frame.iterrows()}
        if "rating_count" in frame.columns and not frame.empty:
            self.max_log_rating_count = max(
                1.0,
                float(frame["rating_count"].fillna(0).map(lambda value: math.log1p(value)).max()),
            )
        return self

    @staticmethod
    def _overlap_ratio(query_tokens: set[str], value: Any) -> float:
        if not query_tokens or pd.isna(value):
            return 0.0
        value_tokens = set(tokenize(str(value)))
        if not value_tokens:
            return 0.0
        return len(query_tokens.intersection(value_tokens)) / len(query_tokens)

    @staticmethod
    def _normalize_scores(values: list[float]) -> list[float]:
        if not values:
            return []
        minimum = min(values)
        maximum = max(values)
        if maximum <= minimum:
            return [0.0 for _ in values]
        return [(value - minimum) / (maximum - minimum) for value in values]

    def _feature_scores(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, float]]:
        query_tokens = set(tokenize(query))
        bm25_scores = self._normalize_scores(
            [float(candidate["score"]) for candidate in candidates]
        )
        rows: list[dict[str, float]] = []
        for candidate, bm25_score in zip(candidates, bm25_scores, strict=True):
            item = self.items_by_id.get(str(candidate["item_id"]), {})
            rating = float(item.get("avg_rating") or 0.0) / 5.0
            popularity = (
                math.log1p(float(item.get("rating_count") or 0.0)) / self.max_log_rating_count
            )
            rows.append(
                {
                    "bm25": bm25_score,
                    "title_overlap": self._overlap_ratio(query_tokens, item.get("title", "")),
                    "category_overlap": self._overlap_ratio(query_tokens, item.get("category", "")),
                    "brand_overlap": self._overlap_ratio(query_tokens, item.get("brand", "")),
                    "description_overlap": self._overlap_ratio(
                        query_tokens,
                        item.get("description", ""),
                    ),
                    "rating": rating,
                    "popularity": popularity,
                }
            )
        return rows

    def search(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        """Search BM25 candidates, rerank them, and return the requested top-k."""
        pool_size = max(int(top_k), self.candidate_pool_size)
        candidates = self.base_retriever.search(query, top_k=pool_size)
        feature_rows = self._feature_scores(query, candidates)
        scored = []
        for candidate, features in zip(candidates, feature_rows, strict=True):
            rerank_score = sum(
                self.weights.get(name, 0.0) * value for name, value in features.items()
            )
            scored.append((candidate, features, float(rerank_score)))

        ranked = sorted(
            scored,
            key=lambda row: (-row[2], -float(row[0]["score"]), str(row[0]["item_id"])),
        )[: max(0, int(top_k))]
        return [
            {
                **candidate,
                "score": rerank_score,
                "bm25_score": float(candidate["score"]),
                "rerank_features": features,
                "rank": rank,
            }
            for rank, (candidate, features, rerank_score) in enumerate(ranked, start=1)
        ]

    def save(self, path: str | Path) -> None:
        """Persist the reranker with pickle."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> OverlapRerankRetriever:
        """Load a persisted reranker."""
        with Path(path).open("rb") as file:
            loaded = pickle.load(file)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded
