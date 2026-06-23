"""Query-aware BM25 retriever with deterministic metadata boosts."""

from __future__ import annotations

import math
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.text_utils import build_item_text, tokenize


class QueryAwareBM25Retriever(BM25Retriever):
    """BM25 plus transparent category, brand, use-case, and quality boosts."""

    method_name = "query_aware_bm25"

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        item_text_fields: Sequence[str] | None = None,
        field_weights: dict[str, float] | None = None,
        category_match_boost: float = 0.0,
        brand_match_boost: float = 0.0,
        use_case_term_boost: float = 0.0,
        use_case_terms: dict[str, list[str]] | None = None,
        rating_boost: float = 0.0,
        popularity_boost: float = 0.0,
    ) -> None:
        super().__init__(
            k1=k1,
            b=b,
            item_text_fields=item_text_fields,
            field_weights=field_weights,
        )
        self.category_match_boost = float(category_match_boost)
        self.brand_match_boost = float(brand_match_boost)
        self.use_case_term_boost = float(use_case_term_boost)
        self.use_case_terms = {
            str(key): [str(term) for term in terms] for key, terms in (use_case_terms or {}).items()
        }
        self.rating_boost = float(rating_boost)
        self.popularity_boost = float(popularity_boost)
        self.item_metadata = pd.DataFrame()
        self.item_search_text: dict[str, str] = {}
        self.item_search_terms: dict[str, set[str]] = {}
        self.use_case_trigger_terms: dict[str, list[str]] = {}
        self.use_case_term_sets: dict[str, set[str]] = {}
        self.quality_scores: dict[str, float] = {}
        self.popularity_scores: dict[str, float] = {}

    def fit(self, items: pd.DataFrame) -> QueryAwareBM25Retriever:
        """Build the base BM25 index and cached metadata boost features."""
        super().fit(items)
        sorted_items = items.copy()
        sorted_items["item_id"] = sorted_items["item_id"].astype(str)
        sorted_items = sorted_items.sort_values("item_id").reset_index(drop=True)
        self.item_metadata = sorted_items
        self.item_search_text = {
            str(row["item_id"]): build_item_text(row, self.item_text_fields)
            for _, row in sorted_items.iterrows()
        }
        self.item_search_terms = {
            item_id: set(tokenize(text)) for item_id, text in self.item_search_text.items()
        }
        self.use_case_term_sets = {
            trigger: set(tokenize(" ".join(related_terms)))
            for trigger, related_terms in self.use_case_terms.items()
        }
        self.use_case_trigger_terms = {
            trigger: tokenize(trigger) for trigger in self.use_case_terms
        }
        self.quality_scores = self._normalized_numeric_scores(sorted_items, "avg_rating")
        self.popularity_scores = self._normalized_numeric_scores(
            sorted_items,
            "rating_count",
            log_scale=True,
        )
        return self

    @staticmethod
    def _normalized_numeric_scores(
        items: pd.DataFrame,
        column: str,
        log_scale: bool = False,
    ) -> dict[str, float]:
        if column not in items.columns:
            return {str(item_id): 0.0 for item_id in items["item_id"]}
        values = pd.to_numeric(items[column], errors="coerce").fillna(0.0).astype(float)
        if log_scale:
            values = values.map(lambda value: math.log1p(max(0.0, float(value))))
        min_value = float(values.min())
        max_value = float(values.max())
        if max_value == min_value:
            return {str(item_id): 0.0 for item_id in items["item_id"]}
        normalized = (values - min_value) / (max_value - min_value)
        return {
            str(item_id): float(score)
            for item_id, score in zip(items["item_id"], normalized, strict=True)
        }

    @staticmethod
    def _all_terms_present(terms: list[str], query_terms: set[str]) -> bool:
        return bool(terms) and all(term in query_terms for term in terms)

    def _metadata_boost(
        self,
        query_terms: set[str],
        item_id: str,
        row: pd.Series,
    ) -> float:
        boost = 0.0

        category_terms = tokenize(str(row.get("category", "")))
        if self._all_terms_present(category_terms, query_terms):
            boost += self.category_match_boost

        brand_terms = tokenize(str(row.get("brand", "")))
        if self._all_terms_present(brand_terms, query_terms):
            boost += self.brand_match_boost

        item_text_terms = self.item_search_terms.get(item_id, set())
        for trigger, related_terms in self.use_case_term_sets.items():
            if self._all_terms_present(self.use_case_trigger_terms[trigger], query_terms):
                overlap_count = len(item_text_terms.intersection(related_terms))
                if overlap_count:
                    boost += self.use_case_term_boost * float(overlap_count)

        boost += self.rating_boost * self.quality_scores.get(item_id, 0.0)
        boost += self.popularity_boost * self.popularity_scores.get(item_id, 0.0)
        return boost

    def search(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        """Search with BM25 scores plus deterministic query-aware boosts."""
        scores = self._score_query(query)
        query_terms = set(tokenize(query))
        limit = max(0, min(int(top_k), len(self.item_ids)))
        rows = []
        for index, item_id in enumerate(self.item_ids):
            item_row = self.item_metadata.iloc[index]
            bm25_score = float(scores[index])
            boost = self._metadata_boost(query_terms, item_id, item_row)
            rows.append(
                {
                    "item_id": item_id,
                    "score": bm25_score + boost,
                    "bm25_score": bm25_score,
                    "metadata_boost": float(boost),
                }
            )

        ranked = sorted(rows, key=lambda row: (-row["score"], row["item_id"]))[:limit]
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        return ranked

    def save(self, path: str | Path) -> None:
        """Persist the query-aware index with pickle."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> QueryAwareBM25Retriever:
        """Load a persisted query-aware index."""
        with Path(path).open("rb") as file:
            loaded = pickle.load(file)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded
