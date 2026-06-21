"""Average user-history embedding recommendation baseline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.features.user_features import build_user_event_weights
from src.recommendation.base import BaseRecommender
from src.recommendation.popularity import PopularityRecommender
from src.retrieval.dense_retriever import TfidfDenseRetriever
from src.retrieval.text_utils import DEFAULT_ITEM_TEXT_FIELDS


class UserHistoryEmbeddingRecommender(BaseRecommender):
    """Recommend items similar to the average TF-IDF vector of a user's history."""

    method_name = "user_history_embedding"

    def __init__(
        self,
        text_fields: Sequence[str] | None = None,
        max_features: int = 5000,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.text_fields = list(text_fields or DEFAULT_ITEM_TEXT_FIELDS)
        self.max_features = int(max_features)
        self.normalize = bool(normalize)
        self.vectorizer = TfidfDenseRetriever(
            max_features=self.max_features,
            normalize=self.normalize,
            item_text_fields=self.text_fields,
        )
        self.item_index: dict[str, int] = {}
        self.user_profiles: dict[str, np.ndarray] = {}
        self.user_event_weights: dict[str, dict[str, float]] = {}
        self.popularity = PopularityRecommender()

    def fit(
        self,
        train_interactions: pd.DataFrame,
        items: pd.DataFrame | None = None,
    ) -> UserHistoryEmbeddingRecommender:
        """Fit item text vectors and average user history profiles."""
        if items is None:
            raise ValueError("UserHistoryEmbeddingRecommender requires item metadata")

        self._prepare_common_state(train_interactions, items)
        self.popularity.fit(train_interactions, items)
        self.vectorizer.fit(items)
        self.all_item_ids = list(self.vectorizer.item_ids)
        self.item_index = {item_id: index for index, item_id in enumerate(self.all_item_ids)}
        self.user_event_weights = build_user_event_weights(train_interactions)
        self.user_profiles = {}

        for user_id, item_weights in self.user_event_weights.items():
            vectors = []
            weights = []
            for item_id, weight in item_weights.items():
                item_index = self.item_index.get(item_id)
                if item_index is None:
                    continue
                vectors.append(self.vectorizer.item_vectors[item_index])
                weights.append(float(weight))
            if not vectors:
                continue
            matrix = np.vstack(vectors)
            weight_array = np.asarray(weights, dtype=float)
            profile = np.average(matrix, axis=0, weights=weight_array)
            norm = np.linalg.norm(profile)
            if norm > 0:
                profile = profile / norm
            self.user_profiles[user_id] = profile

        return self

    def _score_user(self, user_id: str) -> dict[str, float]:
        profile = self.user_profiles.get(str(user_id))
        if profile is None or self.vectorizer.item_vectors.size == 0:
            return self.popularity.item_scores
        scores = self.vectorizer.item_vectors @ profile
        return {item_id: float(scores[index]) for index, item_id in enumerate(self.all_item_ids)}

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> list[dict[str, Any]]:
        """Recommend items by user-profile cosine similarity."""
        return self._rank_scores(
            user_id=str(user_id),
            item_scores=self._score_user(str(user_id)),
            top_k=top_k,
            exclude_seen=exclude_seen,
        )
