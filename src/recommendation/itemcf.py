"""Item-based collaborative filtering baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.user_features import build_user_event_weights
from src.recommendation.base import BaseRecommender
from src.recommendation.popularity import PopularityRecommender


class ItemCFRecommender(BaseRecommender):
    """Item-based collaborative filtering with cosine similarity."""

    method_name = "itemcf"

    def __init__(
        self,
        similarity: str = "cosine",
        min_cooccurrence: int = 1,
        max_neighbors: int = 50,
    ) -> None:
        super().__init__()
        self.similarity = similarity
        self.min_cooccurrence = int(min_cooccurrence)
        self.max_neighbors = int(max_neighbors)
        self.item_index: dict[str, int] = {}
        self.user_event_weights: dict[str, dict[str, float]] = {}
        self.similarity_matrix = np.zeros((0, 0), dtype=float)
        self.popularity = PopularityRecommender()

    def fit(
        self,
        train_interactions: pd.DataFrame,
        items: pd.DataFrame | None = None,
    ) -> ItemCFRecommender:
        """Fit item-item cosine similarities from implicit feedback."""
        if self.similarity != "cosine":
            raise ValueError("Stage 4 ItemCFRecommender supports similarity='cosine' only")

        self._prepare_common_state(train_interactions, items)
        self.popularity.fit(train_interactions, items)
        self.user_event_weights = build_user_event_weights(train_interactions)
        self.item_index = {item_id: index for index, item_id in enumerate(self.all_item_ids)}
        user_ids = sorted(self.user_event_weights)
        matrix = np.zeros((len(user_ids), len(self.all_item_ids)), dtype=float)
        for user_index, user_id in enumerate(user_ids):
            for item_id, weight in self.user_event_weights[user_id].items():
                item_index = self.item_index.get(item_id)
                if item_index is not None:
                    matrix[user_index, item_index] = float(weight)

        if matrix.size == 0:
            self.similarity_matrix = np.zeros((len(self.all_item_ids), len(self.all_item_ids)))
            return self

        item_matrix = matrix.T
        norms = np.linalg.norm(item_matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = item_matrix / norms
        similarity = normalized @ normalized.T
        cooccurrence = (matrix > 0).astype(float).T @ (matrix > 0).astype(float)
        similarity[cooccurrence < self.min_cooccurrence] = 0.0
        np.fill_diagonal(similarity, 0.0)

        if self.max_neighbors > 0 and self.max_neighbors < similarity.shape[0]:
            pruned = np.zeros_like(similarity)
            for column_index in range(similarity.shape[1]):
                column = similarity[:, column_index]
                top_indices = np.argsort(-column)[: self.max_neighbors]
                pruned[top_indices, column_index] = column[top_indices]
            similarity = pruned

        self.similarity_matrix = similarity
        return self

    def _score_user(self, user_id: str) -> dict[str, float]:
        weights = self.user_event_weights.get(str(user_id))
        if not weights:
            return self.popularity.item_scores

        scores = np.zeros(len(self.all_item_ids), dtype=float)
        for seen_item_id, weight in weights.items():
            seen_index = self.item_index.get(seen_item_id)
            if seen_index is not None:
                scores += self.similarity_matrix[:, seen_index] * float(weight)

        if not np.any(scores):
            return self.popularity.item_scores
        return {item_id: float(scores[index]) for index, item_id in enumerate(self.all_item_ids)}

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> list[dict[str, Any]]:
        """Recommend items using item-item similarities."""
        return self._rank_scores(
            user_id=str(user_id),
            item_scores=self._score_user(str(user_id)),
            top_k=top_k,
            exclude_seen=exclude_seen,
        )
