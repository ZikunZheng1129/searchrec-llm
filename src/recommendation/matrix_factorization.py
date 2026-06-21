"""Lightweight NumPy matrix factorization baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.user_features import build_user_event_weights
from src.recommendation.base import BaseRecommender
from src.recommendation.popularity import PopularityRecommender


class MatrixFactorizationRecommender(BaseRecommender):
    """Simple implicit-feedback matrix factorization trained with SGD."""

    method_name = "matrix_factorization"

    def __init__(
        self,
        factors: int = 16,
        epochs: int = 20,
        learning_rate: float = 0.05,
        regularization: float = 0.01,
        negative_samples: int = 3,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.factors = int(factors)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.negative_samples = int(negative_samples)
        self.seed = int(seed)
        self.user_index: dict[str, int] = {}
        self.item_index: dict[str, int] = {}
        self.user_event_weights: dict[str, dict[str, float]] = {}
        self.user_factors = np.zeros((0, 0), dtype=float)
        self.item_factors = np.zeros((0, 0), dtype=float)
        self.popularity = PopularityRecommender()

    def fit(
        self,
        train_interactions: pd.DataFrame,
        items: pd.DataFrame | None = None,
    ) -> MatrixFactorizationRecommender:
        """Fit user and item latent factors using positive and sampled negative pairs."""
        self._prepare_common_state(train_interactions, items)
        self.popularity.fit(train_interactions, items)
        self.user_event_weights = build_user_event_weights(train_interactions)
        user_ids = sorted(self.user_event_weights)
        self.user_index = {user_id: index for index, user_id in enumerate(user_ids)}
        self.item_index = {item_id: index for index, item_id in enumerate(self.all_item_ids)}

        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0.0, 0.05, size=(len(user_ids), self.factors))
        self.item_factors = rng.normal(0.0, 0.05, size=(len(self.all_item_ids), self.factors))

        positive_pairs = [
            (self.user_index[user_id], self.item_index[item_id])
            for user_id, item_weights in self.user_event_weights.items()
            for item_id in item_weights
            if user_id in self.user_index and item_id in self.item_index
        ]
        if not positive_pairs or not self.all_item_ids:
            return self

        all_item_indices = np.arange(len(self.all_item_ids))
        seen_indices_by_user = {
            self.user_index[user_id]: {self.item_index[item_id] for item_id in item_weights}
            for user_id, item_weights in self.user_event_weights.items()
            if user_id in self.user_index
        }

        for _ in range(self.epochs):
            order = np.arange(len(positive_pairs))
            rng.shuffle(order)
            for pair_index in order:
                user_idx, item_idx = positive_pairs[int(pair_index)]
                self._sgd_update(user_idx, item_idx, label=1.0)
                negatives = self._sample_negative_indices(
                    rng=rng,
                    all_item_indices=all_item_indices,
                    seen_indices=seen_indices_by_user.get(user_idx, set()),
                )
                for negative_item_idx in negatives:
                    self._sgd_update(user_idx, int(negative_item_idx), label=0.0)

        return self

    def _sample_negative_indices(
        self,
        rng: np.random.Generator,
        all_item_indices: np.ndarray,
        seen_indices: set[int],
    ) -> np.ndarray:
        candidates = np.array(
            [item_idx for item_idx in all_item_indices.tolist() if item_idx not in seen_indices],
            dtype=int,
        )
        if len(candidates) == 0 or self.negative_samples <= 0:
            return np.array([], dtype=int)
        sample_size = min(self.negative_samples, len(candidates))
        return rng.choice(candidates, size=sample_size, replace=False)

    def _sgd_update(self, user_idx: int, item_idx: int, label: float) -> None:
        user_vector = self.user_factors[user_idx].copy()
        item_vector = self.item_factors[item_idx].copy()
        prediction = float(user_vector @ item_vector)
        error = float(label - prediction)
        self.user_factors[user_idx] += self.learning_rate * (
            error * item_vector - self.regularization * user_vector
        )
        self.item_factors[item_idx] += self.learning_rate * (
            error * user_vector - self.regularization * item_vector
        )

    def _score_user(self, user_id: str) -> dict[str, float]:
        user_idx = self.user_index.get(str(user_id))
        if user_idx is None or self.item_factors.size == 0:
            return self.popularity.item_scores
        scores = self.item_factors @ self.user_factors[user_idx]
        return {item_id: float(scores[index]) for index, item_id in enumerate(self.all_item_ids)}

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> list[dict[str, Any]]:
        """Recommend items from latent-factor scores."""
        return self._rank_scores(
            user_id=str(user_id),
            item_scores=self._score_user(str(user_id)),
            top_k=top_k,
            exclude_seen=exclude_seen,
        )
