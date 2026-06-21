"""Popularity recommendation baseline."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.recommendation.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Recommend globally popular items from training interactions."""

    method_name = "popularity"

    def __init__(self, score_column: str = "event_weight") -> None:
        super().__init__()
        self.score_column = score_column
        self.item_scores: dict[str, float] = {}

    def fit(
        self,
        train_interactions: pd.DataFrame,
        items: pd.DataFrame | None = None,
    ) -> PopularityRecommender:
        """Fit global item popularity scores."""
        self._prepare_common_state(train_interactions, items)
        working = train_interactions.copy()
        if working.empty:
            self.item_scores = {item_id: 0.0 for item_id in self.all_item_ids}
            return self

        working["item_id"] = working["item_id"].astype(str)
        if self.score_column in working.columns:
            working[self.score_column] = pd.to_numeric(
                working[self.score_column],
                errors="coerce",
            ).fillna(1.0)
            scores = working.groupby("item_id")[self.score_column].sum()
        else:
            scores = working.groupby("item_id").size()

        self.item_scores = {
            item_id: float(scores.get(item_id, 0.0)) for item_id in self.all_item_ids
        }
        return self

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> list[dict[str, Any]]:
        """Recommend globally popular items."""
        return self._rank_scores(
            user_id=str(user_id),
            item_scores=self.item_scores,
            top_k=top_k,
            exclude_seen=exclude_seen,
        )
