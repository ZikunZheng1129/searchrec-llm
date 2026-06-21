"""Base interface and shared helpers for recommender baselines."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.features.user_features import build_user_item_history


class BaseRecommender:
    """Small recommender interface used by Stage 4 baselines."""

    method_name = "base"

    def __init__(self) -> None:
        self.all_item_ids: list[str] = []
        self.user_history: dict[str, set[str]] = {}

    def fit(
        self,
        train_interactions: pd.DataFrame,
        items: pd.DataFrame | None = None,
    ) -> BaseRecommender:
        """Fit the recommender."""
        raise NotImplementedError

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> list[dict[str, Any]]:
        """Recommend items for one user."""
        raise NotImplementedError

    def batch_recommend(
        self,
        user_ids: list[str],
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> dict[str, list[dict[str, Any]]]:
        """Recommend items for several users."""
        return {
            str(user_id): self.recommend(str(user_id), top_k=top_k, exclude_seen=exclude_seen)
            for user_id in user_ids
        }

    def _prepare_common_state(
        self,
        train_interactions: pd.DataFrame,
        items: pd.DataFrame | None = None,
    ) -> None:
        working = train_interactions.copy()
        if not working.empty:
            working["user_id"] = working["user_id"].astype(str)
            working["item_id"] = working["item_id"].astype(str)
        self.user_history = build_user_item_history(working)

        if items is not None and "item_id" in items.columns:
            self.all_item_ids = sorted(items["item_id"].dropna().astype(str).unique().tolist())
        elif not working.empty:
            self.all_item_ids = sorted(working["item_id"].dropna().astype(str).unique().tolist())
        else:
            self.all_item_ids = []

    def _rank_scores(
        self,
        user_id: str,
        item_scores: dict[str, float],
        top_k: int,
        exclude_seen: bool,
    ) -> list[dict[str, Any]]:
        seen_items = self.user_history.get(str(user_id), set()) if exclude_seen else set()
        scored_items = []
        for item_id in self.all_item_ids:
            if item_id in seen_items:
                continue
            scored_items.append((item_id, float(item_scores.get(item_id, 0.0))))

        limit = max(0, min(int(top_k), len(scored_items)))
        ranked = sorted(scored_items, key=lambda row: (-row[1], row[0]))[:limit]
        return [
            {"item_id": item_id, "score": float(score), "rank": rank}
            for rank, (item_id, score) in enumerate(ranked, start=1)
        ]
