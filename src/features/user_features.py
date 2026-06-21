"""Lightweight user history feature helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_user_item_history(interactions: pd.DataFrame) -> dict[str, set[str]]:
    """Build a mapping from user ID to interacted item IDs."""
    if interactions.empty:
        return {}
    working = interactions.copy()
    working["user_id"] = working["user_id"].astype(str)
    working["item_id"] = working["item_id"].astype(str)
    return working.groupby("user_id")["item_id"].apply(lambda values: set(values)).to_dict()


def build_user_event_weights(interactions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Build summed event weights for each user's interacted items."""
    if interactions.empty:
        return {}
    working = interactions.copy()
    working["user_id"] = working["user_id"].astype(str)
    working["item_id"] = working["item_id"].astype(str)
    if "event_weight" not in working.columns:
        working["event_weight"] = 1.0
    working["event_weight"] = pd.to_numeric(working["event_weight"], errors="coerce").fillna(1.0)
    grouped = working.groupby(["user_id", "item_id"], as_index=False)["event_weight"].sum()
    histories: dict[str, dict[str, float]] = {}
    for _, row in grouped.iterrows():
        histories.setdefault(str(row["user_id"]), {})[str(row["item_id"])] = float(
            row["event_weight"]
        )
    return histories


def get_seen_items(user_id: str, user_history: dict[str, set[str]]) -> set[str]:
    """Return seen items for a user, or an empty set for unknown users."""
    return set(user_history.get(str(user_id), set()))


def build_user_profile_summary(
    user_id: str,
    interactions: pd.DataFrame,
    items: pd.DataFrame,
) -> dict[str, Any]:
    """Build a compact, non-LLM summary of a user's interaction history."""
    user_id = str(user_id)
    if interactions.empty:
        return {
            "user_id": user_id,
            "num_interactions": 0,
            "num_unique_items": 0,
            "top_categories": [],
            "top_brands": [],
            "avg_event_weight": 0.0,
        }

    user_interactions = interactions[interactions["user_id"].astype(str) == user_id].copy()
    if user_interactions.empty:
        return {
            "user_id": user_id,
            "num_interactions": 0,
            "num_unique_items": 0,
            "top_categories": [],
            "top_brands": [],
            "avg_event_weight": 0.0,
        }

    merged = user_interactions.merge(items, on="item_id", how="left")
    event_weights = pd.to_numeric(merged.get("event_weight", 1.0), errors="coerce").fillna(1.0)
    return {
        "user_id": user_id,
        "num_interactions": int(len(user_interactions)),
        "num_unique_items": int(user_interactions["item_id"].nunique()),
        "top_categories": (
            merged["category"].dropna().astype(str).value_counts().head(3).index.tolist()
        ),
        "top_brands": merged["brand"].dropna().astype(str).value_counts().head(3).index.tolist(),
        "avg_event_weight": float(event_weights.mean()),
    }
