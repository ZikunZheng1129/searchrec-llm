"""Lightweight schema helpers for Stage 1 datasets."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

USER_COLUMNS = ["user_id", "user_age_bucket", "user_region"]
ITEM_COLUMNS = [
    "item_id",
    "title",
    "category",
    "brand",
    "price",
    "avg_rating",
    "rating_count",
    "description",
]
INTERACTION_COLUMNS = [
    "user_id",
    "item_id",
    "timestamp",
    "event_type",
    "rating",
    "event_weight",
]


def validate_columns(df: pd.DataFrame, required_columns: Sequence[str], name: str) -> None:
    """Validate that a DataFrame contains the required columns."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{name} is missing required columns: {missing_text}")


def validate_interactions_schema(interactions: pd.DataFrame) -> None:
    """Validate Stage 1 interaction columns."""
    validate_columns(interactions, INTERACTION_COLUMNS, "interactions")


def validate_items_schema(items: pd.DataFrame) -> None:
    """Validate Stage 1 item metadata columns."""
    validate_columns(items, ITEM_COLUMNS, "items")


def validate_users_schema(users: pd.DataFrame) -> None:
    """Validate Stage 1 user columns."""
    validate_columns(users, USER_COLUMNS, "users")


def attach_split_column(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Attach split labels and return one combined interactions DataFrame."""
    split_frames = []
    for split_name, frame in (("train", train), ("val", val), ("test", test)):
        split_frame = frame.copy()
        split_frame["split"] = split_name
        split_frames.append(split_frame)

    if not split_frames:
        return pd.DataFrame()

    return pd.concat(split_frames, ignore_index=True)
