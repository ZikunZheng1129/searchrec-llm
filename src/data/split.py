"""Time-aware data splitting and sequence construction."""

from __future__ import annotations

import pandas as pd

from src.data.dataset import validate_interactions_schema


def leave_one_out_split(
    interactions: pd.DataFrame,
    val_last_n: int = 1,
    test_last_n: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split each user's interactions into train, validation, and test by time."""
    validate_interactions_schema(interactions)
    if interactions.empty:
        return interactions.copy(), interactions.copy(), interactions.copy()

    sorted_df = interactions.copy()
    sorted_df["timestamp"] = pd.to_datetime(sorted_df["timestamp"])
    sorted_df["_original_order"] = range(len(sorted_df))
    sorted_df = sorted_df.sort_values(["user_id", "timestamp", "_original_order"])

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    val_count = max(0, int(val_last_n))
    test_count = max(0, int(test_last_n))

    for _, group in sorted_df.groupby("user_id", sort=False):
        test_group = group.tail(test_count) if test_count else group.iloc[0:0]
        remaining = group.drop(index=test_group.index)
        val_group = remaining.tail(val_count) if val_count else remaining.iloc[0:0]
        train_group = remaining.drop(index=val_group.index)

        train_indices.extend(train_group.index.tolist())
        val_indices.extend(val_group.index.tolist())
        test_indices.extend(test_group.index.tolist())

    def finalize(indices: list[int]) -> pd.DataFrame:
        frame = sorted_df.loc[indices].drop(columns=["_original_order"])
        return frame.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)

    return finalize(train_indices), finalize(val_indices), finalize(test_indices)


def build_user_sequences(interactions: pd.DataFrame) -> pd.DataFrame:
    """Build one chronological item interaction sequence per user."""
    validate_interactions_schema(interactions)
    if interactions.empty:
        return pd.DataFrame(
            columns=[
                "user_id",
                "item_sequence",
                "timestamp_sequence",
                "event_type_sequence",
                "event_weight_sequence",
                "sequence_length",
            ]
        )

    sorted_df = interactions.copy()
    sorted_df["timestamp"] = pd.to_datetime(sorted_df["timestamp"])
    sorted_df = sorted_df.sort_values(["user_id", "timestamp", "item_id"])

    rows = []
    for user_id, group in sorted_df.groupby("user_id", sort=False):
        item_sequence = group["item_id"].tolist()
        rows.append(
            {
                "user_id": user_id,
                "item_sequence": item_sequence,
                "timestamp_sequence": [
                    timestamp.isoformat() for timestamp in group["timestamp"].tolist()
                ],
                "event_type_sequence": group["event_type"].tolist(),
                "event_weight_sequence": [float(value) for value in group["event_weight"].tolist()],
                "sequence_length": len(item_sequence),
            }
        )

    return pd.DataFrame(rows)
