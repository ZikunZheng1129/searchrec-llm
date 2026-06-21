"""Negative sampling helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.dataset import validate_interactions_schema


def sample_negative_items(
    interactions: pd.DataFrame,
    all_item_ids: list[str],
    num_negatives_per_positive: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample items each user has not interacted with for every positive row."""
    validate_interactions_schema(interactions)
    columns = ["user_id", "positive_item_id", "negative_item_id", "split"]
    if interactions.empty or not all_item_ids or num_negatives_per_positive <= 0:
        return pd.DataFrame(columns=columns)

    rng = np.random.default_rng(seed)
    unique_item_ids = sorted({str(item_id) for item_id in all_item_ids})
    working = interactions.copy()
    working["user_id"] = working["user_id"].astype(str)
    working["item_id"] = working["item_id"].astype(str)

    interacted_by_user = (
        working.groupby("user_id")["item_id"].apply(lambda values: set(values.tolist())).to_dict()
    )
    records: list[dict[str, Any]] = []

    for _, row in working.iterrows():
        user_id = str(row["user_id"])
        interacted_items = interacted_by_user.get(user_id, set())
        candidates = [item_id for item_id in unique_item_ids if item_id not in interacted_items]
        if not candidates:
            continue

        sample_size = min(int(num_negatives_per_positive), len(candidates))
        sampled = rng.choice(candidates, size=sample_size, replace=False)
        split = str(row["split"]) if "split" in working.columns else "unknown"

        for negative_item_id in sampled.tolist():
            records.append(
                {
                    "user_id": user_id,
                    "positive_item_id": str(row["item_id"]),
                    "negative_item_id": str(negative_item_id),
                    "split": split,
                }
            )

    return pd.DataFrame(records, columns=columns)
