"""Cold-start and long-tail slice utilities."""

from __future__ import annotations

import pandas as pd


def compute_item_interaction_counts(train_interactions: pd.DataFrame) -> pd.DataFrame:
    """Compute train interaction counts by item."""
    if train_interactions.empty:
        return pd.DataFrame(columns=["item_id", "interaction_count"])
    counts = (
        train_interactions.assign(item_id=train_interactions["item_id"].astype(str))
        .groupby("item_id", sort=True)
        .size()
        .rename("interaction_count")
        .reset_index()
    )
    return counts


def _bottom_quantile_items(
    items: pd.DataFrame,
    train_interactions: pd.DataFrame,
    quantile: float,
) -> set[str]:
    item_ids = items["item_id"].astype(str).sort_values().tolist()
    if not item_ids:
        return set()
    counts = compute_item_interaction_counts(train_interactions)
    table = (
        pd.DataFrame({"item_id": item_ids})
        .merge(counts, on="item_id", how="left")
        .fillna({"interaction_count": 0})
    )
    threshold = table["interaction_count"].quantile(float(quantile))
    selected = table[table["interaction_count"] <= threshold]["item_id"].astype(str).tolist()
    if not selected:
        min_count = table["interaction_count"].min()
        selected = table[table["interaction_count"] == min_count]["item_id"].astype(str).tolist()
    return set(selected)


def identify_cold_start_items(
    items: pd.DataFrame,
    train_interactions: pd.DataFrame,
    quantile: float = 0.25,
) -> set[str]:
    """Identify a simulated cold-start slice from low interaction counts."""
    return _bottom_quantile_items(items, train_interactions, quantile)


def identify_long_tail_items(
    items: pd.DataFrame,
    train_interactions: pd.DataFrame,
    quantile: float = 0.25,
) -> set[str]:
    """Identify a synthetic long-tail slice from low interaction counts."""
    return _bottom_quantile_items(items, train_interactions, quantile)


def add_cold_start_flags(
    items: pd.DataFrame,
    train_interactions: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Add cold-start and long-tail flags to item metadata."""
    evaluation_config = config.get("evaluation", {})
    cold = identify_cold_start_items(
        items,
        train_interactions,
        float(evaluation_config.get("cold_start_quantile", 0.25)),
    )
    long_tail = identify_long_tail_items(
        items,
        train_interactions,
        float(evaluation_config.get("long_tail_quantile", 0.25)),
    )
    output = items.copy()
    output["item_id"] = output["item_id"].astype(str)
    output["is_cold_start"] = output["item_id"].isin(cold)
    output["is_long_tail"] = output["item_id"].isin(long_tail)
    return output
