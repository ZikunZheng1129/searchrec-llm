"""Synthetic debug data generation and cleaning."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.dataset import (
    validate_interactions_schema,
    validate_items_schema,
    validate_users_schema,
)

EVENT_WEIGHTS = {
    "view": 1.0,
    "click": 2.0,
    "add_to_cart": 3.0,
    "purchase": 4.0,
}

AGE_BUCKETS = ["18-24", "25-34", "35-44", "45-54", "55+"]
REGIONS = ["West", "Midwest", "Northeast", "South"]
BRANDS = ["Nova", "Orbit", "Luma", "Stride", "Haven", "Pulse", "Aster", "Kindlewood"]
CATEGORY_TERMS = {
    "Sports & Outdoors": [
        ("Trail Running Shoes", "Lightweight traction for daily runs and weekend hikes."),
        ("Insulated Water Bottle", "Keeps drinks cold during workouts and commutes."),
        ("Adjustable Yoga Mat", "Cushioned grip for stretching, balance, and recovery."),
    ],
    "Beauty": [
        ("Hydrating Face Serum", "A gentle serum for a smooth daily skincare routine."),
        ("Soft Matte Lip Tint", "Buildable color with a comfortable natural finish."),
        ("Daily Glow Moisturizer", "Light hydration for morning and evening use."),
    ],
    "Electronics": [
        ("Wireless Earbuds", "Clear audio and compact charging for everyday listening."),
        ("Portable Phone Stand", "Foldable desk stand for calls, videos, and recipes."),
        ("USB-C Charging Hub", "Compact expansion for laptops, tablets, and phones."),
    ],
    "Clothing": [
        ("Relaxed Cotton Hoodie", "Soft midweight fleece for casual everyday layering."),
        ("Stretch Travel Pants", "Wrinkle-resistant comfort for workdays and trips."),
        ("Classic Knit Beanie", "Warm ribbed knit style for cool weather."),
    ],
    "Health & Personal Care": [
        ("Electric Toothbrush", "Simple brushing modes with a compact travel case."),
        ("Aromatherapy Shower Steamers", "Refreshing shower tablets for a spa-like routine."),
        ("Digital Fitness Scale", "Slim profile scale for daily wellness tracking."),
    ],
}


def generate_synthetic_items(
    num_items: int,
    categories: list[str],
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic synthetic item metadata."""
    rng = np.random.default_rng(seed)
    rows = []
    category_choices = categories or list(CATEGORY_TERMS)

    for index in range(num_items):
        category = str(rng.choice(category_choices))
        terms = CATEGORY_TERMS.get(category, CATEGORY_TERMS["Electronics"])
        title_base, description_base = terms[int(rng.integers(0, len(terms)))]
        brand = str(rng.choice(BRANDS))
        item_number = index + 1

        rows.append(
            {
                "item_id": f"item_{item_number:05d}",
                "title": f"{brand} {title_base}",
                "category": category,
                "brand": brand,
                "price": round(float(rng.uniform(8.0, 220.0)), 2),
                "avg_rating": round(float(rng.uniform(3.2, 5.0)), 2),
                "rating_count": int(rng.integers(5, 2500)),
                "description": f"{description_base} Synthetic debug item #{item_number}.",
            }
        )

    return pd.DataFrame(rows)


def generate_synthetic_users(num_users: int, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic synthetic users."""
    rng = np.random.default_rng(seed)
    rows = []

    for index in range(num_users):
        rows.append(
            {
                "user_id": f"user_{index + 1:05d}",
                "user_age_bucket": str(rng.choice(AGE_BUCKETS)),
                "user_region": str(rng.choice(REGIONS)),
            }
        )

    return pd.DataFrame(rows)


def _random_timestamp(
    rng: np.random.Generator,
    start_timestamp: str,
    end_timestamp: str,
    offset_seconds: int,
) -> pd.Timestamp:
    start = pd.Timestamp(start_timestamp)
    end = pd.Timestamp(end_timestamp)
    total_seconds = max(1, int((end - start).total_seconds()))
    random_seconds = int(rng.integers(0, total_seconds))
    return start + pd.Timedelta(seconds=random_seconds + offset_seconds)


def _rating_for_event(event_type: str, rng: np.random.Generator) -> float:
    baseline = {
        "view": 2.0,
        "click": 3.0,
        "add_to_cart": 4.0,
        "purchase": 5.0,
    }[event_type]
    return float(np.clip(round(baseline + rng.normal(0, 0.35), 1), 1.0, 5.0))


def generate_synthetic_interactions(
    users: pd.DataFrame,
    items: pd.DataFrame,
    num_interactions: int,
    start_timestamp: str,
    end_timestamp: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic implicit-feedback interactions."""
    validate_users_schema(users)
    validate_items_schema(items)

    rng = np.random.default_rng(seed)
    user_ids = users["user_id"].tolist()
    item_ids = items["item_id"].tolist()
    if not user_ids or not item_ids or num_interactions <= 0:
        return pd.DataFrame(
            columns=["user_id", "item_id", "timestamp", "event_type", "rating", "event_weight"]
        )

    records: list[dict[str, Any]] = []
    event_types = np.array(list(EVENT_WEIGHTS))
    event_probs = np.array([0.55, 0.25, 0.12, 0.08])

    def append_record(user_id: str, item_id: str) -> None:
        event_type = str(rng.choice(event_types, p=event_probs))
        records.append(
            {
                "user_id": user_id,
                "item_id": item_id,
                "timestamp": _random_timestamp(
                    rng,
                    start_timestamp,
                    end_timestamp,
                    offset_seconds=len(records),
                ),
                "event_type": event_type,
                "rating": _rating_for_event(event_type, rng),
                "event_weight": EVENT_WEIGHTS[event_type],
            }
        )

    min_user_events = 3 if num_interactions >= len(user_ids) * 3 else 1
    for user_id in user_ids:
        for _ in range(min_user_events):
            if len(records) >= num_interactions:
                break
            append_record(user_id, str(rng.choice(item_ids)))

    seen_item_ids = {record["item_id"] for record in records}
    for item_id in item_ids:
        if len(records) >= num_interactions:
            break
        if item_id not in seen_item_ids:
            append_record(str(rng.choice(user_ids)), item_id)

    while len(records) < num_interactions:
        append_record(str(rng.choice(user_ids)), str(rng.choice(item_ids)))

    return pd.DataFrame(records)


def clean_items(items: pd.DataFrame) -> pd.DataFrame:
    """Clean item metadata and validate the expected schema."""
    validate_items_schema(items)
    cleaned = items.copy()
    cleaned = cleaned.dropna(subset=["item_id"]).drop_duplicates(subset=["item_id"])
    cleaned["item_id"] = cleaned["item_id"].astype(str)
    cleaned["title"] = cleaned["title"].fillna("").astype(str)
    cleaned["category"] = cleaned["category"].fillna("Unknown").astype(str)
    cleaned["brand"] = cleaned["brand"].fillna("Unknown").astype(str)
    cleaned["description"] = cleaned["description"].fillna("").astype(str)
    cleaned["price"] = pd.to_numeric(cleaned["price"], errors="coerce").fillna(0.0).round(2)
    cleaned["avg_rating"] = pd.to_numeric(cleaned["avg_rating"], errors="coerce").fillna(0.0)
    cleaned["rating_count"] = (
        pd.to_numeric(cleaned["rating_count"], errors="coerce").fillna(0).astype(int)
    )
    validate_items_schema(cleaned)
    return cleaned.reset_index(drop=True)


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    """Clean users and validate the expected schema."""
    validate_users_schema(users)
    cleaned = users.copy()
    cleaned = cleaned.dropna(subset=["user_id"]).drop_duplicates(subset=["user_id"])
    cleaned["user_id"] = cleaned["user_id"].astype(str)
    cleaned["user_age_bucket"] = cleaned["user_age_bucket"].fillna("Unknown").astype(str)
    cleaned["user_region"] = cleaned["user_region"].fillna("Unknown").astype(str)
    validate_users_schema(cleaned)
    return cleaned.reset_index(drop=True)


def clean_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    """Clean interactions and validate the expected schema."""
    validate_interactions_schema(interactions)
    cleaned = interactions.copy()
    cleaned = cleaned.dropna(subset=["user_id", "item_id", "timestamp", "event_type"])
    cleaned["user_id"] = cleaned["user_id"].astype(str)
    cleaned["item_id"] = cleaned["item_id"].astype(str)
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
    cleaned = cleaned.dropna(subset=["timestamp"])
    cleaned["event_type"] = cleaned["event_type"].astype(str)
    cleaned["event_weight"] = (
        cleaned["event_type"].map(EVENT_WEIGHTS).fillna(cleaned["event_weight"])
    )
    cleaned["event_weight"] = pd.to_numeric(cleaned["event_weight"], errors="coerce").fillna(1.0)
    cleaned["rating"] = pd.to_numeric(cleaned["rating"], errors="coerce").fillna(0.0)
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    validate_interactions_schema(cleaned)
    return cleaned


def _apply_minimum_filters(
    users: pd.DataFrame,
    items: pd.DataFrame,
    interactions: pd.DataFrame,
    min_interactions_per_user: int,
    min_interactions_per_item: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    filtered = interactions.copy()

    if min_interactions_per_user > 1:
        user_counts = filtered["user_id"].value_counts()
        keep_users = user_counts[user_counts >= min_interactions_per_user].index
        filtered = filtered[filtered["user_id"].isin(keep_users)]

    if min_interactions_per_item > 1:
        item_counts = filtered["item_id"].value_counts()
        keep_items = item_counts[item_counts >= min_interactions_per_item].index
        filtered = filtered[filtered["item_id"].isin(keep_items)]

    users = users[users["user_id"].isin(filtered["user_id"].unique())].reset_index(drop=True)
    items = items[items["item_id"].isin(filtered["item_id"].unique())].reset_index(drop=True)
    filtered = filtered.reset_index(drop=True)
    return users, items, filtered


def build_debug_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build cleaned synthetic users, items, and interactions from a Stage 1 config."""
    seed = int(config.get("seed", 42))
    data_config = config.get("data", {})
    synthetic_config = config.get("synthetic", {})

    if data_config.get("mode") != "synthetic_debug":
        raise ValueError("Stage 1 currently supports only data.mode='synthetic_debug'")

    users = generate_synthetic_users(
        num_users=int(synthetic_config.get("num_users", 50)),
        seed=seed,
    )
    items = generate_synthetic_items(
        num_items=int(synthetic_config.get("num_items", 100)),
        categories=list(synthetic_config.get("categories", list(CATEGORY_TERMS))),
        seed=seed,
    )
    interactions = generate_synthetic_interactions(
        users=users,
        items=items,
        num_interactions=int(synthetic_config.get("num_interactions", 500)),
        start_timestamp=str(synthetic_config.get("start_timestamp", "2024-01-01")),
        end_timestamp=str(synthetic_config.get("end_timestamp", "2024-03-31")),
        seed=seed,
    )

    users = clean_users(users)
    items = clean_items(items)
    interactions = clean_interactions(interactions)

    return _apply_minimum_filters(
        users=users,
        items=items,
        interactions=interactions,
        min_interactions_per_user=int(data_config.get("min_interactions_per_user", 1)),
        min_interactions_per_item=int(data_config.get("min_interactions_per_item", 1)),
    )
