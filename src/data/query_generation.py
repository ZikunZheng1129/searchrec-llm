"""Synthetic query generation from item metadata."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np
import pandas as pd

from src.query_understanding.intent_classifier import classify_intent
from src.query_understanding.query_parser import normalize_query

REQUIRED_QUERY_COLUMNS = [
    "query_id",
    "query_text",
    "target_item_id",
    "category",
    "intent",
    "source",
]
OPTIONAL_QUERY_COLUMNS = [
    "brand",
    "price_constraint",
    "use_case",
    "relevance_label",
    "split",
]

USE_CASE_BY_CATEGORY = {
    "sports & outdoors": ["running", "outdoor", "beginner"],
    "beauty": ["dry skin", "gift", "beginner"],
    "electronics": ["travel", "office", "school"],
    "clothing": ["walking", "office", "travel"],
    "health & personal care": ["gym", "travel", "gift"],
}


def _stable_seed(seed: int, item_id: str) -> int:
    digest = hashlib.md5(f"{seed}:{item_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", normalize_query(value)).strip("_")
    return slug or "query"


def _category_label(category: str) -> str:
    return normalize_query(category).replace("&", "and")


def _price_constraint(price: float) -> str:
    if price < 30:
        return "budget"
    if price >= 120:
        return "premium"
    return "affordable"


def _candidate_queries(item: pd.Series, config: dict[str, Any]) -> list[dict[str, Any]]:
    item_id = str(item["item_id"])
    title = normalize_query(str(item.get("title", "")))
    category = str(item.get("category", ""))
    normalized_category = _category_label(category)
    brand = str(item.get("brand", ""))
    normalized_brand = normalize_query(brand)
    price = float(item.get("price", 0.0))
    price_constraint = _price_constraint(price)
    use_cases = USE_CASE_BY_CATEGORY.get(normalized_category, ["gift", "travel", "beginner"])
    rating = float(item.get("avg_rating", 0.0))

    candidates: list[dict[str, Any]] = []

    def add(
        query_text: str,
        source: str,
        *,
        use_case: str | None = None,
        price_value: str | None = None,
    ) -> None:
        normalized_query = normalize_query(query_text)
        if not normalized_query:
            return
        candidates.append(
            {
                "query_text": normalized_query,
                "target_item_id": item_id,
                "category": category,
                "intent": classify_intent(normalized_query),
                "source": source,
                "brand": brand,
                "price_constraint": price_value,
                "use_case": use_case,
                "relevance_label": 1,
            }
        )

    if config.get("include_title_queries", True):
        add(title, "title")

    if config.get("include_category_queries", True):
        add(f"{normalized_category} products", "category")

    if config.get("include_brand_queries", True):
        add(f"{normalized_brand} {normalized_category} product", "brand_category")

    if config.get("include_use_case_queries", True):
        for use_case in use_cases:
            add(f"{use_case} {normalized_category}", "use_case", use_case=use_case)

    if config.get("include_price_queries", True):
        add(
            f"{price_constraint} {normalized_category}",
            "price_category",
            price_value=price_constraint,
        )

    descriptive_prefix = "high rated" if rating >= 4.4 else "popular"
    add(f"{descriptive_prefix} {normalized_category} item", "descriptive")

    deduped = {}
    for candidate in candidates:
        key = (candidate["query_text"], candidate["source"])
        deduped[key] = candidate
    return list(deduped.values())


def generate_queries_for_item(
    item: pd.Series,
    queries_per_item: int = 3,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate deterministic synthetic query dictionaries for one item."""
    item_id = str(item["item_id"])
    config = {
        "include_title_queries": True,
        "include_category_queries": True,
        "include_brand_queries": True,
        "include_use_case_queries": True,
        "include_price_queries": True,
    }
    candidates = _candidate_queries(item, config)
    rng = np.random.default_rng(_stable_seed(seed, item_id))
    order = rng.permutation(len(candidates)) if candidates else []
    selected = [candidates[index] for index in order[: max(0, queries_per_item)]]

    for index, query in enumerate(selected):
        query["query_id"] = f"q_{_slug(item_id)}_{index + 1:02d}_{_slug(query['source'])}"

    return selected


def generate_query_item_pairs(
    items: pd.DataFrame,
    config: dict[str, Any],
    seed: int = 42,
) -> pd.DataFrame:
    """Generate positive query-item relevance pairs from item metadata."""
    if items.empty:
        return pd.DataFrame(columns=REQUIRED_QUERY_COLUMNS + OPTIONAL_QUERY_COLUMNS)

    query_config = config.get("query_generation", config)
    queries_per_item = int(query_config.get("queries_per_item", 3))
    rows = []

    for _, item in items.sort_values("item_id").iterrows():
        candidates = _candidate_queries(item, query_config)
        rng = np.random.default_rng(_stable_seed(seed, str(item["item_id"])))
        order = rng.permutation(len(candidates)) if candidates else []
        selected = [candidates[index] for index in order[: max(0, queries_per_item)]]
        for index, query in enumerate(selected):
            query = query.copy()
            query["query_id"] = (
                f"q_{_slug(str(item['item_id']))}_{index + 1:02d}_{_slug(query['source'])}"
            )
            rows.append(query)

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pd.DataFrame(columns=REQUIRED_QUERY_COLUMNS + OPTIONAL_QUERY_COLUMNS)

    pairs = add_query_splits(pairs, seed=seed)
    validate_query_item_pairs(pairs)
    return pairs


def add_query_splits(query_item_pairs: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Assign deterministic train, validation, and test splits."""
    if query_item_pairs.empty:
        output = query_item_pairs.copy()
        output["split"] = []
        return output

    rng = np.random.default_rng(seed)
    output = query_item_pairs.copy().sort_values("query_id").reset_index(drop=True)
    split_values = rng.choice(["train", "val", "test"], size=len(output), p=[0.8, 0.1, 0.1])
    output["split"] = split_values
    return output


def validate_query_item_pairs(query_item_pairs: pd.DataFrame) -> None:
    """Validate the Stage 2 query-item pair schema."""
    missing = [
        column
        for column in REQUIRED_QUERY_COLUMNS + OPTIONAL_QUERY_COLUMNS
        if column not in query_item_pairs.columns
    ]
    if missing:
        raise ValueError(f"query_item_pairs is missing required columns: {', '.join(missing)}")

    if query_item_pairs["query_id"].duplicated().any():
        raise ValueError("query_item_pairs contains duplicate query_id values")

    invalid_splits = set(query_item_pairs["split"].dropna().unique()) - {"train", "val", "test"}
    if invalid_splits:
        raise ValueError(
            f"query_item_pairs contains invalid split values: {sorted(invalid_splits)}"
        )

    invalid_labels = set(query_item_pairs["relevance_label"].dropna().unique()) - {1}
    if invalid_labels:
        raise ValueError("Stage 2 generated query-item pairs must have relevance_label=1")
