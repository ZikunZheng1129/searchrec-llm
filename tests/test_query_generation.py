import pandas as pd

from src.data.preprocess import generate_synthetic_items
from src.data.query_generation import (
    REQUIRED_QUERY_COLUMNS,
    generate_query_item_pairs,
    validate_query_item_pairs,
)


def _query_generation_config() -> dict:
    return {
        "query_generation": {
            "queries_per_item": 4,
            "include_title_queries": True,
            "include_category_queries": True,
            "include_brand_queries": True,
            "include_use_case_queries": True,
            "include_price_queries": True,
        }
    }


def test_generate_query_item_pairs_schema_and_values() -> None:
    items = generate_synthetic_items(
        num_items=12,
        categories=["Electronics", "Beauty", "Sports & Outdoors"],
        seed=21,
    )

    pairs = generate_query_item_pairs(items=items, config=_query_generation_config(), seed=21)

    assert not pairs.empty
    assert set(REQUIRED_QUERY_COLUMNS).issubset(pairs.columns)
    assert pairs["query_id"].is_unique
    assert set(pairs["target_item_id"]).issubset(set(items["item_id"]))
    assert set(pairs["relevance_label"]) == {1}
    assert set(pairs["split"]).issubset({"train", "val", "test"})
    validate_query_item_pairs(pairs)


def test_generate_query_item_pairs_is_deterministic() -> None:
    items = generate_synthetic_items(
        num_items=10,
        categories=["Electronics", "Beauty"],
        seed=33,
    )

    first = generate_query_item_pairs(items=items, config=_query_generation_config(), seed=33)
    second = generate_query_item_pairs(items=items, config=_query_generation_config(), seed=33)

    pd.testing.assert_frame_equal(first, second)
