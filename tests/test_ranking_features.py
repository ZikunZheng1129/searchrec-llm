from __future__ import annotations

import numpy as np
import pandas as pd

from src.ranking.features import (
    build_ranking_features,
    compute_authority_score,
    compute_business_proxy_features,
)


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id": "item_1",
                "title": "Aster wireless earbuds",
                "category": "Electronics",
                "brand": "Aster",
                "price": 99.0,
                "avg_rating": 4.5,
                "rating_count": 100,
                "description": "Clear audio for daily listening",
            },
            {
                "item_id": "item_2",
                "title": "Pulse yoga mat",
                "category": "Sports",
                "brand": "Pulse",
                "price": 30.0,
                "avg_rating": 4.0,
                "rating_count": 0,
                "description": "Cushioned mat for stretching",
            },
            {
                "item_id": "item_3",
                "title": "Haven face serum",
                "category": "Beauty",
                "brand": "Haven",
                "price": 20.0,
                "avg_rating": 4.8,
                "rating_count": 500,
                "description": "Gentle hydrating skincare serum",
            },
        ]
    )


def _queries() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": "q1",
                "query_text": "wireless aster audio",
                "target_item_id": "item_1",
                "split": "test",
            },
            {
                "query_id": "q2",
                "query_text": "hydrating serum",
                "target_item_id": "item_3",
                "split": "train",
            },
        ]
    )


def _interactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "item_1", "event_type": "purchase", "event_weight": 4.0},
            {
                "user_id": "u2",
                "item_id": "item_1",
                "event_type": "add_to_cart",
                "event_weight": 3.0,
            },
            {"user_id": "u3", "item_id": "item_2", "event_type": "view", "event_weight": 1.0},
        ]
    )


def _config() -> dict:
    return {
        "candidate_generation": {"ensure_positive_candidate": True},
        "retrieval": {
            "bm25": {"k1": 1.5, "b": 0.75},
            "dense": {"normalize": True, "max_features": 100},
            "hybrid": {
                "bm25_weight": 0.5,
                "dense_weight": 0.5,
                "score_normalization": "minmax",
            },
        },
    }


def test_ranking_feature_generation_required_behavior() -> None:
    candidates = build_ranking_features(
        query_item_pairs=_queries(),
        items=_items(),
        train_interactions=_interactions(),
        candidate_pool_size=2,
        random_negatives_per_query=1,
        seed=7,
        config=_config(),
    )

    assert not candidates.empty
    required = {
        "query_id",
        "candidate_item_id",
        "label",
        "hybrid_score",
        "title_token_overlap",
        "authority_score",
        "conversion_proxy",
        "cold_start_score",
    }
    assert required.issubset(candidates.columns)
    assert candidates.groupby("query_id")["label"].max().eq(1).all()
    assert not candidates.duplicated(["query_id", "candidate_item_id"]).any()
    for _, row in candidates.iterrows():
        assert row["label"] == int(row["candidate_item_id"] == row["target_item_id"])


def test_ranking_feature_generation_is_deterministic() -> None:
    first = build_ranking_features(_queries(), _items(), _interactions(), 2, 1, 42, _config())
    second = build_ranking_features(_queries(), _items(), _interactions(), 2, 1, 42, _config())
    pd.testing.assert_frame_equal(first, second)


def test_business_proxy_features_are_finite() -> None:
    proxies = compute_business_proxy_features(_interactions(), _items())
    assert np.isfinite(proxies.select_dtypes(include="number").to_numpy()).all()
    assert 0.0 <= compute_authority_score(4.5, 100) <= 1.0
