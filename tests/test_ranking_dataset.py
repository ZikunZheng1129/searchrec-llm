from __future__ import annotations

import pandas as pd
import pytest

from src.ranking.dataset import (
    RankingFeatureDataset,
    get_numeric_feature_columns,
    standardize_features,
    validate_ranking_candidates,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": "q1",
                "query_text": "earbuds",
                "split": "train",
                "target_item_id": "i1",
                "candidate_item_id": "i1",
                "candidate_source": "hybrid",
                "label": 1,
                "hybrid_score": 0.9,
                "constant_feature": 1.0,
                "category": "Electronics",
            },
            {
                "query_id": "q1",
                "query_text": "earbuds",
                "split": "train",
                "target_item_id": "i1",
                "candidate_item_id": "i2",
                "candidate_source": "hybrid",
                "label": 0,
                "hybrid_score": 0.1,
                "constant_feature": 1.0,
                "category": "Beauty",
            },
        ]
    )


def test_validate_ranking_candidates_catches_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required"):
        validate_ranking_candidates(pd.DataFrame({"query_id": ["q1"]}))


def test_get_numeric_feature_columns_excludes_ids_text_and_label() -> None:
    columns = get_numeric_feature_columns(_candidates())
    assert "hybrid_score" in columns
    assert "constant_feature" in columns
    assert "label" not in columns
    assert "query_id" not in columns


def test_standardize_features_handles_zero_variance() -> None:
    standardized, state = standardize_features(_candidates(), _candidates())
    assert state["std"]["constant_feature"] == 1.0
    assert standardized["constant_feature"].eq(0.0).all()


def test_ranking_feature_dataset_returns_shapes() -> None:
    columns = get_numeric_feature_columns(_candidates())
    dataset = RankingFeatureDataset(_candidates(), columns)
    row = dataset[0]
    assert row["features"].shape[0] == len(columns)
    assert row["label"].shape == ()
    assert row["query_id"] == "q1"
