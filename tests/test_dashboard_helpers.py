from __future__ import annotations

import pandas as pd

from app.dashboard.data_loader import artifact_status, load_parquet_if_exists
from app.dashboard.demo_helpers import (
    build_business_metric_summary,
    find_query_examples,
    format_item_card,
    get_command_hints_for_missing_artifacts,
    get_query_candidates,
    get_user_examples,
    summarize_leaderboard,
)


def test_loader_missing_file_and_artifact_status(tmp_path):
    missing = tmp_path / "missing.parquet"
    assert load_parquet_if_exists(missing) is None
    status = artifact_status({"artifacts": {"item_metadata_path": str(missing)}})
    assert list(status.columns) == ["name", "path", "exists", "command_hint"]
    assert get_command_hints_for_missing_artifacts(status)


def test_query_and_user_helpers():
    pairs = pd.DataFrame({"query_id": ["q1"], "query_text": ["gift beauty"]})
    profiles = pd.DataFrame({"user_id": ["user_1"]})
    assert find_query_examples(pairs) == ["gift beauty"]
    assert get_user_examples(profiles) == ["user_1"]


def test_candidate_leaderboard_business_and_item_helpers():
    pairs = pd.DataFrame({"query_id": ["q1"], "query_text": ["gift beauty"], "split": ["test"]})
    candidates = pd.DataFrame(
        {
            "query_id": ["q1"],
            "candidate_item_id": ["item_1"],
            "multimodal_score": [0.9],
        }
    )
    artifacts = {
        "query_item_pairs_path": pairs,
        "ranking_candidates_multimodal_path": candidates,
        "item_metadata_path": pd.DataFrame(
            {
                "item_id": ["item_1"],
                "category": ["Beauty"],
                "avg_rating": [4.5],
                "rating_count": [100],
            }
        ),
    }
    assert get_query_candidates("gift beauty", artifacts, top_k=1)["candidate_item_id"].iloc[0]
    leaderboard = pd.DataFrame(
        {
            "stage": ["retrieval", "retrieval"],
            "method": ["a", "b"],
            "primary_score": [0.1, 0.2],
            "secondary_score": [0.0, 0.0],
            "coverage_score": [1.0, 1.0],
            "avg_latency_ms": [1.0, 2.0],
        }
    )
    assert summarize_leaderboard(leaderboard)["method"].iloc[0] == "b"
    metrics = build_business_metric_summary(artifacts)
    assert metrics["num_items"] == 1
    assert format_item_card({"candidate_item_id": "item_1"})["item_id"] == "item_1"
