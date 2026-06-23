import pandas as pd
import pytest

from src.pipelines.run_retrieval_tuning_test import (
    build_comparison_rows,
    build_multi_relevant_metrics,
    build_query_type_breakdown,
    summarize_multi_relevant_metrics,
    target_rank,
)


def test_build_comparison_rows_reports_metric_deltas() -> None:
    benchmark = {
        "method": "bm25",
        "config_path": "benchmark.yaml",
        "split": "test",
        "num_queries": 2,
        "top_k": 50,
        "recall_at_50": 1.0,
        "mrr_at_10": 0.2,
        "avg_latency_ms": 0.1,
    }
    candidate = {
        "method": "hybrid",
        "config_path": "candidate.yaml",
        "split": "test",
        "num_queries": 2,
        "top_k": 50,
        "recall_at_50": 1.0,
        "mrr_at_10": 0.25,
        "avg_latency_ms": 0.15,
    }

    comparison = build_comparison_rows(
        benchmark,
        [candidate],
        metrics=["recall_at_50", "mrr_at_10", "avg_latency_ms"],
    )

    assert comparison.loc[0, "benchmark_method"] == "bm25"
    assert comparison.loc[0, "candidate_method"] == "hybrid"
    assert comparison.loc[0, "delta_recall_at_50"] == 0.0
    assert comparison.loc[0, "delta_mrr_at_10"] == pytest.approx(0.05)
    assert comparison.loc[0, "delta_avg_latency_ms"] == pytest.approx(0.05)


def test_target_rank_returns_one_based_rank_or_zero() -> None:
    assert target_rank(["item_a", "item_b"], "item_b") == 2
    assert target_rank(["item_a", "item_b"], "item_z") == 0


def test_build_query_type_breakdown_groups_metrics() -> None:
    per_query = pd.DataFrame(
        [
            {
                "config_name": "benchmark",
                "query_type": "title",
                "target_rank": 1,
                "hit_at_10": 1,
                "recall_at_10": 1.0,
                "mrr_at_10": 1.0,
            },
            {
                "config_name": "benchmark",
                "query_type": "title",
                "target_rank": 0,
                "hit_at_10": 0,
                "recall_at_10": 0.0,
                "mrr_at_10": 0.0,
            },
        ]
    )

    breakdown = build_query_type_breakdown(per_query)

    assert breakdown.loc[0, "num_queries"] == 2
    assert breakdown.loc[0, "recall_at_10"] == 0.5
    assert breakdown.loc[0, "miss_rate_at_10"] == 0.5
    assert breakdown.loc[0, "avg_target_rank"] == 1.0


def test_multi_relevant_metrics_group_targets_by_query_text() -> None:
    per_query = pd.DataFrame(
        [
            {
                "config_name": "benchmark",
                "query_text": "beauty products",
                "query_type": "category",
                "target_item_id": "item_a",
                "retrieved_item_ids": ["item_x", "item_a", "item_y"],
                "split": "test",
                "method": "bm25",
                "top_k": 3,
            },
            {
                "config_name": "benchmark",
                "query_text": "beauty products",
                "query_type": "category",
                "target_item_id": "item_b",
                "retrieved_item_ids": ["item_x", "item_a", "item_y"],
                "split": "test",
                "method": "bm25",
                "top_k": 3,
            },
        ]
    )

    multi_relevant = build_multi_relevant_metrics(per_query, [1, 3])
    summary = summarize_multi_relevant_metrics(multi_relevant)

    assert len(multi_relevant) == 1
    assert multi_relevant.loc[0, "num_relevant_items"] == 2
    assert multi_relevant.loc[0, "hit_any_at_1"] == 0.0
    assert multi_relevant.loc[0, "hit_any_at_3"] == 1.0
    assert multi_relevant.loc[0, "recall_multi_at_3"] == 0.5
    assert multi_relevant.loc[0, "mrr_any_at_3"] == 0.5
    assert summary.loc[0, "num_unique_query_texts"] == 1
