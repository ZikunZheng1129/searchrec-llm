from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.report import (
    LEADERBOARD_COLUMNS,
    build_final_leaderboard,
    normalize_recommendation_results,
    normalize_retrieval_results,
    read_results_csv,
    render_markdown_table,
    select_best_models,
)


def _retrieval_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "bm25",
                "split": "test",
                "num_queries": 10,
                "recall_at_50": 0.8,
                "mrr_at_10": 0.2,
                "query_coverage": 1.0,
                "avg_latency_ms": 1.5,
                "p95_latency_ms": 2.0,
                "index_backend": "local",
                "config_path": "configs/retrieval/bm25_debug.yaml",
            }
        ]
    )


def _recommendation_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "itemcf",
                "split": "test",
                "num_users": 12,
                "ndcg_at_10": 0.4,
                "recall_at_10": 0.3,
                "coverage_at_10": 0.9,
                "avg_latency_ms": 2.0,
                "p95_latency_ms": 3.0,
                "config_path": "configs/recommendation/itemcf_debug.yaml",
            }
        ]
    )


def test_normalize_retrieval_results_outputs_standard_columns() -> None:
    normalized = normalize_retrieval_results(_retrieval_results(), "retrieval.csv")

    assert list(normalized.columns) == LEADERBOARD_COLUMNS
    assert normalized.loc[0, "stage"] == "retrieval"
    assert normalized.loc[0, "primary_metric_name"] == "recall_at_50"
    assert normalized.loc[0, "primary_score"] == 0.8


def test_normalize_recommendation_results_outputs_standard_columns() -> None:
    normalized = normalize_recommendation_results(_recommendation_results(), "recommendation.csv")

    assert list(normalized.columns) == LEADERBOARD_COLUMNS
    assert normalized.loc[0, "stage"] == "recommendation"
    assert normalized.loc[0, "primary_metric_name"] == "ndcg_at_10"
    assert normalized.loc[0, "index_backend"] == "n/a"


def test_build_final_leaderboard_combines_rows() -> None:
    leaderboard = build_final_leaderboard(_retrieval_results(), _recommendation_results())

    assert len(leaderboard) == 2
    assert set(leaderboard["stage"]) == {"retrieval", "recommendation"}


def test_select_best_models_uses_tie_break_rules() -> None:
    leaderboard = pd.DataFrame(
        [
            {
                **{column: "" for column in LEADERBOARD_COLUMNS},
                "stage": "retrieval",
                "method": "slow",
                "primary_score": 1.0,
                "secondary_score": 0.5,
                "coverage_score": 1.0,
                "avg_latency_ms": 10.0,
            },
            {
                **{column: "" for column in LEADERBOARD_COLUMNS},
                "stage": "retrieval",
                "method": "fast",
                "primary_score": 1.0,
                "secondary_score": 0.5,
                "coverage_score": 1.0,
                "avg_latency_ms": 1.0,
            },
        ]
    )

    best = select_best_models(leaderboard)

    assert best.loc[0, "method"] == "fast"


def test_render_markdown_table_returns_table() -> None:
    table = render_markdown_table(pd.DataFrame({"method": ["bm25"], "score": [1.0]}))

    assert "| method | score |" in table
    assert "| bm25 | 1.0000 |" in table


def test_missing_required_input_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_results_csv(tmp_path / "missing.csv", required=True)


def test_missing_optional_input_returns_empty_dataframe(tmp_path: Path) -> None:
    result = read_results_csv(tmp_path / "future_missing.csv", required=False)

    assert result.empty
