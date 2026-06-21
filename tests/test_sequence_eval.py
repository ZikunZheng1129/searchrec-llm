from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.report import (
    build_final_leaderboard,
    generate_validation_artifacts,
    normalize_sequence_results,
)
from src.sequence_models.dataset import build_item_id_mapping
from src.sequence_models.evaluate import evaluate_sequence_model, summarize_sequence_metrics
from src.sequence_models.gru4rec import GRU4Rec


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["item_a", "item_b", "item_c"],
            "title": ["A", "B", "C"],
            "category": ["x", "x", "y"],
            "brand": ["b1", "b1", "b2"],
            "description": ["first", "second", "third"],
        }
    )


def _train() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "item_a", "timestamp": "2024-01-01"},
            {"user_id": "u1", "item_id": "item_b", "timestamp": "2024-01-02"},
        ]
    )


def _test() -> pd.DataFrame:
    return pd.DataFrame([{"user_id": "u1", "item_id": "item_c", "timestamp": "2024-01-03"}])


def test_sequence_evaluation_computes_expected_columns() -> None:
    mappings = build_item_id_mapping(_items(), pd.concat([_train(), _test()]))
    model = GRU4Rec(num_items=mappings.num_items, embedding_dim=8, hidden_dim=8)
    config = {
        "sequence": {"method": "gru4rec", "max_seq_len": 4, "eval_split": "test"},
        "evaluation": {"top_k": 2, "k_values": [1, 2]},
    }

    per_user = evaluate_sequence_model(model, _train(), _test(), _items(), mappings, config)
    summary = summarize_sequence_metrics(per_user)

    assert {"hit_rate_at_1", "recall_at_2", "ndcg_at_2", "mrr_at_2"}.issubset(per_user.columns)
    assert summary.loc[0, "num_users"] == 1


def test_normalize_sequence_results_outputs_leaderboard_rows() -> None:
    sequence_results = pd.DataFrame(
        [
            {
                "method": "gru4rec",
                "split": "test",
                "num_users": 10,
                "ndcg_at_10": 0.2,
                "recall_at_10": 0.3,
                "coverage_at_10": 0.4,
                "avg_latency_ms": 1.0,
                "p95_latency_ms": 2.0,
                "config_path": "configs/sequence/gru4rec_debug.yaml",
            }
        ]
    )

    normalized = normalize_sequence_results(sequence_results)
    leaderboard = build_final_leaderboard(sequence_results=normalized)

    assert normalized.loc[0, "stage"] == "sequence"
    assert leaderboard.loc[0, "stage"] == "sequence"


def test_missing_sequence_results_does_not_break_validation_report(tmp_path: Path) -> None:
    results_dir = tmp_path / "validation" / "results"
    reports_dir = tmp_path / "validation" / "reports"
    experiments_dir = tmp_path / "validation" / "experiments"
    results_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)
    experiments_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "method": "bm25",
                "split": "test",
                "num_queries": 1,
                "recall_at_50": 1.0,
                "mrr_at_10": 1.0,
                "query_coverage": 1.0,
                "avg_latency_ms": 1.0,
                "p95_latency_ms": 1.0,
                "index_backend": "local",
                "config_path": "retrieval.yaml",
            }
        ]
    ).to_csv(results_dir / "retrieval_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "method": "itemcf",
                "split": "test",
                "num_users": 1,
                "ndcg_at_10": 1.0,
                "recall_at_10": 1.0,
                "coverage_at_10": 1.0,
                "avg_latency_ms": 1.0,
                "p95_latency_ms": 1.0,
                "config_path": "rec.yaml",
            }
        ]
    ).to_csv(results_dir / "recommender_baselines.csv", index=False)
    config = {
        "inputs": {
            "retrieval_results_path": "validation/results/retrieval_results.csv",
            "recommender_results_path": "validation/results/recommender_baselines.csv",
        },
        "optional_inputs": {
            "sequence_results_path": "validation/results/sequence_results.csv",
        },
        "outputs": {
            "final_leaderboard_path": "validation/results/final_leaderboard.csv",
            "final_model_selection_report_path": (
                "validation/reports/final_model_selection_report.md"
            ),
            "latency_quality_report_path": "validation/reports/latency_quality_tradeoff.md",
            "validation_summary_path": "validation/reports/validation_summary.md",
        },
    }
    config_path = experiments_dir / "stage5.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    artifacts = generate_validation_artifacts(config, config_path)
    leaderboard = pd.read_csv(artifacts["final_leaderboard_path"])

    assert set(leaderboard["stage"]) == {"retrieval", "recommendation"}
