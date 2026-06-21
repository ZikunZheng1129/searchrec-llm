from __future__ import annotations

import pandas as pd
import yaml

from src.evaluation.report import build_final_leaderboard, normalize_ranking_results
from src.pipelines.build_ranking_dataset import run_build_ranking_dataset
from src.pipelines.evaluate_ranker import run_evaluate_ranker
from src.pipelines.train_ranker import run_train_ranker


def _write_inputs(tmp_path) -> dict[str, str]:
    items = pd.DataFrame(
        [
            {
                "item_id": "i1",
                "title": "Wireless Earbuds",
                "category": "Electronics",
                "brand": "Aster",
                "price": 99.0,
                "avg_rating": 4.5,
                "rating_count": 100,
                "description": "Clear audio",
            },
            {
                "item_id": "i2",
                "title": "Yoga Mat",
                "category": "Sports",
                "brand": "Pulse",
                "price": 30.0,
                "avg_rating": 4.0,
                "rating_count": 0,
                "description": "Stretching mat",
            },
            {
                "item_id": "i3",
                "title": "Hydrating Serum",
                "category": "Beauty",
                "brand": "Haven",
                "price": 20.0,
                "avg_rating": 4.8,
                "rating_count": 500,
                "description": "Gentle skincare",
            },
        ]
    )
    queries = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "query_text": "wireless audio",
                "target_item_id": "i1",
                "split": "train",
            },
            {
                "query_id": "q2",
                "query_text": "hydrating serum",
                "target_item_id": "i3",
                "split": "test",
            },
        ]
    )
    interactions = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "event_weight": 4.0},
            {"user_id": "u2", "item_id": "i2", "event_type": "view", "event_weight": 1.0},
        ]
    )
    paths = {
        "items": str(tmp_path / "items.parquet"),
        "queries": str(tmp_path / "queries.parquet"),
        "train": str(tmp_path / "train.parquet"),
        "val": str(tmp_path / "val.parquet"),
        "test": str(tmp_path / "test.parquet"),
    }
    items.to_parquet(paths["items"], index=False)
    queries.to_parquet(paths["queries"], index=False)
    interactions.to_parquet(paths["train"], index=False)
    interactions.to_parquet(paths["val"], index=False)
    interactions.to_parquet(paths["test"], index=False)
    return paths


def test_ranking_pipeline_and_report_normalization(tmp_path) -> None:
    paths = _write_inputs(tmp_path)
    candidates_path = tmp_path / "ranking_candidates.parquet"
    dataset_config = {
        "seed": 42,
        "input": {
            "item_metadata_path": paths["items"],
            "query_item_pairs_path": paths["queries"],
            "train_interactions_path": paths["train"],
            "val_interactions_path": paths["val"],
            "test_interactions_path": paths["test"],
        },
        "output": {"ranking_candidates_path": str(candidates_path)},
        "candidate_generation": {
            "candidate_pool_size": 2,
            "random_negatives_per_query": 1,
            "ensure_positive_candidate": True,
        },
        "retrieval": {
            "bm25": {"k1": 1.5, "b": 0.75},
            "dense": {"normalize": True, "max_features": 100},
            "hybrid": {"bm25_weight": 0.5, "dense_weight": 0.5, "score_normalization": "minmax"},
        },
    }
    dataset_config_path = tmp_path / "ranking_dataset.yaml"
    dataset_config_path.write_text(yaml.safe_dump(dataset_config), encoding="utf-8")
    summary = run_build_ranking_dataset(dataset_config_path)
    assert summary["positive_candidate_coverage"] == 1.0
    assert candidates_path.exists()

    model_path = tmp_path / "ranker.pkl"
    results_path = tmp_path / "ranking_results.csv"
    ranker_config = {
        "seed": 42,
        "input": {"ranking_candidates_path": str(candidates_path)},
        "output": {"model_path": str(model_path), "results_path": str(results_path)},
        "ranking": {
            "method": "lightgbm",
            "backend": "numpy_linear",
            "fallback_backend": "numpy_linear",
            "top_k": 2,
            "label_column": "label",
            "group_column": "query_id",
        },
        "training": {
            "learning_rate": 0.05,
            "num_boost_round": 2,
            "num_epochs_fallback": 3,
            "regularization": 0.001,
        },
        "evaluation": {"split": "test", "k_values": [1, 2]},
    }
    ranker_config_path = tmp_path / "ranker.yaml"
    ranker_config_path.write_text(yaml.safe_dump(ranker_config), encoding="utf-8")
    train_summary = run_train_ranker(ranker_config_path)
    assert train_summary["backend"] == "numpy_linear"
    eval_summary = run_evaluate_ranker(ranker_config_path)
    assert eval_summary["num_queries"] == 1
    assert results_path.exists()

    normalized = normalize_ranking_results(
        pd.read_csv(results_path), source_results_path=results_path
    )
    leaderboard = build_final_leaderboard(ranking_results=normalized)
    assert set(leaderboard["stage"]) == {"ranking"}
