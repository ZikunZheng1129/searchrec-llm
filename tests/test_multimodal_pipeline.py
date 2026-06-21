from __future__ import annotations

import pandas as pd
import yaml

from src.evaluation.report import build_final_leaderboard, normalize_multimodal_results
from src.pipelines.augment_ranking_with_multimodal import run_augment_ranking_with_multimodal
from src.pipelines.build_multimodal_embeddings import run_build_multimodal_embeddings
from src.pipelines.evaluate_multimodal import run_evaluate_multimodal


def _write_inputs(tmp_path) -> dict[str, str]:
    items = pd.DataFrame(
        [
            {
                "item_id": "i1",
                "title": "Wireless Earbuds",
                "category": "Electronics",
                "brand": "Aster",
                "description": "Clear audio",
                "price": 99.0,
                "avg_rating": 4.5,
                "rating_count": 100,
            },
            {
                "item_id": "i2",
                "title": "Yoga Mat",
                "category": "Sports",
                "brand": "Pulse",
                "description": "Stretching mat",
                "price": 30.0,
                "avg_rating": 4.0,
                "rating_count": 0,
            },
        ]
    )
    queries = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "query_text": "wireless audio",
                "target_item_id": "i1",
                "split": "test",
            }
        ]
    )
    train = pd.DataFrame([{"user_id": "u1", "item_id": "i2"}])
    ranking = pd.DataFrame(
        [
            {"query_id": "q1", "candidate_item_id": "i1", "label": 1},
            {"query_id": "q1", "candidate_item_id": "i2", "label": 0},
        ]
    )
    paths = {
        "items": str(tmp_path / "items.parquet"),
        "queries": str(tmp_path / "queries.parquet"),
        "train": str(tmp_path / "train.parquet"),
        "ranking": str(tmp_path / "ranking.parquet"),
    }
    items.to_parquet(paths["items"], index=False)
    queries.to_parquet(paths["queries"], index=False)
    train.to_parquet(paths["train"], index=False)
    ranking.to_parquet(paths["ranking"], index=False)
    return paths


def _config(tmp_path, paths: dict[str, str]) -> dict:
    return {
        "seed": 42,
        "input": {
            "item_metadata_path": paths["items"],
            "query_item_pairs_path": paths["queries"],
            "train_interactions_path": paths["train"],
            "ranking_candidates_path": paths["ranking"],
        },
        "output": {
            "item_embeddings_path": str(tmp_path / "embeddings.parquet"),
            "ranking_candidates_output_path": str(tmp_path / "ranking_mm.parquet"),
            "results_path": str(tmp_path / "multimodal_results.csv"),
        },
        "multimodal": {"method": "text_metadata_fusion", "split": "test", "top_k": 2},
        "text_encoder": {
            "fields": ["title", "category", "brand", "description"],
            "max_features": 50,
            "normalize": True,
        },
        "metadata_encoder": {
            "categorical_fields": ["category", "brand"],
            "numeric_fields": ["price", "avg_rating", "rating_count"],
            "add_log_numeric": ["rating_count"],
            "normalize_numeric": True,
        },
        "fusion": {"strategy": "concat", "normalize_output": True},
        "evaluation": {"k_values": [1, 2], "cold_start_quantile": 0.25, "long_tail_quantile": 0.25},
    }


def test_multimodal_pipelines_and_report_normalization(tmp_path) -> None:
    paths = _write_inputs(tmp_path)
    config = _config(tmp_path, paths)
    config_path = tmp_path / "multimodal.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    build_summary = run_build_multimodal_embeddings(config_path)
    assert build_summary["embedding_dim"] > 0
    eval_summary = run_evaluate_multimodal(config_path)
    assert eval_summary["num_queries"] == 1
    results_path = config["output"]["results_path"]
    normalized = normalize_multimodal_results(pd.read_csv(results_path))
    leaderboard = build_final_leaderboard(multimodal_results=normalized)
    assert set(leaderboard["stage"]) == {"multimodal"}


def test_augment_ranking_with_multimodal_keeps_row_count(tmp_path) -> None:
    paths = _write_inputs(tmp_path)
    config = _config(tmp_path, paths)
    config_path = tmp_path / "multimodal.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    summary = run_augment_ranking_with_multimodal(config_path)
    assert summary["input_rows"] == summary["output_rows"] == 2
    output = pd.read_parquet(config["output"]["ranking_candidates_output_path"])
    assert {"multimodal_score", "multimodal_rank"}.issubset(output.columns)
