from __future__ import annotations

import pandas as pd

from src.evaluation.report import build_final_leaderboard, generate_validation_artifacts
from src.pipelines.evaluate_genrec import run_evaluate_genrec
from src.utils.config import save_yaml_config


def _write_inputs(tmp_path):
    query_pairs = pd.DataFrame(
        {
            "query_id": ["q1", "q2"],
            "query_text": ["beauty serum", "phone stand"],
            "split": ["test", "test"],
            "target_item_id": ["item_1", "item_2"],
        }
    )
    items = pd.DataFrame(
        {
            "item_id": ["item_1", "item_2", "item_3"],
            "title": ["Beauty Serum", "Phone Stand", "Yoga Mat"],
            "category": ["Beauty", "Electronics", "Sports"],
            "brand": ["Haven", "Nova", "Pulse"],
            "price": [12.0, 20.0, 30.0],
            "avg_rating": [4.5, 4.3, 4.1],
            "rating_count": [100, 80, 50],
            "description": ["Gentle serum", "Desk phone stand", "Exercise mat"],
        }
    )
    candidates = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2", "q2"],
            "candidate_item_id": ["item_1", "item_3", "item_2", "item_3"],
            "multimodal_score": [0.9, 0.1, 0.8, 0.2],
            "multimodal_rank": [1, 2, 1, 2],
            "category": ["Beauty", "Sports", "Electronics", "Sports"],
            "brand": ["Haven", "Pulse", "Nova", "Pulse"],
            "price": [12.0, 30.0, 20.0, 30.0],
            "avg_rating": [4.5, 4.1, 4.3, 4.1],
            "rating_count": [100, 50, 80, 50],
        }
    )
    paths = {
        "query_pairs": tmp_path / "query_item_pairs.parquet",
        "items": tmp_path / "item_metadata.parquet",
        "candidates": tmp_path / "ranking_candidates.parquet",
        "multimodal_candidates": tmp_path / "ranking_candidates_multimodal.parquet",
    }
    query_pairs.to_parquet(paths["query_pairs"])
    items.to_parquet(paths["items"])
    candidates.to_parquet(paths["candidates"])
    candidates.to_parquet(paths["multimodal_candidates"])
    return paths


def _config(tmp_path):
    paths = _write_inputs(tmp_path)
    return {
        "seed": 42,
        "input": {
            "query_item_pairs_path": str(paths["query_pairs"]),
            "item_metadata_path": str(paths["items"]),
            "ranking_candidates_path": str(paths["candidates"]),
            "ranking_candidates_multimodal_path": str(paths["multimodal_candidates"]),
            "llm_query_understanding_path": str(tmp_path / "missing_llm.parquet"),
            "user_profiles_path": str(tmp_path / "missing_profiles.parquet"),
        },
        "output": {
            "genrec_outputs_path": str(tmp_path / "genrec_outputs.parquet"),
            "llm_rerank_outputs_path": str(tmp_path / "llm_rerank_outputs.parquet"),
            "explanations_path": str(tmp_path / "recommendation_explanations.parquet"),
            "results_path": str(tmp_path / "genrec_results.csv"),
            "final_llm_decision_report_path": str(tmp_path / "final_llm_decision.md"),
        },
        "llm": {
            "client": "mock",
            "model": "mock-genrec-v1",
            "temperature": 0.0,
            "max_tokens": 256,
            "allow_external_api_calls": False,
        },
        "genrec": {
            "split": "test",
            "candidate_source": "auto",
            "candidate_pool_size": 2,
            "final_top_k": 2,
            "experiments": [
                "direct_generation",
                "candidate_constrained_generation",
                "llm_rerank_top_10",
                "template_explanation",
                "evidence_grounded_llm_explanation",
            ],
        },
        "constraints": {
            "require_candidate_item_ids": True,
            "allow_fallback_to_ranked_candidates": True,
            "max_recommendations": 2,
        },
        "explanations": {"max_evidence_items": 2},
        "evaluation": {"k_values": [10]},
    }


def test_evaluate_genrec_pipeline_writes_outputs_and_results(tmp_path):
    config_path = tmp_path / "genrec.yaml"
    save_yaml_config(_config(tmp_path), config_path)
    result = run_evaluate_genrec(config_path)
    assert pd.read_parquet(result["genrec_outputs_path"]).shape[0] == 4
    assert pd.read_parquet(result["llm_rerank_outputs_path"]).shape[0] == 2
    assert pd.read_parquet(result["explanations_path"]).shape[0] == 4
    results = pd.read_csv(result["results_path"])
    assert {
        "direct_generation",
        "candidate_constrained_generation",
        "llm_rerank_top_10",
        "template_explanation",
        "evidence_grounded_llm_explanation",
    }.issubset(set(results["method"]))
    assert (tmp_path / "final_llm_decision.md").exists()
    assert "candidate-constrained" in (tmp_path / "final_llm_decision.md").read_text()


def test_report_optional_genrec_normalization_and_missing_optional(tmp_path):
    genrec = pd.DataFrame(
        {
            "stage": ["genrec"],
            "method": ["candidate_constrained_generation"],
            "provider": ["mock"],
            "model": ["mock-genrec-v1"],
            "split": ["test"],
            "num_queries": [2],
            "valid_item_rate": [1.0],
            "hallucination_rate": [0.0],
            "output_parse_success_rate": [1.0],
            "ndcg_at_10": [0.5],
            "avg_latency_ms": [0.1],
            "p95_latency_ms": [0.1],
            "config_path": ["cfg.yaml"],
        }
    )
    leaderboard = build_final_leaderboard(genrec_results=genrec)
    assert leaderboard["stage"].tolist() == ["genrec"]

    retrieval = pd.DataFrame(
        {
            "method": ["bm25"],
            "config_path": ["r.yaml"],
            "split": ["test"],
            "num_queries": [1],
            "recall_at_50": [1.0],
            "mrr_at_10": [1.0],
            "query_coverage": [1.0],
            "avg_latency_ms": [1.0],
            "p95_latency_ms": [1.0],
            "index_backend": ["local"],
        }
    )
    recommendation = pd.DataFrame(
        {
            "method": ["itemcf"],
            "config_path": ["c.yaml"],
            "split": ["test"],
            "num_users": [1],
            "ndcg_at_10": [1.0],
            "recall_at_10": [1.0],
            "coverage_at_10": [1.0],
            "avg_latency_ms": [1.0],
            "p95_latency_ms": [1.0],
        }
    )
    retrieval_path = tmp_path / "retrieval.csv"
    recommendation_path = tmp_path / "recommendation.csv"
    retrieval.to_csv(retrieval_path, index=False)
    recommendation.to_csv(recommendation_path, index=False)
    config = {
        "inputs": {
            "retrieval_results_path": str(retrieval_path),
            "recommender_results_path": str(recommendation_path),
        },
        "optional_inputs": {"genrec_results_path": str(tmp_path / "missing.csv")},
        "outputs": {
            "final_leaderboard_path": str(tmp_path / "leaderboard.csv"),
            "final_model_selection_report_path": str(tmp_path / "selection.md"),
            "latency_quality_report_path": str(tmp_path / "latency.md"),
            "validation_summary_path": str(tmp_path / "summary.md"),
        },
    }
    artifacts = generate_validation_artifacts(config)
    assert (tmp_path / "leaderboard.csv").exists()
    assert "final_leaderboard_path" in artifacts
