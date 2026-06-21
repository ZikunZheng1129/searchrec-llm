from __future__ import annotations

import pandas as pd
import pytest

from app.api.errors import ArtifactMissingError
from app.api.service import DemoService


def _write_demo_artifacts(tmp_path):
    items = pd.DataFrame(
        {
            "item_id": ["item_1", "item_2"],
            "title": ["Beauty Serum", "Phone Stand"],
            "category": ["Beauty", "Electronics"],
            "brand": ["Haven", "Nova"],
            "price": [12.0, 20.0],
            "avg_rating": [4.5, 4.2],
            "rating_count": [100, 80],
            "description": ["Gentle serum", "Desk stand"],
        }
    )
    pairs = pd.DataFrame(
        {
            "query_id": ["q1"],
            "query_text": ["gift beauty"],
            "split": ["test"],
            "target_item_id": ["item_1"],
        }
    )
    candidates = pd.DataFrame(
        {
            "query_id": ["q1", "q1"],
            "candidate_item_id": ["item_1", "item_2"],
            "multimodal_score": [0.9, 0.1],
            "multimodal_rank": [1, 2],
        }
    )
    profiles = pd.DataFrame(
        {
            "user_id": ["user_1"],
            "top_categories": [["Beauty"]],
            "top_brands": [["Haven"]],
            "profile_text": ["Beauty profile"],
        }
    )
    leaderboard = pd.DataFrame(
        {
            "stage": ["genrec"],
            "method": ["candidate_constrained_generation"],
            "primary_score": [1.0],
            "secondary_score": [0.5],
            "coverage_score": [1.0],
            "avg_latency_ms": [0.1],
        }
    )
    paths = {
        "item_metadata_path": tmp_path / "items.parquet",
        "query_item_pairs_path": tmp_path / "queries.parquet",
        "ranking_candidates_path": tmp_path / "ranking.parquet",
        "ranking_candidates_multimodal_path": tmp_path / "ranking_multi.parquet",
        "user_profiles_path": tmp_path / "profiles.parquet",
        "final_leaderboard_path": tmp_path / "leaderboard.csv",
    }
    items.to_parquet(paths["item_metadata_path"])
    pairs.to_parquet(paths["query_item_pairs_path"])
    candidates.to_parquet(paths["ranking_candidates_path"])
    candidates.to_parquet(paths["ranking_candidates_multimodal_path"])
    profiles.to_parquet(paths["user_profiles_path"])
    leaderboard.to_csv(paths["final_leaderboard_path"], index=False)
    return paths


def _service(tmp_path) -> DemoService:
    paths = _write_demo_artifacts(tmp_path)
    config = {
        "app": {"name": "test-api", "version": "0.0.1", "environment": "test"},
        "artifacts": {key: str(value) for key, value in paths.items()},
    }
    return DemoService(config)


def test_artifact_status_and_health(tmp_path):
    service = _service(tmp_path)
    status = service.artifact_status()["artifacts"]
    assert all(row["exists"] for row in status)
    assert service.health()["status"] == "ok"


def test_understand_query_search_recommend_and_genrec(tmp_path):
    service = _service(tmp_path)
    parsed = service.understand_query("gift beauty")
    assert parsed["provider"] == "mock"
    search = service.search("gift beauty", top_k=1)
    assert search["items"][0]["item_id"] == "item_1"
    recs = service.recommend("user_1", top_k=1)
    assert recs["items"][0]["item_id"] == "item_1"
    genrec = service.genrec("gift beauty", top_k=1)
    assert genrec["recommended_items"][0]["item_id"] == "item_1"
    assert genrec["hallucination_rate"] == 0.0


def test_model_comparison_and_missing_artifact(tmp_path):
    service = _service(tmp_path)
    assert service.model_comparison()["best_by_stage"][0]["stage"] == "genrec"
    missing = DemoService({"artifacts": {"item_metadata_path": str(tmp_path / "missing.parquet")}})
    with pytest.raises(ArtifactMissingError):
        missing.search("gift beauty")
