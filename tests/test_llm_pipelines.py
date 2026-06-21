from __future__ import annotations

import pandas as pd
import yaml

from src.evaluation.report import build_final_leaderboard, normalize_llm_query_understanding_results
from src.pipelines.evaluate_llm_query_understanding import run_evaluate_llm_query_understanding
from src.pipelines.generate_user_profiles import run_generate_user_profiles


def _write_query_inputs(tmp_path) -> tuple[str, str]:
    items = pd.DataFrame(
        [{"item_id": "i1", "category": "Electronics", "brand": "Aster", "title": "Earbuds"}]
    )
    queries = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "query_text": "budget electronics",
                "target_item_id": "i1",
                "intent": "price_sensitive_search",
                "category": "Electronics",
                "brand": None,
                "price_constraint": "budget",
                "use_case": None,
                "split": "test",
            }
        ]
    )
    item_path = tmp_path / "items.parquet"
    query_path = tmp_path / "queries.parquet"
    items.to_parquet(item_path, index=False)
    queries.to_parquet(query_path, index=False)
    return str(item_path), str(query_path)


def test_llm_query_pipeline_and_leaderboard(tmp_path) -> None:
    item_path, query_path = _write_query_inputs(tmp_path)
    parsed_path = tmp_path / "parsed.parquet"
    results_path = tmp_path / "llm_results.csv"
    config = {
        "seed": 42,
        "input": {"query_item_pairs_path": query_path, "item_metadata_path": item_path},
        "output": {"parsed_queries_path": str(parsed_path), "results_path": str(results_path)},
        "llm": {"client": "mock", "model": "mock-query-understanding-v1"},
        "query_understanding": {"split": "test", "max_expanded_queries": 3},
    }
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    row = run_evaluate_llm_query_understanding(config_path)
    assert row["num_queries"] == 1
    assert parsed_path.exists()
    assert results_path.exists()
    normalized = normalize_llm_query_understanding_results(pd.read_csv(results_path))
    leaderboard = build_final_leaderboard(llm_query_understanding_results=normalized)
    assert set(leaderboard["stage"]) == {"llm_query_understanding"}


def test_user_profile_pipeline_writes_outputs(tmp_path) -> None:
    items = pd.DataFrame(
        [
            {
                "item_id": "i1",
                "title": "Earbuds",
                "category": "Electronics",
                "brand": "Aster",
                "price": 99.0,
            }
        ]
    )
    train = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "event_type": "click", "event_weight": 2.0}]
    )
    item_path = tmp_path / "items.parquet"
    train_path = tmp_path / "train.parquet"
    profiles_path = tmp_path / "profiles.parquet"
    embeddings_path = tmp_path / "embeddings.parquet"
    results_path = tmp_path / "profile_results.csv"
    items.to_parquet(item_path, index=False)
    train.to_parquet(train_path, index=False)
    config = {
        "seed": 42,
        "input": {
            "train_path": str(train_path),
            "val_path": str(train_path),
            "test_path": str(train_path),
            "item_metadata_path": str(item_path),
        },
        "output": {
            "user_profiles_path": str(profiles_path),
            "user_profile_embeddings_path": str(embeddings_path),
            "results_path": str(results_path),
        },
        "llm": {"client": "mock", "model": "mock-user-profile-v1"},
        "user_profile": {"min_history_items": 1, "max_recent_items": 2},
        "profile_embedding": {"method": "bag_of_words", "max_features": 16, "normalize": True},
    }
    config_path = tmp_path / "profile.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    row = run_generate_user_profiles(config_path)
    assert row["num_users"] == 1
    assert profiles_path.exists()
    assert embeddings_path.exists()
    assert results_path.exists()
