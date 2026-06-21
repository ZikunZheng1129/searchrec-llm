from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.llm_eval import (
    evaluate_query_understanding_outputs,
    evaluate_user_profile_outputs,
    exact_or_null_match,
    summarize_llm_metrics,
)


def test_exact_or_null_match() -> None:
    assert exact_or_null_match("Electronics", "electronics") == 1.0
    assert exact_or_null_match(None, None) == 1.0
    assert exact_or_null_match("Aster", None) == 0.0


def test_query_understanding_metrics() -> None:
    df = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "provider": "mock",
                "model": "mock",
                "split": "test",
                "parse_success": True,
                "schema_valid": True,
                "llm_intent": "brand_search",
                "source_intent": "brand_search",
                "llm_category": None,
                "source_category": None,
                "llm_brand": "Aster",
                "source_brand": "Aster",
                "llm_price_constraint": None,
                "source_price_constraint": None,
                "llm_use_case": None,
                "source_use_case": None,
                "expanded_queries": ["aster earbuds"],
                "latency_ms": 1.0,
                "estimated_cost_usd": 0.0,
            }
        ]
    )
    per_query = evaluate_query_understanding_outputs(df)
    summary = summarize_llm_metrics(per_query)
    assert summary["schema_valid_rate"].iloc[0] == 1.0
    assert summary["intent_match_rate"].iloc[0] == 1.0
    assert summary["estimated_cost_per_1000_queries"].iloc[0] == 0.0


def test_user_profile_metrics_and_empty_errors() -> None:
    profiles = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "provider": "mock",
                "model": "mock",
                "parse_success": True,
                "schema_valid": True,
                "profile_text": "Interests: electronics.",
                "latency_ms": 2.0,
                "estimated_cost_usd": 0.0,
            }
        ]
    )
    embeddings = pd.DataFrame([{"user_id": "u1", "embedding": [1.0], "embedding_dim": 1}])
    per_user = evaluate_user_profile_outputs(profiles, embeddings)
    summary = summarize_llm_metrics(per_user)
    assert summary["profile_generation_success_rate"].iloc[0] == 1.0
    assert summary["embedding_coverage"].iloc[0] == 1.0
    with pytest.raises(ValueError):
        evaluate_query_understanding_outputs(pd.DataFrame())
