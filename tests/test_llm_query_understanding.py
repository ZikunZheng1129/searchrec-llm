from __future__ import annotations

import pandas as pd

from src.llm.clients.mock_client import MockLLMClient
from src.llm.prompts.query_understanding_prompts import build_query_understanding_prompt
from src.llm.query_understanding.intent_extractor import LLMIntentExtractor
from src.llm.query_understanding.query_rewriter import normalize_expanded_queries
from src.llm.query_understanding.query_understanding_pipeline import run_query_understanding
from src.llm.query_understanding.schemas import validate_query_understanding_result


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item_id": "i1", "category": "Electronics", "brand": "Aster"},
            {"item_id": "i2", "category": "Sports & Outdoors", "brand": "Pulse"},
        ]
    )


def test_query_prompt_contains_json_instructions() -> None:
    prompt = build_query_understanding_prompt("budget electronics")
    assert "JSON" in prompt
    assert "intent" in prompt
    assert "constraints" in prompt


def test_query_schema_normalization_flattens_constraints() -> None:
    normalized = validate_query_understanding_result(
        {
            "intent": "price_sensitive_search",
            "category": "Electronics",
            "brand": None,
            "constraints": {"price": "budget", "use_case": "gym"},
            "rewritten_query": "budget electronics gym",
            "expanded_queries": ["budget electronics"],
            "confidence": "0.8",
        }
    )
    assert normalized["price_constraint"] == "budget"
    assert normalized["use_case"] == "gym"
    assert normalized["confidence"] == 0.8


def test_intent_extractor_parses_price_and_use_case() -> None:
    extractor = LLMIntentExtractor(
        MockLLMClient(),
        known_categories=["Electronics"],
        known_brands=["Aster"],
    )
    parsed = extractor.parse_query("budget electronics for gym")
    assert parsed["intent"] == "price_sensitive_search"
    assert parsed["category"] == "Electronics"
    assert parsed["price_constraint"] == "budget"
    assert parsed["use_case"] == "gym"
    assert parsed["parse_success"]


def test_query_expansion_removes_duplicates() -> None:
    expanded = normalize_expanded_queries(["Budget Electronics", "budget electronics"], "Budget")
    assert expanded == ["budget", "budget electronics"]


def test_batch_parsing_and_pipeline_outputs() -> None:
    pairs = pd.DataFrame(
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
    config = {
        "seed": 42,
        "llm": {"client": "mock", "model": "mock-query-understanding-v1"},
        "query_understanding": {"split": "test", "max_expanded_queries": 3},
    }
    output = run_query_understanding(pairs, _items(), config)
    assert {"query_id", "llm_intent", "rewritten_query", "expanded_queries"}.issubset(
        output.columns
    )
    assert output["schema_valid"].iloc[0]


def test_fallback_when_mock_returns_invalid_json() -> None:
    extractor = LLMIntentExtractor(
        MockLLMClient(simulate_invalid_json=True),
        known_categories=["Electronics"],
    )
    parsed = extractor.parse_query("budget electronics")
    assert not parsed["parse_success"]
    assert parsed["intent"] == "price_sensitive_search"
