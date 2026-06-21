"""Pipeline helpers for Stage 8 query understanding."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.llm.clients.base_client import BaseLLMClient
from src.llm.clients.local_hf_client import LocalHFClient
from src.llm.clients.mock_client import MockLLMClient
from src.llm.clients.openai_client import OpenAIClient
from src.llm.query_understanding.intent_extractor import LLMIntentExtractor


def build_llm_client_from_config(config: dict[str, Any]) -> BaseLLMClient:
    """Create an LLM client from config without triggering external calls."""
    llm_config = config.get("llm", {})
    client_name = str(llm_config.get("client", "mock"))
    if client_name == "mock":
        return MockLLMClient(
            model=str(llm_config.get("model", "mock-query-understanding-v1")),
            seed=int(config.get("seed", 42)),
            simulate_invalid_json=bool(llm_config.get("simulate_invalid_json", False)),
        )
    if client_name == "openai":
        return OpenAIClient(
            model=str(llm_config.get("model", "gpt-4.1-mini")),
            allow_external_api_calls=bool(llm_config.get("allow_external_api_calls", False)),
        )
    if client_name == "local_hf":
        return LocalHFClient(
            model=str(llm_config.get("model", "")),
            allow_model_load=bool(llm_config.get("allow_model_load", False)),
        )
    raise ValueError(f"Unsupported llm.client: {client_name}")


def _known_values(items: pd.DataFrame, column: str, enabled: bool) -> list[str]:
    if not enabled or column not in items.columns:
        return []
    return sorted(items[column].dropna().astype(str).unique().tolist())


def run_query_understanding(
    query_item_pairs: pd.DataFrame,
    items: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Run structured LLM query understanding for query-item rows."""
    query_config = config.get("query_understanding", {})
    pairs = query_item_pairs.copy()
    split = query_config.get("split")
    if split is not None:
        pairs = pairs[pairs["split"].astype(str) == str(split)].copy()
    max_queries = query_config.get("max_queries")
    if max_queries is not None:
        pairs = pairs.head(int(max_queries)).copy()
    pairs = pairs.sort_values(["split", "query_id"]).reset_index(drop=True)

    client = build_llm_client_from_config(config)
    extractor = LLMIntentExtractor(
        client=client,
        known_categories=_known_values(
            items,
            "category",
            bool(query_config.get("known_categories_from_items", True)),
        ),
        known_brands=_known_values(
            items,
            "brand",
            bool(query_config.get("known_brands_from_items", True)),
        ),
        max_expanded_queries=int(query_config.get("max_expanded_queries", 3)),
    )

    rows: list[dict[str, Any]] = []
    for _, row in pairs.iterrows():
        parsed = extractor.parse_query(str(row["query_text"]))
        rows.append(
            {
                "query_id": str(row.get("query_id", "")),
                "query_text": str(row.get("query_text", "")),
                "split": str(row.get("split", "")),
                "target_item_id": str(row.get("target_item_id", "")),
                "source_intent": row.get("intent"),
                "llm_intent": parsed.get("intent"),
                "source_category": row.get("category"),
                "llm_category": parsed.get("category"),
                "source_brand": row.get("brand"),
                "llm_brand": parsed.get("brand"),
                "source_price_constraint": row.get("price_constraint"),
                "llm_price_constraint": parsed.get("price_constraint"),
                "source_use_case": row.get("use_case"),
                "llm_use_case": parsed.get("use_case"),
                "rewritten_query": parsed.get("rewritten_query"),
                "expanded_queries": parsed.get("expanded_queries", []),
                "confidence": parsed.get("confidence"),
                "parse_success": parsed.get("parse_success", False),
                "schema_valid": parsed.get("schema_valid", False),
                "parse_error": parsed.get("parse_error"),
                "provider": parsed.get("provider"),
                "model": parsed.get("model"),
                "latency_ms": parsed.get("latency_ms", 0.0),
                "estimated_cost_usd": parsed.get("estimated_cost_usd", 0.0),
            }
        )
    return pd.DataFrame(rows)
