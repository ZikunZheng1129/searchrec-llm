"""Evaluation helpers for Stage 8 LLM outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    text = str(value).strip().lower()
    return text in {"", "none", "null", "nan", "unknown"}


def exact_or_null_match(predicted: Any, expected: Any) -> float:
    """Return exact normalized match, treating null expected values fairly."""
    if _is_null(expected):
        return 1.0 if _is_null(predicted) else 0.0
    if _is_null(predicted):
        return 0.0
    return float(str(predicted).strip().lower() == str(expected).strip().lower())


def list_nonempty_rate(values: list[list[str]]) -> float:
    """Compute fraction of lists with at least one non-empty item."""
    if not values:
        return 0.0
    return float(
        np.mean(
            [bool([str(item).strip() for item in value if str(item).strip()]) for value in values]
        )
    )


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [str(value)]


def evaluate_query_understanding_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate parsed query-understanding rows against synthetic labels."""
    if df.empty:
        raise ValueError("Query-understanding outputs are empty")
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "query_id": str(row.get("query_id", "")),
                "method": "llm_query_understanding",
                "provider": str(row.get("provider", "")),
                "model": str(row.get("model", "")),
                "split": str(row.get("split", "")),
                "parse_success": float(bool(row.get("parse_success", False))),
                "schema_valid": float(bool(row.get("schema_valid", False))),
                "intent_match": exact_or_null_match(
                    row.get("llm_intent"), row.get("source_intent")
                ),
                "category_match": exact_or_null_match(
                    row.get("llm_category"),
                    row.get("source_category"),
                ),
                "brand_match": exact_or_null_match(row.get("llm_brand"), row.get("source_brand")),
                "price_constraint_match": exact_or_null_match(
                    row.get("llm_price_constraint"),
                    row.get("source_price_constraint"),
                ),
                "use_case_match": exact_or_null_match(
                    row.get("llm_use_case"),
                    row.get("source_use_case"),
                ),
                "num_expanded_queries": len(_as_list(row.get("expanded_queries"))),
                "latency_ms": float(row.get("latency_ms", 0.0) or 0.0),
                "estimated_cost_usd": float(row.get("estimated_cost_usd", 0.0) or 0.0),
            }
        )
    return pd.DataFrame(rows)


def evaluate_user_profile_outputs(
    profiles: pd.DataFrame,
    embeddings: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate generated user profiles and embedding coverage."""
    if profiles.empty:
        raise ValueError("User profile outputs are empty")
    embedded_users = set(embeddings["user_id"].astype(str)) if not embeddings.empty else set()
    rows = []
    for _, row in profiles.iterrows():
        profile_text = str(row.get("profile_text", "") or "")
        rows.append(
            {
                "user_id": str(row.get("user_id", "")),
                "method": "llm_user_profile",
                "provider": str(row.get("provider", "")),
                "model": str(row.get("model", "")),
                "profile_generation_success": float(bool(row.get("parse_success", False))),
                "schema_valid": float(bool(row.get("schema_valid", False))),
                "profile_nonempty": float(bool(profile_text.strip())),
                "profile_length": float(len(profile_text.split())),
                "embedding_present": float(str(row.get("user_id", "")) in embedded_users),
                "latency_ms": float(row.get("latency_ms", 0.0) or 0.0),
                "estimated_cost_usd": float(row.get("estimated_cost_usd", 0.0) or 0.0),
            }
        )
    return pd.DataFrame(rows)


def summarize_llm_metrics(per_example_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize query-understanding or user-profile per-example metrics."""
    if per_example_results.empty:
        raise ValueError("LLM per-example results are empty")
    method = str(per_example_results["method"].iloc[0])
    provider = str(per_example_results["provider"].iloc[0])
    model = str(per_example_results["model"].iloc[0])
    latency = per_example_results["latency_ms"].astype(float)
    cost = per_example_results["estimated_cost_usd"].astype(float)
    base = {
        "method": method,
        "provider": provider,
        "model": model,
        "avg_latency_ms": float(latency.mean()),
        "p95_latency_ms": float(np.percentile(latency, 95)),
    }
    if method == "llm_query_understanding":
        base.update(
            {
                "stage": "llm_query_understanding",
                "split": str(per_example_results["split"].iloc[0]),
                "num_queries": int(len(per_example_results)),
                "output_parse_success_rate": float(per_example_results["parse_success"].mean()),
                "schema_valid_rate": float(per_example_results["schema_valid"].mean()),
                "intent_match_rate": float(per_example_results["intent_match"].mean()),
                "category_match_rate": float(per_example_results["category_match"].mean()),
                "brand_match_rate": float(per_example_results["brand_match"].mean()),
                "price_constraint_match_rate": float(
                    per_example_results["price_constraint_match"].mean()
                ),
                "use_case_match_rate": float(per_example_results["use_case_match"].mean()),
                "avg_expanded_queries": float(per_example_results["num_expanded_queries"].mean()),
                "estimated_cost_per_1000_queries": float(
                    cost.sum() / len(per_example_results) * 1000
                ),
            }
        )
    elif method == "llm_user_profile":
        base.update(
            {
                "stage": "llm_user_profile",
                "num_users": int(len(per_example_results)),
                "profile_generation_success_rate": float(
                    per_example_results["profile_generation_success"].mean()
                ),
                "schema_valid_rate": float(per_example_results["schema_valid"].mean()),
                "profile_nonempty_rate": float(per_example_results["profile_nonempty"].mean()),
                "embedding_coverage": float(per_example_results["embedding_present"].mean()),
                "avg_profile_length": float(per_example_results["profile_length"].mean()),
                "estimated_cost_per_1000_users": float(
                    cost.sum() / len(per_example_results) * 1000
                ),
            }
        )
    else:
        raise ValueError(f"Unsupported LLM method: {method}")
    return pd.DataFrame([base])
