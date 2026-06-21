"""Deterministic query rewrite and expansion helpers."""

from __future__ import annotations

import re
from typing import Any


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def normalize_expanded_queries(
    expanded_queries: list[str],
    original_query: str,
    max_queries: int = 3,
) -> list[str]:
    """Normalize, de-duplicate, and bound expanded queries."""
    candidates = [original_query, *expanded_queries]
    output: list[str] = []
    for candidate in candidates:
        normalized = _normalize(candidate)
        if normalized and normalized not in output:
            output.append(normalized)
    return output[: int(max_queries)]


def fallback_query_rewrite(query_text: str, parsed_query: dict[str, Any]) -> str:
    """Build a deterministic rewrite when LLM output is unavailable."""
    pieces = [
        parsed_query.get("price_constraint"),
        parsed_query.get("brand"),
        parsed_query.get("category"),
        parsed_query.get("use_case"),
    ]
    rewrite = _normalize(" ".join(str(piece) for piece in pieces if piece))
    return rewrite or _normalize(query_text)


def fallback_query_expansion(
    query_text: str,
    parsed_query: dict[str, Any],
    max_queries: int = 3,
) -> list[str]:
    """Build deterministic query expansions from parsed fields."""
    rewrite = fallback_query_rewrite(query_text, parsed_query)
    candidates = [
        query_text,
        rewrite,
        " ".join(
            str(value)
            for value in [parsed_query.get("category"), parsed_query.get("use_case")]
            if value
        ),
    ]
    return normalize_expanded_queries(candidates, query_text, max_queries=max_queries)
