"""Prompt templates for structured query understanding."""

from __future__ import annotations


def build_query_understanding_system_prompt() -> str:
    """Return the query-understanding system prompt."""
    return (
        "You extract shopping query structure. Return JSON only. "
        "Do not invent category or brand values outside provided lists; use null when unknown."
    )


def build_query_understanding_prompt(
    query_text: str,
    known_categories: list[str] | None = None,
    known_brands: list[str] | None = None,
    max_expanded_queries: int = 3,
) -> str:
    """Build a concise structured query-understanding prompt."""
    categories = " | ".join(sorted(known_categories or [])) or "None"
    brands = " | ".join(sorted(known_brands or [])) or "None"
    return f"""QUERY_UNDERSTANDING_TASK
Return one JSON object with exactly these fields:
intent, category, brand, constraints, rewritten_query, expanded_queries, confidence.
constraints must contain price and use_case.
expanded_queries must contain at most {int(max_expanded_queries)} strings.

Query: {query_text}
Known categories: {categories}
Known brands: {brands}
Max expanded queries: {int(max_expanded_queries)}
"""
