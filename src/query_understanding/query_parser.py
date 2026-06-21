"""Rule-based query parsing."""

from __future__ import annotations

import re
from typing import Any

from src.query_understanding.intent_classifier import classify_intent

PRICE_WORDS = ["affordable", "budget", "cheap", "premium"]
USE_CASES = [
    "beginner",
    "gym",
    "running",
    "travel",
    "dry skin",
    "walking",
    "outdoor",
    "office",
    "school",
    "gift",
]


def normalize_query(query_text: str) -> str:
    """Lowercase a query and collapse extra whitespace."""
    return re.sub(r"\s+", " ", query_text.strip().lower())


def extract_price_constraint(query_text: str) -> str | None:
    """Extract a simple price constraint from query text."""
    normalized = normalize_query(query_text)
    if not normalized:
        return None

    under_match = re.search(r"\bunder\s+\$?(\d+(?:\.\d+)?)\b", normalized)
    if under_match:
        amount = under_match.group(1).replace(".", "_")
        return f"under_{amount}"

    for word in PRICE_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", normalized):
            return word

    return None


def extract_use_case(query_text: str) -> str | None:
    """Extract a supported use-case phrase from query text."""
    normalized = normalize_query(query_text)
    for use_case in USE_CASES:
        if re.search(rf"\b{re.escape(use_case)}\b", normalized):
            return use_case
    return None


def extract_category(query_text: str, known_categories: list[str]) -> str | None:
    """Extract the first matching known category."""
    normalized = normalize_query(query_text)
    for category in known_categories:
        normalized_category = normalize_query(category)
        category_tokens = [
            token for token in re.findall(r"[a-z0-9]+", normalized_category) if token != "and"
        ]
        if normalized_category and normalized_category in normalized:
            return category
        has_category_token = any(
            re.search(rf"\b{re.escape(token)}\b", normalized) for token in category_tokens
        )
        if category_tokens and has_category_token:
            return category
    return None


def extract_brand(query_text: str, known_brands: list[str]) -> str | None:
    """Extract the first matching known brand."""
    normalized = normalize_query(query_text)
    for brand in known_brands:
        normalized_brand = normalize_query(brand)
        if normalized_brand and re.search(rf"\b{re.escape(normalized_brand)}\b", normalized):
            return brand
    return None


def parse_query(
    query_text: str,
    known_categories: list[str] | None = None,
    known_brands: list[str] | None = None,
) -> dict[str, Any]:
    """Parse a query into lightweight rule-based fields."""
    normalized = normalize_query(query_text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    categories = known_categories or []
    brands = known_brands or []

    return {
        "query_text": query_text,
        "normalized_query": normalized,
        "intent": classify_intent(normalized),
        "category": extract_category(normalized, categories),
        "brand": extract_brand(normalized, brands),
        "price_constraint": extract_price_constraint(normalized),
        "use_case": extract_use_case(normalized),
        "tokens": tokens,
    }
