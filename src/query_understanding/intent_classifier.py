"""Rule-based query intent classification."""

from __future__ import annotations

import re

PRICE_WORDS = {"cheap", "budget", "affordable", "under", "low cost", "deal", "discount"}
USE_CASE_WORDS = {
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
}
CATEGORY_WORDS = {
    "sports",
    "outdoors",
    "beauty",
    "electronics",
    "clothing",
    "health",
    "personal care",
    "skincare",
}
KNOWN_SYNTHETIC_BRANDS = {
    "nova",
    "orbit",
    "luma",
    "stride",
    "haven",
    "pulse",
    "aster",
    "kindlewood",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def classify_intent(query_text: str) -> str:
    """Classify a query with deterministic keyword rules."""
    normalized = _normalize(query_text)
    if not normalized:
        return "unknown"

    if any(word in normalized for word in PRICE_WORDS) or re.search(r"\bunder\s+\d+", normalized):
        return "price_sensitive_search"

    if any(word in normalized for word in USE_CASE_WORDS):
        return "use_case_search"

    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    if tokens & KNOWN_SYNTHETIC_BRANDS:
        return "brand_search"

    category_hits = sum(1 for word in CATEGORY_WORDS if word in normalized)
    if category_hits and len(tokens) <= 5:
        return "category_search"

    return "product_search"
