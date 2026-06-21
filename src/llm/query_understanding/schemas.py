"""Lightweight schema normalization for query-understanding outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class QueryUnderstandingResult:
    """Normalized query-understanding fields."""

    intent: str | None
    category: str | None
    brand: str | None
    price_constraint: str | None
    use_case: str | None
    rewritten_query: str
    expanded_queries: list[str]
    confidence: float | None


def _nullable_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "unknown"}:
        return None
    return text


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    output: list[str] = []
    for item in values:
        text = str(item).strip()
        if text and text not in output:
            output.append(text)
    return output


def validate_query_understanding_result(obj: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw LLM JSON object into flat query-understanding fields."""
    constraints = obj.get("constraints") if isinstance(obj.get("constraints"), dict) else {}
    rewritten = str(obj.get("rewritten_query") or "").strip()
    confidence_raw = obj.get("confidence")
    confidence = None
    if confidence_raw is not None:
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = None
    result = QueryUnderstandingResult(
        intent=_nullable_string(obj.get("intent")),
        category=_nullable_string(obj.get("category")),
        brand=_nullable_string(obj.get("brand")),
        price_constraint=_nullable_string(constraints.get("price") or obj.get("price_constraint")),
        use_case=_nullable_string(constraints.get("use_case") or obj.get("use_case")),
        rewritten_query=rewritten,
        expanded_queries=_string_list(obj.get("expanded_queries")),
        confidence=confidence,
    )
    return {
        "intent": result.intent,
        "category": result.category,
        "brand": result.brand,
        "price_constraint": result.price_constraint,
        "use_case": result.use_case,
        "rewritten_query": result.rewritten_query,
        "expanded_queries": result.expanded_queries,
        "confidence": result.confidence,
    }
