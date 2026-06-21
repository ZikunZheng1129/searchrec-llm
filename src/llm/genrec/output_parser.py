"""Structured GenRec output parsing, validation, and fallback logic."""

from __future__ import annotations

from typing import Any

from src.llm.utils import safe_json_loads


def _ordered_unique(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        item_id = str(value).strip()
        if item_id and item_id not in seen:
            seen.add(item_id)
            output.append(item_id)
    return output


def _base_output(parse_success: bool, parse_error: str | None = None) -> dict[str, Any]:
    return {
        "recommended_item_ids": [],
        "ranked_items": [],
        "parse_success": bool(parse_success),
        "schema_valid": bool(parse_success),
        "invalid_item_ids": [],
        "used_fallback": False,
        "parse_error": parse_error,
    }


def parse_recommendation_output(text: str) -> dict[str, Any]:
    """Parse a recommendation JSON object without raising."""
    obj, success, error = safe_json_loads(text)
    if not success:
        return _base_output(False, error)
    output = _base_output(True)
    output.update(obj)
    return normalize_recommendation_items(output)


def parse_rerank_output(text: str) -> dict[str, Any]:
    """Parse a reranker JSON object without raising."""
    return parse_recommendation_output(text)


def parse_explanation_output(text: str) -> dict[str, Any]:
    """Parse an explanation JSON object without raising."""
    obj, success, error = safe_json_loads(text)
    if not success:
        return {
            "query_id": "",
            "explanations": [],
            "parse_success": False,
            "schema_valid": False,
            "used_fallback": False,
            "parse_error": error,
        }
    explanations = obj.get("explanations", [])
    if not isinstance(explanations, list):
        explanations = []
    normalized = []
    for explanation in explanations:
        if not isinstance(explanation, dict):
            continue
        evidence_item_ids = explanation.get("evidence_item_ids", [])
        evidence_fields = explanation.get("evidence_fields", [])
        normalized.append(
            {
                "item_id": str(explanation.get("item_id", "")),
                "explanation": str(explanation.get("explanation", "")),
                "evidence_item_ids": _ordered_unique(evidence_item_ids),
                "evidence_fields": [str(field) for field in evidence_fields],
                "confidence": float(explanation.get("confidence", 0.0) or 0.0),
            }
        )
    return {
        "query_id": str(obj.get("query_id", "")),
        "explanations": normalized,
        "parse_success": True,
        "schema_valid": bool(normalized),
        "used_fallback": False,
        "parse_error": None,
    }


def normalize_recommendation_items(
    obj: dict[str, Any],
    candidate_item_ids: set[str] | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Normalize recommendation output to the Stage 10 schema."""
    output = _base_output(bool(obj.get("parse_success", True)), obj.get("parse_error"))
    ranked_items = obj.get("ranked_items", [])
    ids = obj.get("recommended_item_ids", [])
    if not isinstance(ids, list):
        ids = []
    if isinstance(ranked_items, list):
        for ranked in ranked_items:
            if isinstance(ranked, dict) and ranked.get("item_id") is not None:
                ids.append(ranked["item_id"])
    else:
        ranked_items = []

    normalized_ids = _ordered_unique(ids)[: int(top_k)]
    invalid_ids = []
    if candidate_item_ids is not None:
        invalid_ids = [item_id for item_id in normalized_ids if item_id not in candidate_item_ids]

    valid_ids = (
        [item_id for item_id in normalized_ids if item_id in candidate_item_ids]
        if candidate_item_ids is not None
        else normalized_ids
    )
    normalized_ranked = []
    reasons_by_id = {
        str(row.get("item_id")): row
        for row in ranked_items
        if isinstance(row, dict) and row.get("item_id") is not None
    }
    for rank, item_id in enumerate(valid_ids, start=1):
        source = reasons_by_id.get(item_id, {})
        normalized_ranked.append(
            {
                "item_id": item_id,
                "rank": rank,
                "reason": str(source.get("reason", "")),
                "confidence": float(source.get("confidence", 0.0) or 0.0),
            }
        )

    output.update(
        {
            "recommended_item_ids": valid_ids,
            "ranked_items": normalized_ranked,
            "schema_valid": bool(obj.get("schema_valid", True)) and not invalid_ids,
            "invalid_item_ids": invalid_ids,
            "used_fallback": bool(obj.get("used_fallback", False)),
        }
    )
    return output


def validate_candidate_constrained_output(
    obj: dict[str, Any],
    candidate_item_ids: set[str],
) -> dict[str, Any]:
    """Validate that all output IDs are allowed candidate IDs."""
    return normalize_recommendation_items(
        obj,
        candidate_item_ids={str(item_id) for item_id in candidate_item_ids},
        top_k=len(obj.get("recommended_item_ids", [])) or 10,
    )


def fallback_recommendations_from_candidates(
    candidates: list[dict],
    top_k: int = 10,
) -> dict[str, Any]:
    """Return a valid fallback recommendation from original candidate order."""
    item_ids = _ordered_unique([candidate.get("item_id") for candidate in candidates])[: int(top_k)]
    return {
        "recommended_item_ids": item_ids,
        "ranked_items": [
            {
                "item_id": item_id,
                "rank": rank,
                "reason": "Fallback to original ranked candidate order.",
                "confidence": 0.0,
            }
            for rank, item_id in enumerate(item_ids, start=1)
        ],
        "parse_success": True,
        "schema_valid": True,
        "invalid_item_ids": [],
        "used_fallback": True,
        "parse_error": None,
    }
