"""Rule-based validity and hallucination checks for Stage 10."""

from __future__ import annotations

from typing import Any


def check_valid_item_ids(output_item_ids: list[str], valid_item_ids: set[str]) -> dict[str, Any]:
    """Check output IDs against the full catalog."""
    valid_set = {str(item_id) for item_id in valid_item_ids}
    output = [str(item_id) for item_id in output_item_ids]
    invalid = [item_id for item_id in output if item_id not in valid_set]
    valid_count = len(output) - len(invalid)
    total = len(output)
    return {
        "valid_item_count": valid_count,
        "invalid_item_count": len(invalid),
        "valid_item_rate": valid_count / total if total else 0.0,
        "invalid_item_ids": invalid,
        "hallucination_rate": len(invalid) / total if total else 0.0,
    }


def check_candidate_constrained_item_ids(
    output_item_ids: list[str],
    candidate_item_ids: set[str],
) -> dict[str, Any]:
    """Check output IDs against the allowed candidate set."""
    return check_valid_item_ids(output_item_ids, candidate_item_ids)


def _mentioned_unavailable_value(text: str, values: set[str]) -> bool:
    normalized = text.lower()
    return any(value.lower() in normalized for value in values if value)


def check_explanation_faithfulness(
    explanations: list[dict],
    evidence_by_item_id: dict[str, dict],
) -> dict[str, Any]:
    """Run simple rule-based explanation faithfulness checks."""
    ungrounded_claim_count = 0
    missing_evidence_count = 0
    valid_evidence_ids = set(evidence_by_item_id)
    allowed_categories = {
        str(evidence.get("category", ""))
        for evidence in evidence_by_item_id.values()
        if evidence.get("category")
    }
    allowed_brands = {
        str(evidence.get("brand", ""))
        for evidence in evidence_by_item_id.values()
        if evidence.get("brand")
    }
    for explanation in explanations:
        item_id = str(explanation.get("item_id", ""))
        evidence_ids = {str(value) for value in explanation.get("evidence_item_ids", [])}
        text = str(explanation.get("explanation", ""))
        if item_id not in valid_evidence_ids or not evidence_ids.issubset(valid_evidence_ids):
            missing_evidence_count += 1
        item_evidence = evidence_by_item_id.get(item_id, {})
        unavailable_categories = allowed_categories - {str(item_evidence.get("category", ""))}
        unavailable_brands = allowed_brands - {str(item_evidence.get("brand", ""))}
        if _mentioned_unavailable_value(text, unavailable_categories | unavailable_brands):
            ungrounded_claim_count += 1
    return {
        "explanation_faithful": missing_evidence_count == 0 and ungrounded_claim_count == 0,
        "ungrounded_claim_count": ungrounded_claim_count,
        "missing_evidence_count": missing_evidence_count,
    }


def compute_hallucination_flags(row: dict) -> dict[str, Any]:
    """Compute hallucination flags from a row-like output object."""
    invalid_item_ids = [str(value) for value in row.get("invalid_item_ids", [])]
    output_item_ids = [str(value) for value in row.get("recommended_item_ids", [])]
    denominator = len(output_item_ids) + len(invalid_item_ids)
    return {
        "invalid_item_count": len(invalid_item_ids),
        "hallucination_rate": len(invalid_item_ids) / denominator if denominator else 0.0,
    }
