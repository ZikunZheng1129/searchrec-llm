"""Prompt templates for candidate-grounded GenRec experiments."""

from __future__ import annotations

import json
from typing import Any


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_jsonable(value.tolist())
    if isinstance(value, float) and value != value:
        return None
    return value


def _json_payload(obj: Any) -> str:
    return json.dumps(_to_jsonable(obj or {}), ensure_ascii=True, sort_keys=True)


def build_genrec_system_prompt() -> str:
    """Return the system prompt for structured GenRec tasks."""
    return (
        "You are a catalog-grounded recommendation assistant. Return structured JSON only. "
        "For constrained tasks, choose only from the provided candidate item IDs and do not "
        "invent item IDs or product facts."
    )


def build_explanation_system_prompt() -> str:
    """Return the system prompt for evidence-grounded explanation tasks."""
    return (
        "You write short recommendation explanations grounded only in provided evidence. "
        "Return structured JSON only and do not invent product claims."
    )


def build_direct_generation_prompt(
    query_text: str,
    parsed_query: dict | None = None,
    user_profile: dict | None = None,
    top_k: int = 10,
) -> str:
    """Build a deliberately unconstrained direct-generation baseline prompt."""
    return "\n".join(
        [
            "GENREC_TASK: DIRECT_GENERATION",
            f"Query: {query_text}",
            f"Top K: {int(top_k)}",
            f"Parsed query JSON: {_json_payload(parsed_query)}",
            f"User profile JSON: {_json_payload(user_profile)}",
            (
                "Return JSON with recommended_item_ids and ranked_items. This baseline is "
                "experimental and may generate item IDs without a candidate list."
            ),
        ]
    )


def build_candidate_constrained_prompt(
    query_text: str,
    candidates: list[dict],
    parsed_query: dict | None = None,
    user_profile: dict | None = None,
    top_k: int = 10,
) -> str:
    """Build a candidate-constrained generation prompt."""
    return "\n".join(
        [
            "GENREC_TASK: CANDIDATE_CONSTRAINED_GENERATION",
            f"Query: {query_text}",
            f"Top K: {int(top_k)}",
            f"Parsed query JSON: {_json_payload(parsed_query)}",
            f"User profile JSON: {_json_payload(user_profile)}",
            f"Candidates JSON: {_json_payload({'candidates': candidates})}",
            (
                "Only choose item IDs from the provided candidates. Do not invent item IDs. "
                "Do not mention facts not present in evidence. Return null or an empty list "
                "if no candidate is suitable."
            ),
            "Return JSON with recommended_item_ids and ranked_items.",
        ]
    )


def build_llm_rerank_prompt(
    query_text: str,
    candidates: list[dict],
    parsed_query: dict | None = None,
    user_profile: dict | None = None,
    top_k: int = 10,
) -> str:
    """Build a candidate-constrained LLM reranking prompt."""
    return "\n".join(
        [
            "GENREC_TASK: LLM_RERANK",
            f"Query: {query_text}",
            f"Top K: {int(top_k)}",
            f"Parsed query JSON: {_json_payload(parsed_query)}",
            f"User profile JSON: {_json_payload(user_profile)}",
            f"Candidates JSON: {_json_payload({'candidates': candidates})}",
            (
                "Rerank only the provided candidate item IDs. Do not invent item IDs. "
                "Do not mention facts not present in evidence."
            ),
            "Return JSON with recommended_item_ids and ranked_items.",
        ]
    )


def build_explanation_prompt(
    query_text: str,
    recommended_items: list[dict],
    evidence: list[dict],
    parsed_query: dict | None = None,
    user_profile: dict | None = None,
) -> str:
    """Build an evidence-grounded explanation prompt."""
    return "\n".join(
        [
            "GENREC_TASK: EVIDENCE_GROUNDED_EXPLANATION",
            f"Query: {query_text}",
            f"Parsed query JSON: {_json_payload(parsed_query)}",
            f"User profile JSON: {_json_payload(user_profile)}",
            f"Recommended items JSON: {_json_payload({'recommended_items': recommended_items})}",
            f"Evidence JSON: {_json_payload({'evidence': evidence})}",
            (
                "The explanation must be grounded in provided item evidence. Do not invent "
                "claims about products. Cite candidate item IDs in explanation evidence fields."
            ),
            "Return JSON with query_id and explanations.",
        ]
    )
