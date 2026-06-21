"""Evidence selection for recommendation explanations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.llm.genrec.candidate_formatter import DEFAULT_SCORE_PRIORITY, truncate_candidate_text


def _item_row(item_id: str, item_metadata: pd.DataFrame) -> pd.Series | None:
    matches = item_metadata[item_metadata["item_id"].astype(str) == str(item_id)]
    if matches.empty:
        return None
    return matches.iloc[0]


def _candidate_row(
    item_id: str,
    ranking_candidates: pd.DataFrame | None,
) -> pd.Series | None:
    if ranking_candidates is None or ranking_candidates.empty:
        return None
    matches = ranking_candidates[
        ranking_candidates["candidate_item_id"].astype(str) == str(item_id)
    ]
    if matches.empty:
        return None
    return matches.iloc[0]


def _value(row: pd.Series | None, column: str, default: Any = "") -> Any:
    if row is None or column not in row or pd.isna(row[column]):
        return default
    return row[column]


def _ranking_score(row: pd.Series | None) -> float:
    if row is None:
        return 0.0
    for column in DEFAULT_SCORE_PRIORITY:
        if column in row and pd.notna(row[column]):
            return float(row[column])
    return 0.0


def build_evidence_text(evidence: dict) -> str:
    """Build compact evidence text from metadata and candidate scores."""
    parts = [
        str(evidence.get("title", "")),
        str(evidence.get("category", "")),
        str(evidence.get("brand", "")),
        f"rating {float(evidence.get('avg_rating', 0.0) or 0.0):.2f}",
        f"price {float(evidence.get('price', 0.0) or 0.0):.2f}",
    ]
    return truncate_candidate_text(" ".join(parts))


def select_item_evidence(
    item_id: str,
    item_metadata: pd.DataFrame,
    ranking_candidates: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Select explanation evidence from item metadata and candidate rows only."""
    item = _item_row(item_id, item_metadata)
    candidate = _candidate_row(item_id, ranking_candidates)
    evidence = {
        "item_id": str(item_id),
        "title": str(_value(item, "title", "")),
        "category": str(_value(item, "category", _value(candidate, "category", ""))),
        "brand": str(_value(item, "brand", _value(candidate, "brand", ""))),
        "price": float(_value(item, "price", _value(candidate, "price", 0.0)) or 0.0),
        "avg_rating": float(
            _value(item, "avg_rating", _value(candidate, "avg_rating", 0.0)) or 0.0
        ),
        "rating_count": int(
            _value(item, "rating_count", _value(candidate, "rating_count", 0)) or 0
        ),
        "retrieval_score": _ranking_score(candidate),
    }
    evidence["evidence_text"] = build_evidence_text(evidence)
    return evidence


def select_evidence_for_recommendations(
    recommended_item_ids: list[str],
    items: pd.DataFrame,
    ranking_candidates_for_query: pd.DataFrame | None = None,
    max_items: int = 5,
) -> list[dict[str, Any]]:
    """Select evidence for a recommendation list."""
    evidence = []
    for item_id in recommended_item_ids[: int(max_items)]:
        evidence.append(select_item_evidence(item_id, items, ranking_candidates_for_query))
    return evidence
