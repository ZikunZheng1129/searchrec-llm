"""Candidate loading and formatting for candidate-grounded GenRec."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import resolve_project_path

DEFAULT_SCORE_PRIORITY = [
    "multimodal_score",
    "hybrid_score",
    "model_score",
    "bm25_score",
    "dense_score",
]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_project_path(str(candidate))


def load_candidate_source(config: dict[str, Any]) -> pd.DataFrame:
    """Load the preferred ranking candidate source from config."""
    input_config = config.get("input", {})
    source = str(config.get("genrec", {}).get("candidate_source", "auto"))
    multimodal_path = input_config.get("ranking_candidates_multimodal_path")
    ranking_path = input_config.get("ranking_candidates_path")

    if source in {"auto", "multimodal"} and multimodal_path:
        resolved = _resolve_path(multimodal_path)
        if resolved.exists():
            return pd.read_parquet(resolved)
        if source == "multimodal":
            raise FileNotFoundError(f"Multimodal ranking candidates not found: {resolved}")

    if source in {"auto", "ranking"} and ranking_path:
        resolved = _resolve_path(ranking_path)
        if resolved.exists():
            return pd.read_parquet(resolved)
        if source == "ranking":
            raise FileNotFoundError(f"Ranking candidates not found: {resolved}")

    raise FileNotFoundError(
        "No ranking candidate file found. Run Stage 7 first, and Stage 9 if "
        "multimodal candidates are desired."
    )


def _score_column(candidates: pd.DataFrame, score_priority: list[str] | None = None) -> str | None:
    for column in score_priority or DEFAULT_SCORE_PRIORITY:
        if column in candidates.columns:
            return column
    return None


def select_top_candidates_for_query(
    ranking_candidates: pd.DataFrame,
    query_id: str,
    top_n: int,
    score_priority: list[str] | None = None,
) -> pd.DataFrame:
    """Select deterministic top candidates for one query with duplicate removal."""
    if ranking_candidates.empty:
        return ranking_candidates.copy()
    query_rows = ranking_candidates[
        ranking_candidates["query_id"].astype(str) == str(query_id)
    ].copy()
    if query_rows.empty:
        return query_rows

    score_col = _score_column(query_rows, score_priority)
    if score_col is not None:
        query_rows["_genrec_score"] = pd.to_numeric(
            query_rows[score_col],
            errors="coerce",
        ).fillna(float("-inf"))
    else:
        query_rows["_genrec_score"] = 0.0
    rank_columns = [
        column
        for column in ["multimodal_rank", "hybrid_rank", "model_rank", "bm25_rank", "dense_rank"]
        if column in query_rows.columns
    ]
    rank_col = rank_columns[0] if rank_columns else None
    query_rows["_genrec_rank"] = (
        pd.to_numeric(query_rows[rank_col], errors="coerce").fillna(1_000_000)
        if rank_col is not None
        else range(1, len(query_rows) + 1)
    )
    query_rows["_candidate_item_id"] = query_rows["candidate_item_id"].astype(str)
    query_rows = query_rows.sort_values(
        ["_genrec_score", "_genrec_rank", "_candidate_item_id"],
        ascending=[False, True, True],
    )
    query_rows = query_rows.drop_duplicates("_candidate_item_id", keep="first")
    return query_rows.head(int(top_n)).drop(
        columns=["_genrec_score", "_genrec_rank", "_candidate_item_id"],
        errors="ignore",
    )


def truncate_candidate_text(text: str, max_chars: int = 240) -> str:
    """Truncate long evidence text for compact prompts."""
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 3)].rstrip() + "..."


def _safe_value(row: pd.Series | None, column: str, default: Any = "") -> Any:
    if row is None or column not in row or pd.isna(row[column]):
        return default
    return row[column]


def format_candidate_for_llm(
    row: pd.Series,
    item_row: pd.Series | None = None,
) -> dict[str, Any]:
    """Format a candidate row into a compact prompt-safe dictionary."""
    score_col = _score_column(pd.DataFrame([row]))
    rank_col = next(
        (
            column
            for column in [
                "multimodal_rank",
                "hybrid_rank",
                "model_rank",
                "bm25_rank",
                "dense_rank",
            ]
            if column in row
        ),
        None,
    )
    item_id = str(row.get("candidate_item_id", row.get("item_id", "")))
    title = str(_safe_value(item_row, "title", row.get("title", "")))
    category = str(_safe_value(item_row, "category", row.get("category", "")))
    brand = str(_safe_value(item_row, "brand", row.get("brand", "")))
    description = str(_safe_value(item_row, "description", row.get("description", "")))
    evidence_parts = [
        title,
        category,
        brand,
        description,
    ]
    return {
        "item_id": item_id,
        "title": title,
        "category": category,
        "brand": brand,
        "price": float(_safe_value(item_row, "price", row.get("price", 0.0)) or 0.0),
        "avg_rating": float(_safe_value(item_row, "avg_rating", row.get("avg_rating", 0.0)) or 0.0),
        "rating_count": int(_safe_value(item_row, "rating_count", row.get("rating_count", 0)) or 0),
        "score": float(row.get(score_col, 0.0)) if score_col is not None else 0.0,
        "rank": int(row.get(rank_col, 0)) if rank_col is not None else 0,
        "evidence_text": truncate_candidate_text(" ".join(evidence_parts)),
    }


def format_candidates_for_llm(
    candidates: pd.DataFrame,
    items: pd.DataFrame,
    max_candidates: int = 20,
) -> list[dict[str, Any]]:
    """Format candidate rows with item metadata for LLM prompts."""
    if candidates.empty:
        return []
    item_lookup = {str(row["item_id"]): row for _, row in items.astype({"item_id": str}).iterrows()}
    formatted = []
    seen: set[str] = set()
    for _, row in candidates.head(int(max_candidates)).iterrows():
        item_id = str(row.get("candidate_item_id", ""))
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        formatted.append(format_candidate_for_llm(row, item_lookup.get(item_id)))
    return formatted


def build_candidate_lookup(candidates: list[dict]) -> dict[str, dict]:
    """Build an item-id lookup from formatted candidates."""
    return {
        str(candidate["item_id"]): candidate for candidate in candidates if candidate.get("item_id")
    }
