"""Pure dashboard helper functions for local demo pages."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def find_query_examples(query_item_pairs: pd.DataFrame | None, limit: int = 20) -> list[str]:
    """Return example query strings."""
    if query_item_pairs is None or query_item_pairs.empty or "query_text" not in query_item_pairs:
        return []
    return (
        query_item_pairs["query_text"].dropna().astype(str).drop_duplicates().head(limit).tolist()
    )


def get_query_candidates(
    query_text: str,
    artifacts: dict[str, Any],
    top_k: int = 10,
) -> pd.DataFrame:
    """Return candidates for an existing query text."""
    pairs = artifacts.get("query_item_pairs_path")
    candidates = artifacts.get("ranking_candidates_multimodal_path")
    if candidates is None:
        candidates = artifacts.get("ranking_candidates_path")
    if pairs is None or candidates is None or pairs.empty or candidates.empty:
        return _empty_df()
    matches = pairs[pairs["query_text"].astype(str).str.lower() == str(query_text).lower()]
    if matches.empty:
        return _empty_df()
    query_id = str(matches.sort_values(["split", "query_id"]).iloc[0]["query_id"])
    rows = candidates[candidates["query_id"].astype(str) == query_id].copy()
    score_cols = [
        column
        for column in [
            "multimodal_score",
            "hybrid_score",
            "model_score",
            "bm25_score",
            "dense_score",
        ]
        if column in rows.columns
    ]
    score_col = score_cols[0] if score_cols else None
    if score_col:
        rows = rows.sort_values([score_col, "candidate_item_id"], ascending=[False, True])
    return rows.head(top_k).reset_index(drop=True)


def get_user_examples(user_profiles: pd.DataFrame | None, limit: int = 20) -> list[str]:
    """Return example user IDs."""
    if user_profiles is None or user_profiles.empty or "user_id" not in user_profiles:
        return []
    return user_profiles["user_id"].dropna().astype(str).drop_duplicates().head(limit).tolist()


def summarize_leaderboard(final_leaderboard: pd.DataFrame | None) -> pd.DataFrame:
    """Select best row per stage from final leaderboard."""
    if final_leaderboard is None or final_leaderboard.empty:
        return _empty_df()
    sorted_board = final_leaderboard.sort_values(
        ["stage", "primary_score", "secondary_score", "coverage_score", "avg_latency_ms", "method"],
        ascending=[True, False, False, False, True, True],
    )
    return sorted_board.groupby("stage", sort=True).head(1).reset_index(drop=True)


def build_business_metric_summary(artifacts: dict[str, Any]) -> dict[str, Any]:
    """Build synthetic business/proxy metric summary."""
    items = artifacts.get("item_metadata_path")
    candidates = artifacts.get("ranking_candidates_multimodal_path")
    metrics: dict[str, Any] = {"synthetic_data_caveat": "Synthetic proxy metrics only."}
    if items is not None and not items.empty:
        metrics["num_items"] = int(len(items))
        metrics["num_categories"] = int(items["category"].nunique()) if "category" in items else 0
        metrics["avg_rating"] = float(pd.to_numeric(items["avg_rating"], errors="coerce").mean())
        metrics["avg_rating_count"] = float(
            pd.to_numeric(items["rating_count"], errors="coerce").mean()
        )
    if candidates is not None and not candidates.empty:
        for column in ["authority_score", "conversion_proxy", "item_event_weight_sum"]:
            if column in candidates.columns:
                metrics[f"avg_{column}"] = float(
                    pd.to_numeric(candidates[column], errors="coerce").mean()
                )
    return metrics


def format_item_card(row: pd.Series | dict) -> dict[str, Any]:
    """Convert item/candidate row into display-ready fields."""
    data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    return {
        "item_id": str(data.get("item_id", data.get("candidate_item_id", ""))),
        "title": str(data.get("title", "")),
        "category": str(data.get("category", "")),
        "brand": str(data.get("brand", "")),
        "score": float(
            data.get(
                "score",
                data.get("multimodal_score", data.get("hybrid_score", 0.0)),
            )
            or 0.0
        ),
    }


def get_command_hints_for_missing_artifacts(status_df: pd.DataFrame) -> list[str]:
    """Return command hints for missing artifacts."""
    if status_df.empty:
        return []
    missing = status_df[(~status_df["exists"]) & status_df["command_hint"].notna()]
    return missing["command_hint"].drop_duplicates().astype(str).tolist()
