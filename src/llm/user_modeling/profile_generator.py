"""LLM-assisted user profile generation from observed behavior."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.llm.clients.base_client import BaseLLMClient
from src.llm.prompts.user_profile_prompts import (
    build_user_profile_prompt,
    build_user_profile_system_prompt,
)
from src.llm.utils import safe_json_loads


def _price_preference(avg_price: float | None) -> str:
    if avg_price is None or pd.isna(avg_price):
        return "unknown"
    if avg_price < 50:
        return "budget"
    if avg_price < 150:
        return "mid_range"
    return "premium"


def _event_counts(group: pd.DataFrame) -> dict[str, int]:
    event_type = group["event_type"].astype(str) if "event_type" in group else pd.Series([])
    return {
        "purchase_count": int((event_type == "purchase").sum()),
        "add_to_cart_count": int((event_type == "add_to_cart").sum()),
        "click_count": int((event_type == "click").sum()),
        "view_count": int((event_type == "view").sum()),
    }


def build_user_behavior_summary(
    user_id: str,
    train_interactions: pd.DataFrame,
    items: pd.DataFrame,
    max_recent_items: int = 5,
) -> dict[str, Any]:
    """Summarize observed user behavior with item metadata evidence only."""
    interactions = train_interactions.copy()
    interactions["user_id"] = interactions["user_id"].astype(str)
    user_rows = interactions[interactions["user_id"] == str(user_id)].copy()
    if user_rows.empty:
        return {
            "user_id": str(user_id),
            "history_length": 0,
            "top_categories": [],
            "top_brands": [],
            "recent_interests": [],
            "behavior_signals": [],
            "price_preference": "unknown",
            "avg_price": None,
            **_event_counts(user_rows),
        }
    item_table = items.copy()
    item_table["item_id"] = item_table["item_id"].astype(str)
    merged = user_rows.merge(item_table, on="item_id", how="left")
    if "timestamp" in merged.columns:
        merged = merged.sort_values("timestamp")
    weight = (
        pd.to_numeric(merged["event_weight"], errors="coerce").fillna(1.0)
        if "event_weight" in merged.columns
        else pd.Series([1.0] * len(merged), index=merged.index)
    )

    def top_values(column: str) -> list[str]:
        if column not in merged.columns:
            return []
        weighted = (
            pd.DataFrame({"value": merged[column].astype(str), "weight": weight})
            .groupby("value", sort=True)["weight"]
            .sum()
            .sort_values(ascending=False)
        )
        return [value for value in weighted.index.tolist() if value and value != "nan"][:3]

    recent_titles = (
        merged["title"].dropna().astype(str).tail(int(max_recent_items)).tolist()
        if "title" in merged.columns
        else []
    )
    avg_price = (
        float(pd.to_numeric(merged["price"], errors="coerce").dropna().mean())
        if "price" in merged.columns and not merged["price"].dropna().empty
        else None
    )
    events = sorted(merged["event_type"].dropna().astype(str).unique().tolist())
    counts = _event_counts(merged)
    return {
        "user_id": str(user_id),
        "history_length": int(len(merged)),
        "top_categories": top_values("category"),
        "top_brands": top_values("brand"),
        "recent_interests": recent_titles,
        "behavior_signals": events,
        "price_preference": _price_preference(avg_price),
        "avg_price": avg_price,
        **counts,
    }


def build_all_user_behavior_summaries(
    train_interactions: pd.DataFrame,
    items: pd.DataFrame,
    min_history_items: int = 1,
    max_recent_items: int = 5,
) -> list[dict[str, Any]]:
    """Build behavior summaries for every user with enough history."""
    users = train_interactions["user_id"].dropna().astype(str).sort_values().unique().tolist()
    summaries = [
        build_user_behavior_summary(user_id, train_interactions, items, max_recent_items)
        for user_id in users
    ]
    return [
        summary
        for summary in summaries
        if int(summary.get("history_length", 0)) >= int(min_history_items)
    ]


def _fallback_profile(summary: dict[str, Any]) -> dict[str, Any]:
    categories = [str(value) for value in summary.get("top_categories", [])]
    brands = [str(value) for value in summary.get("top_brands", [])]
    price = str(summary.get("price_preference", "unknown"))
    profile_text = (
        f"Interests: {', '.join(categories[:2]) or 'observed products'}. "
        f"Brands: {', '.join(brands[:2]) or 'observed brands'}. "
        f"Price preference: {price}."
    )
    return {
        "user_id": str(summary.get("user_id", "")),
        "summary": "Profile based only on observed synthetic product interactions.",
        "top_categories": categories,
        "top_brands": brands,
        "price_preference": price,
        "behavior_signals": [str(value) for value in summary.get("behavior_signals", [])],
        "recent_interests": [str(value) for value in summary.get("recent_interests", [])],
        "profile_text": profile_text,
        "confidence": 0.0,
    }


def _normalize_profile(obj: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    fallback = _fallback_profile(summary)
    normalized = dict(fallback)
    for key in [
        "user_id",
        "summary",
        "price_preference",
        "profile_text",
    ]:
        value = obj.get(key)
        if value is not None and str(value).strip():
            normalized[key] = str(value).strip()
    for key in ["top_categories", "top_brands", "behavior_signals", "recent_interests"]:
        value = obj.get(key)
        if isinstance(value, list):
            normalized[key] = [str(item) for item in value if str(item).strip()]
    try:
        normalized["confidence"] = float(obj.get("confidence", fallback["confidence"]))
    except (TypeError, ValueError):
        normalized["confidence"] = fallback["confidence"]
    return normalized


class LLMUserProfileGenerator:
    """Generate structured user profiles with LLM output and fallback."""

    def __init__(self, client: BaseLLMClient, max_recent_items: int = 5) -> None:
        self.client = client
        self.max_recent_items = int(max_recent_items)

    def generate_profile(self, behavior_summary: dict[str, Any]) -> dict[str, Any]:
        """Generate one profile from a behavior summary."""
        response = self.client.generate(
            prompt=build_user_profile_prompt(
                str(behavior_summary.get("user_id", "")),
                behavior_summary,
                max_recent_items=self.max_recent_items,
            ),
            system_prompt=build_user_profile_system_prompt(),
            temperature=0.0,
            max_tokens=512,
            response_format="json",
        )
        obj, parse_success, parse_error = safe_json_loads(response.text)
        schema_valid = bool(parse_success)
        profile = _normalize_profile(obj if parse_success else {}, behavior_summary)
        profile.update(
            {
                "parse_success": parse_success,
                "schema_valid": schema_valid,
                "parse_error": parse_error,
                "provider": response.provider,
                "model": response.model,
                "latency_ms": float(response.latency_ms),
                "estimated_cost_usd": float(response.estimated_cost_usd or 0.0),
                "history_length": int(behavior_summary.get("history_length", 0)),
                "purchase_count": int(behavior_summary.get("purchase_count", 0)),
                "add_to_cart_count": int(behavior_summary.get("add_to_cart_count", 0)),
                "click_count": int(behavior_summary.get("click_count", 0)),
                "view_count": int(behavior_summary.get("view_count", 0)),
            }
        )
        return profile

    def batch_generate_profiles(self, behavior_summaries: list[dict[str, Any]]) -> pd.DataFrame:
        """Generate profiles for a list of behavior summaries."""
        rows = [self.generate_profile(summary) for summary in behavior_summaries]
        return pd.DataFrame(rows).sort_values("user_id").reset_index(drop=True)
