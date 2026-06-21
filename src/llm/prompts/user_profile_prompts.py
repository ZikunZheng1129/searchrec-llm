"""Prompt templates for behavior-based user profile generation."""

from __future__ import annotations

import json
from typing import Any


def build_user_profile_system_prompt() -> str:
    """Return the user-profile system prompt."""
    return (
        "You summarize shopping behavior from provided evidence only. "
        "Return JSON only. Do not infer sensitive attributes."
    )


def build_user_profile_prompt(
    user_id: str,
    behavior_summary: dict[str, Any],
    max_recent_items: int = 5,
) -> str:
    """Build a concise structured profile prompt."""
    compact_summary = dict(behavior_summary)
    if "recent_interests" in compact_summary:
        compact_summary["recent_interests"] = list(compact_summary["recent_interests"])[
            : int(max_recent_items)
        ]
    return f"""USER_PROFILE_TASK
Return one JSON object with exactly these fields:
user_id, summary, top_categories, top_brands, price_preference, behavior_signals,
recent_interests, profile_text, confidence.
Base the profile only on the provided interaction and item evidence.

User ID: {user_id}
Behavior summary: {json.dumps(compact_summary, sort_keys=True)}
"""
