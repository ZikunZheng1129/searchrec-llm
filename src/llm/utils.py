"""Utilities for structured LLM output."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> str:
    """Extract the first JSON object from plain or fenced text."""
    raw = str(text).strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start : end + 1]
    return raw


def safe_json_loads(text: str) -> tuple[dict[str, Any], bool, str | None]:
    """Parse LLM JSON output without raising on malformed text."""
    try:
        parsed = json.loads(extract_json_object(text))
    except json.JSONDecodeError as exc:
        return {}, False, str(exc)
    if not isinstance(parsed, dict):
        return {}, False, "Parsed JSON is not an object"
    return parsed, True, None


def validate_required_keys(obj: dict[str, Any], required_keys: list[str], name: str) -> None:
    """Raise a clear error if a structured object misses required keys."""
    missing = [key for key in required_keys if key not in obj]
    if missing:
        raise ValueError(f"{name} missing required keys: {missing}")


def estimate_mock_cost(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    provider: str,
    model: str,
) -> float:
    """Return a deterministic tiny cost estimate for local reporting."""
    if provider == "mock":
        return 0.0
    prompt = int(prompt_tokens or 0)
    completion = int(completion_tokens or 0)
    del model
    return float((prompt * 0.000001) + (completion * 0.000002))


def estimate_token_count(text: str) -> int:
    """Estimate tokens with a simple whitespace split for local accounting."""
    return len(re.findall(r"\S+", str(text)))
