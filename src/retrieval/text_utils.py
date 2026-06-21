"""Text helpers for local retrieval baselines."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import pandas as pd

DEFAULT_ITEM_TEXT_FIELDS = ["title", "category", "brand", "description"]


def normalize_text(text: str) -> str:
    """Lowercase text and collapse extra whitespace."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def tokenize(text: str) -> list[str]:
    """Tokenize text with a simple deterministic regex."""
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def build_item_text(
    row: pd.Series,
    fields: Sequence[str] | None = None,
) -> str:
    """Combine item metadata fields into searchable text."""
    selected_fields = list(fields or DEFAULT_ITEM_TEXT_FIELDS)
    values: list[str] = []
    for field in selected_fields:
        value: Any = row.get(field, "")
        if pd.notna(value):
            values.append(str(value))
    return normalize_text(" ".join(values))
