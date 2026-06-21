"""Reusable Streamlit display components."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _st():
    import streamlit as st

    return st


def render_header(title: str = "TikSearchRec-LLM Demo") -> None:
    """Render a page header."""
    st = _st()
    st.title(title)
    st.caption("Local synthetic-data demo. No external API calls by default.")


def render_artifact_status(status_df: pd.DataFrame) -> None:
    """Render artifact status table."""
    st = _st()
    st.subheader("Artifact Status")
    st.dataframe(status_df, use_container_width=True)


def render_metric_cards(metrics: dict[str, Any]) -> None:
    """Render simple metric cards."""
    st = _st()
    columns = st.columns(min(4, max(1, len(metrics))))
    for index, (name, value) in enumerate(metrics.items()):
        with columns[index % len(columns)]:
            st.metric(name.replace("_", " ").title(), value)


def render_item_results(items_df: pd.DataFrame) -> None:
    """Render item results."""
    st = _st()
    st.dataframe(items_df, use_container_width=True)


def render_leaderboard_table(df: pd.DataFrame) -> None:
    """Render leaderboard table."""
    st = _st()
    st.dataframe(df, use_container_width=True)


def render_markdown_report(text: str | None) -> None:
    """Render markdown report or fallback message."""
    st = _st()
    st.markdown(text or "_Report not found._")


def render_command_hints(hints: list[str]) -> None:
    """Render missing-artifact command hints."""
    st = _st()
    if hints:
        st.warning("Some artifacts are missing. Run the relevant commands below.")
        for hint in hints:
            st.code(hint, language="bash")
