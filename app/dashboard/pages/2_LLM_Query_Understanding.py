"""LLM query-understanding demo page."""

from __future__ import annotations

from app.api.service import DemoService
from app.dashboard.components import render_header
from app.dashboard.data_loader import load_dashboard_config


def main() -> None:
    import streamlit as st

    config = load_dashboard_config()
    render_header("LLM Query Understanding")
    query_text = st.text_input("Query text", "gift beauty")
    if query_text:
        service = DemoService(config)
        result = service.understand_query(query_text)
        st.json(result)
        st.caption("Default client is deterministic mock LLM; no API key is required.")


if __name__ == "__main__":
    main()
