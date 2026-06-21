"""Search demo page."""

from __future__ import annotations

from app.dashboard.components import render_header, render_item_results
from app.dashboard.data_loader import load_all_demo_artifacts, load_dashboard_config
from app.dashboard.demo_helpers import find_query_examples, get_query_candidates


def main() -> None:
    import streamlit as st

    config = load_dashboard_config()
    artifacts = load_all_demo_artifacts(config)
    render_header("Search Demo")
    examples = find_query_examples(artifacts.get("query_item_pairs_path"))
    selected = st.selectbox("Example query", examples) if examples else ""
    query_text = st.text_input("Query text", selected)
    top_k = st.slider("Top K", min_value=1, max_value=50, value=10)
    if query_text:
        candidates = get_query_candidates(query_text, artifacts, top_k=top_k)
        st.caption("Candidate source prefers multimodal ranking candidates when available.")
        render_item_results(candidates)


if __name__ == "__main__":
    main()
