"""Ranking pipeline page."""

from __future__ import annotations

from app.dashboard.components import render_header
from app.dashboard.data_loader import load_all_demo_artifacts, load_dashboard_config
from app.dashboard.demo_helpers import find_query_examples, get_query_candidates


def main() -> None:
    import streamlit as st

    config = load_dashboard_config()
    artifacts = load_all_demo_artifacts(config)
    render_header("Ranking Pipeline")
    examples = find_query_examples(artifacts.get("query_item_pairs_path"))
    query_text = st.selectbox("Query", examples) if examples else ""
    top_k = st.slider("Top candidates", min_value=1, max_value=50, value=20)
    candidates = get_query_candidates(query_text, artifacts, top_k=top_k) if query_text else None
    if candidates is not None:
        st.subheader("Ranking Candidates")
        st.dataframe(candidates, use_container_width=True)
    ranking_results = artifacts.get("ranking_results_path")
    if ranking_results is not None:
        st.subheader("Stage 7 Ranking Metrics")
        st.dataframe(ranking_results, use_container_width=True)


if __name__ == "__main__":
    main()
