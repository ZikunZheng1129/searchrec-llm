"""GenRec demo page."""

from __future__ import annotations

from app.api.service import DemoService
from app.dashboard.components import render_header
from app.dashboard.data_loader import load_all_demo_artifacts, load_dashboard_config
from app.dashboard.demo_helpers import find_query_examples


def main() -> None:
    import streamlit as st

    config = load_dashboard_config()
    artifacts = load_all_demo_artifacts(config)
    render_header("GenRec Demo")
    examples = find_query_examples(artifacts.get("query_item_pairs_path"))
    selected = st.selectbox("Example query", examples) if examples else "gift beauty"
    query_text = st.text_input("Query text", selected)
    method = st.selectbox(
        "Method",
        ["candidate_constrained_generation", "llm_rerank_top_10", "llm_rerank_top_20"],
    )
    if query_text:
        result = DemoService(config).genrec(query_text=query_text, method=method)
        st.json(
            {
                "valid_item_rate": result["valid_item_rate"],
                "hallucination_rate": result["hallucination_rate"],
                "used_fallback": result["used_fallback"],
            }
        )
        st.dataframe(result["recommended_items"], use_container_width=True)
        st.subheader("Evidence Explanations")
        st.json(result["explanations"])
        st.caption(
            "Direct generation is a baseline only; this page defaults to constrained methods."
        )


if __name__ == "__main__":
    main()
