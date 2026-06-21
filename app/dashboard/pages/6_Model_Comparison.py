"""Model comparison page."""

from __future__ import annotations

from app.dashboard.components import render_header, render_leaderboard_table
from app.dashboard.data_loader import load_all_demo_artifacts, load_dashboard_config
from app.dashboard.demo_helpers import summarize_leaderboard


def main() -> None:
    import streamlit as st

    config = load_dashboard_config()
    artifacts = load_all_demo_artifacts(config)
    render_header("Model Comparison")
    leaderboard = artifacts.get("final_leaderboard_path")
    if leaderboard is None:
        st.warning("Missing final leaderboard.")
        return
    st.subheader("Best Method Per Stage")
    render_leaderboard_table(summarize_leaderboard(leaderboard))
    st.subheader("Full Leaderboard")
    render_leaderboard_table(leaderboard)
    st.caption("Metrics are stage-specific and not directly comparable across stages.")


if __name__ == "__main__":
    main()
