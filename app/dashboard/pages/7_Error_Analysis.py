"""Error analysis page."""

from __future__ import annotations

from app.dashboard.components import render_header, render_markdown_report
from app.dashboard.data_loader import load_all_demo_artifacts, load_dashboard_config
from app.dashboard.demo_helpers import summarize_leaderboard


def main() -> None:
    import streamlit as st

    config = load_dashboard_config()
    artifacts = load_all_demo_artifacts(config)
    render_header("Error Analysis")
    leaderboard = artifacts.get("final_leaderboard_path")
    if leaderboard is not None:
        best = summarize_leaderboard(leaderboard)
        stages = best["stage"].astype(str).tolist() if not best.empty else []
        if stages:
            st.selectbox("Stage", stages)
    render_markdown_report(artifacts.get("error_taxonomy_path"))


if __name__ == "__main__":
    main()
