"""Synthetic business/proxy metrics page."""

from __future__ import annotations

from app.dashboard.components import render_header, render_metric_cards
from app.dashboard.data_loader import load_all_demo_artifacts, load_dashboard_config
from app.dashboard.demo_helpers import build_business_metric_summary


def main() -> None:
    import streamlit as st

    config = load_dashboard_config()
    artifacts = load_all_demo_artifacts(config)
    render_header("Business Metrics")
    metrics = build_business_metric_summary(artifacts)
    render_metric_cards(metrics)
    items = artifacts.get("item_metadata_path")
    if items is not None and "avg_rating" in items:
        st.subheader("Rating Distribution")
        st.bar_chart(items["avg_rating"])
    st.caption("Synthetic proxy metrics only; not real GMV or production business results.")


if __name__ == "__main__":
    main()
