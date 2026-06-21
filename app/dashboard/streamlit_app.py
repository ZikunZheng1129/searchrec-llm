"""Streamlit landing page for the local TikSearchRec-LLM demo."""

from __future__ import annotations

from app.dashboard.components import (
    render_artifact_status,
    render_command_hints,
    render_header,
)
from app.dashboard.data_loader import artifact_status, load_dashboard_config
from app.dashboard.demo_helpers import get_command_hints_for_missing_artifacts


def main() -> None:
    """Render the dashboard landing page."""
    import streamlit as st

    config = load_dashboard_config()
    dashboard = config.get("dashboard", {})
    st.set_page_config(
        page_title=str(dashboard.get("title", "TikSearchRec-LLM Demo")),
        page_icon=str(dashboard.get("page_icon", "search")),
        layout=str(dashboard.get("layout", "wide")),
    )
    render_header(str(dashboard.get("title", "TikSearchRec-LLM Demo")))
    st.markdown(
        """
        This local dashboard exposes the existing TikSearchRec-LLM debug artifacts:
        search/ranking, mock LLM query understanding, user profiles, candidate-
        constrained GenRec, model comparison, error analysis, and proxy business
        metrics.

        The default mode reads local parquet and CSV files. It does not require
        external API keys.
        """
    )
    status = artifact_status(config)
    render_artifact_status(status)
    render_command_hints(get_command_hints_for_missing_artifacts(status))
    st.subheader("Launch Commands")
    st.code("bash scripts/launch_api.sh", language="bash")
    st.code("bash scripts/launch_dashboard.sh", language="bash")
    st.info("All current metrics use synthetic debug data and are not production claims.")


if __name__ == "__main__":
    main()
