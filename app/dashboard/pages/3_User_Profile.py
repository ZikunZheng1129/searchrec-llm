"""User profile demo page."""

from __future__ import annotations

from app.api.service import DemoService
from app.dashboard.components import render_header
from app.dashboard.data_loader import load_all_demo_artifacts, load_dashboard_config
from app.dashboard.demo_helpers import get_user_examples


def main() -> None:
    import streamlit as st

    config = load_dashboard_config()
    artifacts = load_all_demo_artifacts(config)
    render_header("User Profile")
    users = get_user_examples(artifacts.get("user_profiles_path"))
    user_id = st.selectbox("User ID", users) if users else st.text_input("User ID", "user_00001")
    top_k = st.slider("Recommendations", min_value=1, max_value=50, value=10)
    if user_id:
        result = DemoService(config).recommend(user_id, top_k=top_k)
        st.json(result.get("profile"))
        st.dataframe(result.get("items", []), use_container_width=True)
        st.caption("Synthetic behavior profile; no request-time training is run.")


if __name__ == "__main__":
    main()
