"""
NYC Taxi Tips Predictor & Analyst Dashboard Main Module.

This module serves as the primary entry point for the Streamlit application,
handling the high-level layout, sidebar navigation, and page routing.
"""

import streamlit as st

from src.app.data_loaders import (
    API_URL,
    check_api_health,
    load_metrics,
    load_params,
    load_predictions,
)
from src.app.pages.chat import render_chat
from src.app.pages.dashboard import render_dashboard
from src.app.styles import apply_custom_css


def main() -> None:
    """Main entrypoint for the Streamlit dashboard app."""
    # Set page config
    st.set_page_config(
        page_title="NYC Taxi Tips Predictor",
        page_icon="🚕",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply custom CSS
    apply_custom_css()

    # Load artifacts and states
    metrics = load_metrics()
    predictions_df = load_predictions()
    api_is_healthy, model_name = check_api_health()
    params = load_params()

    st.title("🚕 NYC Taxi Tips Predictor & Analyst")
    st.markdown(
        "*A production-ready FTI architecture serving "
        f"({model_name} model) predictions via FastAPI.*"
    )

    if metrics is None or predictions_df is None:
        st.error(
            "⚠️ Initial artifacts not found. Please run the DVC pipeline (`uv run dvc repro`) first."
        )
        st.stop()

    if not api_is_healthy:
        st.warning(
            "⚠️ FastAPI Backend is not reachable. Ensure the server is running "
            "(`uv run uvicorn src.api.predict_api:app`). Real-time predictions will fail."
        )

    # Sidebar routing
    st.sidebar.image(
        "reports/figures/nyc_taxi_logo.jpg",
        width=150,
    )
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigate", ["📊 Dashboard & Evaluation", "⚡ Interactive Prediction"])

    if page == "📊 Dashboard & Evaluation":
        render_dashboard(
            metrics=metrics if metrics is not None else {},
            predictions_df=predictions_df,
            params=params,
            api_url=API_URL,
        )
    elif page == "⚡ Interactive Prediction":
        render_chat()


if __name__ == "__main__":
    main()
