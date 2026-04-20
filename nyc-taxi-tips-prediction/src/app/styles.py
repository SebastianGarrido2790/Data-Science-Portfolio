"""
Aesthetics and styling for the NYC Taxi Tips Streamlit app.

This module provides custom CSS to align with the project's premium design standards.
"""

import streamlit as st


def apply_custom_css() -> None:
    """Applies custom CSS for aesthetics (Rule: Visual Excellence)."""
    st.markdown(
        """
<style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    /* Header styling */
    h1, h2, h3 {
        color: #FFD700;
        font-family: 'Inter', sans-serif;
    }

    /* Metric boxes */
    div[data-testid="stMetricValue"] {
        color: #00FF7F;
        font-size: 2rem;
        font-weight: 700;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1c24;
    }

    /* Buttons */
    .stButton>button {
        background-color: #FFD700;
        color: #000;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FFC125;
        transform: scale(1.02);
    }
</style>
""",
        unsafe_allow_html=True,
    )
