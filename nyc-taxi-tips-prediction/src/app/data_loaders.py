"""
Data loading utilities for the NYC Taxi Tips Streamlit app.

This module contains functions to load metrics, predictions, and parameters
from the local artifacts directory, as well as checking the FastAPI backend health.
"""

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
import yaml

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Constants for paths
ARTIFACTS_DIR = Path("artifacts")
METRICS_PATH = ARTIFACTS_DIR / "model_evaluation" / "metrics.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions" / "inference_results.csv"
MODEL_DIR = ARTIFACTS_DIR / "model_trainer"
PARAMS_PATH = Path("config") / "params.yaml"


@st.cache_data
def load_metrics() -> dict[str, Any] | None:
    """
    Loads the evaluation metrics (JSON) from the local artifacts directory.

    Returns:
        Dictionary containing metrics if found, else None.
    """
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_predictions() -> pd.DataFrame | None:
    """
    Loads the simulated batch inference results (CSV) from the local artifacts directory.

    Returns:
        DataFrame containing predictions if found, else None.
    """
    if not PREDICTIONS_PATH.exists():
        return None
    return pd.read_csv(PREDICTIONS_PATH)


@st.cache_data(ttl=60)
def check_api_health() -> tuple[bool, str]:
    """Checks if the FastAPI backend is running.

    Returns:
        A tuple of (is_healthy, model_version).
    """
    try:
        response = requests.get(f"{API_URL}/v1/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return True, data.get("model_version", "unknown")
        return False, "unknown"
    except requests.exceptions.RequestException:
        return False, "unknown"


@st.cache_data
def load_params() -> dict[str, Any] | None:
    """
    Loads the parameters (YAML) to extract selection metrics.

    Returns:
        Dictionary containing parameters if found, else None.
    """
    if not PARAMS_PATH.exists():
        return None
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)
