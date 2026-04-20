"""
Data loading and engine initialization helpers for the Gradio UI.
"""

import os
import sys

from src.config.configuration import ConfigurationManager
from src.models.hybrid_recommender import HybridRecommender
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- CONSTANTS ---
PLACEHOLDER_IMG_ABS = os.path.abspath("reports/figures/placeholder.png")
PLACEHOLDER_IMG_URL = PLACEHOLDER_IMG_ABS


def init_recommender() -> HybridRecommender | None:
    """
    Initializes the HybridRecommender engine.

    Returns:
        HybridRecommender | None: The initialized recommender engine or None.
    """
    try:
        config = ConfigurationManager()
        inference_config = config.get_inference_config()
        schema_config = config.get_schema_config()
        recommender = HybridRecommender(config=inference_config, schema=schema_config)
        return recommender
    except Exception as e:
        logger.error(f"Failed to initialize recommender: {CustomException(e, sys)}")
        return None


def get_app_config():
    """
    Loads application parameters from configuration.

    Returns:
        tuple: (CATEGORIES, TONES, TONE_MAP, MAX_RESULTS)
    """
    try:
        config_manager = ConfigurationManager()
        APP_PARAMS = config_manager.params.app
        categories = list(APP_PARAMS.categories)
        tones = list(APP_PARAMS.tones)
        tone_map = dict(APP_PARAMS.tone_map)
        max_results = int(APP_PARAMS.max_results)
        return categories, tones, tone_map, max_results
    except Exception as e:
        logger.error(f"Failed to load app parameters: {e}")
        # Fallback values
        categories = [
            "All",
            "Biography",
            "Fantasy",
            "Fiction",
            "History",
            "Non-Fiction",
            "Science",
            "Thriller",
        ]
        tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
        tone_map = {
            "Happy": "joy",
            "Surprising": "surprise",
            "Angry": "anger",
            "Suspenseful": "fear",
            "Sad": "sadness",
        }
        max_results = 48
        return categories, tones, tone_map, max_results


def get_high_res_image(url: str | None) -> str:
    """
    Cleans and enhances image URLs for high resolution.
    """
    if not isinstance(url, str) or not url or url.lower() == "nan":
        return PLACEHOLDER_IMG_URL

    url = url.strip().strip("'").strip('"')
    if url.startswith("http://"):
        url = url.replace("http://", "https://")
    return f"{url}&fife=w800" if "?" in url else f"{url}?fife=w800"


def format_authors(authors_str: str) -> str:
    """
    Formats the authors string for clean display.
    """
    if not isinstance(authors_str, str) or not authors_str:
        return "Unknown Author"

    clean_str = (
        authors_str.replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
        .replace(";", ",")
    )
    authors = [a.strip() for a in clean_str.split(",")]

    if len(authors) == 0:
        return "Unknown Author"
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    return f"{authors[0]} et al."
