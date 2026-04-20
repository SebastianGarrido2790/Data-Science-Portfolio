"""
Module for testing and validating the accuracy of the Tone Classifier.
Provides functionality for manual verification (Vibe Check) and statistical visualization
of the global tone distribution across the dataset.

Usage Instructions:
    1. Run all tests:
       uv run pytest src/tests

    2. Run tests with verbose output (recommended):
       uv run pytest src/tests -vv

    3. Run a specific test module:
       uv run pytest src/tests/test_tone_accuracy.py

    4. Run specific tests by keyword matching:
       uv run pytest -k "tone"
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import seaborn as sns

from src.config.configuration import ConfigurationManager
from src.utils.exception import CustomException
from src.utils.logger import get_logger

STAGE_NAME = "Tone Accuracy Test"
logger = get_logger(headline=STAGE_NAME)


@pytest.mark.integration
def test_tone_accuracy() -> None:
    """
    Evaluates the accuracy of the Tone Classifier by performing a 'Vibe Check'
    on a random sample of enriched books and generating a distribution plot.

    Flow:
    1. Loads toned data.
    2. Prints a random sample of book titles, tones, and descriptions for manual verification.
    3. Generates and saves a bar chart of the global tone distribution.

    Raises:
        CustomException: If test fails.
    """
    try:
        config_manager = ConfigurationManager()
        data_path = config_manager.get_tone_analysis_config().output_path

        # Verify file exists
        if not os.path.exists(data_path):
            logger.error(f"Toned data not found at {data_path}. Run tone analysis first.")
            return

        df = pd.read_csv(data_path)
        schema = config_manager.get_schema_config()
        cols = schema.columns
        enriched_cols = schema.enriched_columns

        logger.info(f"Loaded {len(df)} books with tone analysis.")

        # Sample random books for manual "Vibe Check"
        sample_size = min(len(df), 20)
        sample = df.sample(n=sample_size)

        logger.info("\n" + "=" * 80)
        logger.info("TONE ANALYSIS VIBE CHECK (Random Sample)")
        logger.info("=" * 80)
        logger.info(f"{'TITLE':<40} | {'TONE':<10} | {'DESCRIPTION SNIPPET'}")
        logger.info("-" * 80)

        title_col = cols["title"]
        tone_col = enriched_cols["dominant_tone"]
        desc_col = cols["description"]

        for _, row in sample.iterrows():
            title = row[title_col][:37] + "..." if len(str(row[title_col])) > 37 else row[title_col]
            tone = str(row[tone_col]).upper()
            desc = str(row[desc_col])[:60].replace("\n", " ") + "..."

            logger.info(f"{title:<40} | {tone:<10} | {desc}")

        logger.info("=" * 80)
        logger.info("Tone 'Vibe Check' completed.")

        # 2. Global Tone Distribution Stats
        logger.info("\nGlobal Tone Distribution:")
        counts = df[tone_col].value_counts()
        logger.info(f"\n{counts}")

        # 3. Visualization
        logger.info("Generating tone distribution plot...")
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Define a consistent color palette for emotions
        emotion_colors = {
            "joy": "#FFD700",  # Gold
            "sadness": "#4682B4",  # SteelBlue
            "fear": "#4B0082",  # Indigo
            "anger": "#B22222",  # FireBrick
            "surprise": "#FF8C00",  # DarkOrange
            "neutral": "#A9A9A9",  # DarkGray
            "disgust": "#556B2F",  # DarkOliveGreen
        }

        sns.barplot(
            x=counts.index,
            y=counts.values,
            hue=counts.index,
            palette=emotion_colors,
            legend=False,
        )
        plt.title("Distribution of Emotional Tones in Book Collection", fontsize=15)
        plt.xlabel("Emotional Tone", fontsize=12)
        plt.ylabel("Number of Books", fontsize=12)

        # Save plot
        figures_dir = "reports/figures"
        os.makedirs(figures_dir, exist_ok=True)
        plot_path = os.path.join(figures_dir, "tone_distribution.png")
        plt.savefig(plot_path)
        logger.info(f"Tone distribution plot saved to {plot_path}")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        test_tone_accuracy()
    except Exception as e:
        raise CustomException(e, sys)
