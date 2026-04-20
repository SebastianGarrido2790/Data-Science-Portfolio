"""
Module for evaluating the broad classification accuracy of the book categorization model.
This test script compares predicted categories against ground truth by mapping them
into high-level binary classes (Fiction vs. Non-Fiction) and generating performance metrics.

Usage Instructions:
    1. Run all tests:
       uv run pytest src/tests

    2. Run tests with verbose output (recommended):
       uv run pytest src/tests -vv

    3. Run a specific test module:
       uv run pytest src/tests/test_broad_accuracy.py

    4. Run specific tests by keyword matching:
       uv run pytest -k "broad"
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.config.configuration import ConfigurationManager
from src.utils.exception import CustomException
from src.utils.logger import get_logger

STAGE_NAME = "Broad Accuracy Test"
logger = get_logger(headline=STAGE_NAME)


@pytest.mark.integration
def test_broad_accuracy() -> None:
    """
    Evaluates accuracy using broad masks (Fiction vs Non-Fiction).

    Flow:
    1. Loads enriched data.
    2. Maps Ground Truth and Predictions to Binary Classes (Fiction/Non-Fiction).
    3. Generates Classification Report.
    4. Plots and saves Confusion Matrix.

    Raises:
        CustomException: If evaluation fails.
    """
    try:
        config_manager = ConfigurationManager()
        data_path = config_manager.get_data_enrichment_config().enriched_data_path
        if not os.path.exists(data_path):
            logger.warning(f"Skipping test_broad_accuracy: {data_path} not found.")
            return

        df = pd.read_csv(data_path)
        schema = config_manager.get_schema_config()
        cols = schema.columns
        enriched_cols = schema.enriched_columns

        # 1. Map Ground Truth to Broad classes
        # Categories starting with 'Juvenile Fiction' or just 'Fiction'
        def get_broad_gt(cat: str) -> str:
            cat = str(cat)
            if "Fiction" in cat or "Novel" in cat or "Literary Criticism" in cat:
                return "Broad_Fiction"
            return "Broad_NonFiction"

        # 2. Map Predictions to Broad classes
        def get_broad_pred(label: str) -> str:
            fiction_labels = ["Fiction", "Fantasy", "Thriller"]
            if label in fiction_labels:
                return "Broad_Fiction"
            return "Broad_NonFiction"

        # Filter for rows that have SOME category label - using schema mapping
        cat_col = cols["categories"]
        simple_cat_col = enriched_cols["simple_category"]

        df = df.dropna(subset=[cat_col])

        y_true = df[cat_col].apply(get_broad_gt)
        y_pred = df[simple_cat_col].apply(get_broad_pred)

        report = classification_report(y_true, y_pred)

        logger.info("\n" + "=" * 60)
        logger.info("BROAD (FICTION VS NON-FICTION) ACCURACY REPORT")
        logger.info("=" * 60)
        logger.info(f"\n{report}")
        logger.info("=" * 60)

        # 3. Output Confusion Matrix
        labels = ["Broad_Fiction", "Broad_NonFiction"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=labels,
            yticklabels=labels,
            cmap="Greens",
        )
        plt.xlabel("Predicted (Zero-Shot Grouped)")
        plt.ylabel("True (Original Grouped)")
        plt.title("Enrichment Accuracy: Broad Confusion Matrix")

        figures_dir = "reports/figures"
        os.makedirs(figures_dir, exist_ok=True)
        plot_path = os.path.join(figures_dir, "enrichment_broad_cm.png")
        plt.savefig(plot_path)
        logger.info(f"Broad confusion matrix saved to {plot_path}")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        test_broad_accuracy()
    except Exception as e:
        raise CustomException(e, sys)
