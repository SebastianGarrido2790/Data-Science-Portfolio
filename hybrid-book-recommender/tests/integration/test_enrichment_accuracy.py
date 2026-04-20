"""
Module for evaluating the accuracy of the Zero-Shot Data Enrichment process.

This module performs a quantitative assessment of the enrichment pipeline by:
1. Mapping original dataset categories to broad target facets (Ground Truth).
2. Calculating classification metrics including Accuracy and F1-Score.
3. Generating and saving confusion matrix heatmaps for error analysis.

Usage Instructions:
    1. Run all tests:
       uv run pytest src/tests

    2. Run tests with verbose output (recommended):
       uv run pytest src/tests -vv

    3. Run a specific test module:
       uv run pytest src/tests/test_enrichment_accuracy.py

    4. Run specific tests by keyword matching:
       uv run pytest -k "enrichment"
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

STAGE_NAME = "Enrichment Accuracy Test"
logger = get_logger(headline=STAGE_NAME)


@pytest.mark.integration
def test_enrichment_accuracy() -> None:
    """
    Evaluates the accuracy of the Zero-Shot Enricher by comparing predictions
    against a manually mapped 'Ground Truth' from the original dataset.

    Flow:
    1. Checks if enriched data exists.
    2. Maps original categories to 'Ground Truth' broad facets.
    3. Calculates classification metrics (Accuracy, F1-Score).
    4. Generates and saves a confusion matrix heatmap.

    Raises:
        CustomException: If evaluation fails.
    """
    try:
        config_manager = ConfigurationManager()
        data_path = config_manager.get_data_enrichment_config().enriched_data_path
        if not os.path.exists(data_path):
            logger.error(f"Enriched data not found at {data_path}. Please run enrichment first.")
            return

        logger.info("Loading enriched dataset...")
        df = pd.read_csv(data_path)
        schema = config_manager.get_schema_config()
        cols = schema.columns
        enriched_cols = schema.enriched_columns

        # 1. Define Ground Truth Mapping
        # We map high-frequency original categories to our broad target facets.
        mapping = {
            "Fiction": "Fiction",
            "Juvenile Fiction": "Fiction",
            "Literary Criticism": "Fiction",
            "Biography & Autobiography": "Biography",
            "History": "History",
            "Science": "Science",
            "Juvenile Nonfiction": "Non-Fiction",
            "Social Science": "Non-Fiction",
            "Religion": "Non-Fiction",
            "Philosophy": "Non-Fiction",
            "Business & Economics": "Non-Fiction",
            "Psychology": "Non-Fiction",
            "Computers": "Science",  # or Technology if we had it, Science is closest
        }

        # 2. Filter for Labeled Data
        # We only keep rows where the original 'categories' is in our mapping keys.
        cat_col = cols["categories"]
        simple_cat_col = enriched_cols["simple_category"]

        df_test = df[df[cat_col].isin(mapping.keys())].copy()

        # Create Ground Truth column
        df_test["ground_truth"] = df_test[cat_col].map(mapping)

        # 3. Handle specific overlap cases
        # Note: Zero-shot might classify 'Literary Criticism' as 'Non-Fiction' (which is technically true),
        # but our mapping expects 'Fiction' if it's about fiction. This reflects the limitation
        # mentioned by the user.

        logger.info(f"Evaluating accuracy on {len(df_test)} high-confidence labeled books.")

        # 4. Generate Metrics
        y_true = df_test["ground_truth"]
        y_pred = df_test[simple_cat_col]

        # Filter out prediction labels that aren't in our ground truth sub-analysis if needed
        # but here we want to see how the model performed overall.

        unique_labels = sorted(set(y_true) | set(y_pred))

        report = classification_report(y_true, y_pred, zero_division=0)

        logger.info("\n" + "=" * 60)
        logger.info("ZERO-SHOT CLASSIFICATION ACCURACY REPORT")
        logger.info("=" * 60)
        logger.info("Comparing: 'categories' (Original) vs 'simple_category' (Zero-Shot)")
        logger.info(f"Sample Size: {len(df_test)} books")
        logger.info("-" * 60)
        logger.info(f"\n{report}")
        logger.info("=" * 60)

        # 5. Output Confusion Matrix
        unique_labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=unique_labels,
            yticklabels=unique_labels,
            cmap="Blues",
        )
        plt.xlabel("Predicted (Zero-Shot)")
        plt.ylabel("True (Original Grouped)")
        plt.title("Enrichment Accuracy: Granular Confusion Matrix")

        figures_dir = "reports/figures"
        os.makedirs(figures_dir, exist_ok=True)
        plot_path = os.path.join(figures_dir, "enrichment_granular_cm.png")
        plt.savefig(plot_path)
        logger.info(f"Granular confusion matrix saved to {plot_path}")

        logger.info("Accuracy test completed.")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        test_enrichment_accuracy()
    except Exception as e:
        raise CustomException(e, sys)
