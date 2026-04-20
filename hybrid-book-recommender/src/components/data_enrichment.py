"""
OFFLINE SCRIPT: Data Enrichment via Zero-Shot Classification.

This script reads the cleaned book dataset and uses a Large Language Model (BART)
to classify each book into a simplified 'Broad Category' based on its description.

WARNING: This process is CPU/GPU intensive.
On a standard CPU, expect approx 0.5 - 1.0 seconds per book.
"""

import sys
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

from src.entity.config_entity import DataEnrichmentConfig, SchemaConfig
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataEnrichment:
    """
    Component for enriching book metadata using Zero-Shot Classification.
    Standardizes messy categories into a few broad, usable facets.

    Attributes:
        config (DataEnrichmentConfig): Configuration entity containing data paths,
            model name, and batch size for the enrichment process.
        schema (SchemaConfig): Data contract mapping logical -> physical column names.
    """

    def __init__(self, config: DataEnrichmentConfig, schema: SchemaConfig):
        self.config = config
        self.schema = schema

    def initiate_data_enrichment(self) -> None:
        """
        Executes the enrichment process.

        This method performs the following steps:
        1. Reads the cleaned book dataset from the specified path.
        2. Initializes the Zero-Shot Classification model.
        3. Enriches the dataset with broad categories based on book descriptions.
        4. Saves the enriched dataset to a new CSV file.

        Raises:
            CustomException: If any error occurs during the enrichment process.
        """
        try:
            logger.info(f"Loading data from {self.config.data_path}")
            df = pd.read_csv(self.config.data_path)

            cols = self.schema.columns
            enriched_cols = self.schema.enriched_columns

            # Fill NaNs in descriptions with empty string using schema mapping
            descriptions = df[cols["description"]].fillna("").tolist()

            logger.info(f"Initializing Zero-Shot Classification model: {self.config.model_name}")
            # Device selection: use GPU if available
            device = 0 if torch.cuda.is_available() else -1
            if device == 0:
                logger.info("GPU detected, using GPU for classification.")
            else:
                logger.info("No GPU detected, using CPU. This might be slow.")

            classifier = pipeline(
                "zero-shot-classification", model=self.config.model_name, device=device
            )

            candidate_labels = self.config.candidate_labels
            batch_size = self.config.batch_size

            logger.info(f"Enriching {len(df)} books in batches of {batch_size}...")

            new_categories = []

            for i in tqdm(range(0, len(descriptions), batch_size), desc="Classifying Descriptions"):
                batch = descriptions[i : i + batch_size]

                # Zero-shot classification
                results: Any = classifier(batch, candidate_labels, multi_label=False)

                # Extract top labels
                if results:
                    # results can be a single dict (if batch_size=1) or a list of dicts.
                    if isinstance(results, dict):
                        results = [results]

                    for res in results:
                        if res and "labels" in res:
                            new_categories.append(str(res["labels"][0]))

            # Use schema mapping for enriched column name
            df[enriched_cols["simple_category"]] = new_categories

            logger.info(f"Saving enriched data to {self.config.enriched_data_path}")
            df.to_csv(self.config.enriched_data_path, index=False)

            logger.info("✅ Data enrichment completed successfully.")

        except Exception as e:
            raise CustomException(e, sys)
