"""
This module serves as the 'Worker' for the Data Transformation Stage of the pipeline.
It handles the train-test-validation split for model development and evaluation.
"""

import os
import sys
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataTransformationConfig
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataTransformation:
    """
    This class implements the 'Transformation' stage of the CRISP-DM lifecycle,
    focusing on deterministic data partitioning (Train/Val/Test) for MLOps tracking.

    Attributes:
        config (DataTransformationConfig): Configuration entity containing data paths,
            model name, and batch size for the enrichment process.
    """

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def initiate_data_transformation(self) -> None:
        """
        Orchestrates the full transformation pipeline.

        This method performs the following steps:
        1. Reads the cleaned book dataset from the specified path.
        2. Performs a 3-way split (Train/Val/Test) using the specified test_size and val_size.
        3. Saves the resulting artifacts (train.csv, val.csv, test.csv) to the root directory.

        Raises:
            CustomException: If any error occurs during the transformation process.
        """
        try:
            # NOTE: We read the CLEAN data, not the raw data
            data_path = self.config.data_path
            df = pd.read_csv(data_path)
            logger.info(f"Loaded CLEAN data from {data_path} with shape {df.shape}")

            # Perform 3-Way Split
            # Using Any to satisfy strict type checkers about split result being a list/tuple
            split_1: Any = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )
            train_df, temp_df = split_1

            split_2: Any = train_test_split(
                temp_df,
                test_size=self.config.val_size,
                random_state=self.config.random_state,
            )
            val_df, test_df = split_2

            logger.info(
                f"Split Ratios - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
            )

            # Explicitly use pandas to_csv
            train_df.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
            val_df.to_csv(os.path.join(self.config.root_dir, "val.csv"), index=False)
            test_df.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

            logger.info(f"Transformation Completed. Artifacts saved to {self.config.root_dir}")

        except Exception as e:
            raise CustomException(e, sys)
