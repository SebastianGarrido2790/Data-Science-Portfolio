"""
Feature Engineering Component.

This module handles the creation of new features (cyclical time encodings) and
the final splitting of the dataset into Train, Validation, and Test sets.
"""

import sys

import polars as pl

from src.entity.config_entity import FeatureEngineeringConfig
from src.utils.exception import CustomExceptionError
from src.utils.feature_utils import encode_cyclical
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineering:
    """
    Handles feature engineering and temporal splitting.
    """

    def __init__(self, config: FeatureEngineeringConfig):
        """
        Initializes the FeatureEngineering component.

        Args:
            config (FeatureEngineeringConfig): Configuration for feature engineering.
        """
        self.config = config

    def _create_cyclical_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Creates cyclical features for hour, day, and month.

        Uses Sine/Cosine transformations to preserve temporal proximity
        (e.g., Hour 23 is close to Hour 0).

        Args:
            df (pl.DataFrame): Input DataFrame with 'tpep_pickup_datetime'.

        Returns:
            pl.DataFrame: DataFrame with added cyclical features.
        """
        logger.info("Creating cyclical time features...")

        # 1. Extract base time features
        df = df.with_columns(
            pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
            pl.col("tpep_pickup_datetime").dt.weekday().alias("pickup_day"),  # Monday=1, Sunday=7
            pl.col("tpep_pickup_datetime").dt.month().alias("pickup_month"),
            pl.col("tpep_pickup_datetime").dt.ordinal_day().alias("pickup_day_of_year"),
        )

        # 2. Cyclical Encoding - uses shared encode_cyclical() to guarantee
        # training-serving parity with predict_api.py
        # Hour (0-23, period=24)
        df = df.with_columns(
            pl.Series(
                "pickup_hour_sin",
                [encode_cyclical(h, 24)[0] for h in df["pickup_hour"].to_list()],
            ),
            pl.Series(
                "pickup_hour_cos",
                [encode_cyclical(h, 24)[1] for h in df["pickup_hour"].to_list()],
            ),
        )

        # Day of Week (1-7 -> shift to 0-6, period=7)
        df = df.with_columns(
            pl.Series(
                "pickup_day_sin",
                [encode_cyclical(d - 1, 7)[0] for d in df["pickup_day"].to_list()],
            ),
            pl.Series(
                "pickup_day_cos",
                [encode_cyclical(d - 1, 7)[1] for d in df["pickup_day"].to_list()],
            ),
        )

        # Month (1-12 -> shift to 0-11, period=12)
        df = df.with_columns(
            pl.Series(
                "pickup_month_sin",
                [encode_cyclical(m - 1, 12)[0] for m in df["pickup_month"].to_list()],
            ),
            pl.Series(
                "pickup_month_cos",
                [encode_cyclical(m - 1, 12)[1] for m in df["pickup_month"].to_list()],
            ),
        )

        logger.info("Cyclical features created successfully.")
        return df

    def _split_data(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Splits data into Train, Validation, and Test sets based on months.

        Args:
            df (pl.DataFrame): Input DataFrame with 'pickup_month'.

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        logger.info(
            f"Splitting data (Train: {self.config.train_months_start}-"
            f"{self.config.train_months_end}, Val: {self.config.val_months_start}-"
            f"{self.config.val_months_end}, Test: {self.config.test_months_start}-"
            f"{self.config.test_months_end})..."
        )

        train_df = df.filter(
            pl.col("pickup_month").is_between(
                self.config.train_months_start, self.config.train_months_end
            )
        )
        val_df = df.filter(
            pl.col("pickup_month").is_between(
                self.config.val_months_start, self.config.val_months_end
            )
        )
        test_df = df.filter(
            pl.col("pickup_month").is_between(
                self.config.test_months_start, self.config.test_months_end
            )
        )

        logger.info(f"Train Set: {train_df.shape}")
        logger.info(f"Val Set:   {val_df.shape}")
        logger.info(f"Test Set:  {test_df.shape}")

        if train_df.is_empty() or val_df.is_empty() or test_df.is_empty():
            logger.warning("One of the splits is empty! Check date ranges in source data.")

        return train_df, val_df, test_df

    def initiate_feature_engineering(self) -> None:
        """
        Executes the feature engineering pipeline:
        1. Loads cleaned data.
        2. Creates cyclical features.
        3. Splits data (Train: Jan-Aug, Val: Sept-Oct, Test: Nov-Dec).
        4. Saves artifacts.

        Raises:
            CustomExceptionError: If execution fails.
        """
        try:
            logger.info("Loading cleaned data for feature engineering...")
            if not self.config.data_path.exists():
                raise FileNotFoundError(f"Data file not found at: {self.config.data_path}")

            df = pl.read_parquet(self.config.data_path)
            logger.info(f"Loaded data shape: {df.shape}")

            # 1. Feature Engineering
            df = self._create_cyclical_features(df)

            # 2. Temporal Splitting
            train_df, val_df, test_df = self._split_data(df)

            # 3. Save Artifacts
            train_path = self.config.root_dir / "train.parquet"
            val_path = self.config.root_dir / "val.parquet"
            test_path = self.config.root_dir / "test.parquet"

            logger.info(f"Saving train set to {train_path}")
            train_df.write_parquet(train_path)

            logger.info(f"Saving val set to {val_path}")
            val_df.write_parquet(val_path)

            logger.info(f"Saving test set to {test_path}")
            test_df.write_parquet(test_path)

            logger.info("Feature Engineering completed successfully.")

        except Exception as e:
            raise CustomExceptionError(e, sys) from e
