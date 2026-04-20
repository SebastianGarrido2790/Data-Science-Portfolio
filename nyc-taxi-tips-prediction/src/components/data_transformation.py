"""
Data Transformation Component.

This module handles the cleaning, imputation, and temporal splitting of the dataset.
It implements the FTI (Feature-Training-Inference) architecture by ensuring
the data is devoid of outliers and missing values before feature engineering.
"""

import sys

import polars as pl

from src.entity.config_entity import DataTransformationConfig
from src.utils.exception import CustomExceptionError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataTransformation:
    """
    Transforms data: Cleaning, Imputation, Dropping, and Temporal Splitting.
    """

    def __init__(self, config: DataTransformationConfig):
        """
        Initializes the DataTransformation component.

        Args:
            config (DataTransformationConfig): Configuration for data transformation.
        """
        self.config = config

    def _clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Applies cleaning, imputation, and filtering rules.

        Rules:
        - Imputation:
          - airport_fee, congestion_surcharge -> 0 (if null)
          - passenger_count -> 1 (if null or 0)
          - RatecodeID -> 99 (if null)
        - Dropping:
          - store_and_fwd_flag (irrelevant)
        - Filtering:
          - Negative Values: Drop row if ANY financial column < 0
          - Trip Distance: 0.5 < x < 100 miles
          - Total Amount: $3.70 <= x <= $1000

        Args:
            df (pl.DataFrame): Input DataFrame.

        Returns:
            pl.DataFrame: Cleaned DataFrame.
        """
        logger.info("Starting Data Cleaning (Imputation & Filtering)...")
        initial_rows = df.height

        # 1. Dropping Columns
        if "store_and_fwd_flag" in df.columns:
            df = df.drop("store_and_fwd_flag")
            logger.info("Dropped 'store_and_fwd_flag' column.")

        # 2. Imputation
        df = df.with_columns(
            pl.col("airport_fee").fill_null(0.0),
            pl.col("congestion_surcharge").fill_null(0.0),
            pl.col("passenger_count").fill_null(1).replace(0, 1),  # Replace 0 with 1 as well
            pl.col("RatecodeID").fill_null(99),
        )
        logger.info("Applied basic imputation rules.")

        # 3. Filtering
        # Financial columns to check for negatives
        financial_cols = [
            "fare_amount",
            "extra",
            "mta_tax",
            "tip_amount",
            "tolls_amount",
            "improvement_surcharge",
            "total_amount",
            "congestion_surcharge",
            "airport_fee",
        ]

        # Ensure we only check columns that exist in the dataframe
        existing_financial_cols = [c for c in financial_cols if c in df.columns]

        # Construct filter mask
        # 1. No negative values in ANY financial column
        non_negative_mask = pl.all_horizontal([pl.col(c) >= 0 for c in existing_financial_cols])

        filter_mask = (
            non_negative_mask
            & (pl.col("trip_distance") > self.config.min_trip_distance)
            & (pl.col("trip_distance") < self.config.max_trip_distance)
            & (pl.col("total_amount") >= self.config.min_total_amount)
            & (pl.col("total_amount") <= self.config.max_total_amount)
        )

        df_cleaned = df.filter(filter_mask)
        final_rows = df_cleaned.height

        dropped_rows = initial_rows - final_rows
        percentage_dropped = (dropped_rows / initial_rows) * 100

        logger.info(
            f"Filtering complete. Dropped {dropped_rows} rows ({percentage_dropped:.2f}%). "
            f"Final shape: {df_cleaned.shape}"
        )

        return df_cleaned

    def initiate_data_transformation(self) -> None:
        """
        Executes the data transformation pipeline:
        1. Loads enriched data.
        2. Cleans and filters data.
        3. Parses datetime columns.
        4. Saves cleaned data as Parquet.

        Raises:
            CustomExceptionError: If transformation fails.
        """
        try:
            logger.info("Loading data for transformation...")
            if not self.config.data_path.exists():
                raise FileNotFoundError(f"Data file not found at: {self.config.data_path}")

            df = pl.read_parquet(self.config.data_path)
            logger.info(f"Loaded raw data: {df.shape}")

            # 1. Clean Data
            df = self._clean_data(df)

            # 2. Robust Date Parsing
            # Ensure datetime type using precise format (e.g., "08/16/2023 05:24:41 PM")
            df = df.with_columns(
                pl.col("tpep_pickup_datetime").str.strptime(
                    pl.Datetime, "%m/%d/%Y %I:%M:%S %p", strict=False
                )
            )

            # Drop rows where datetime parsing failed (NaTs)
            initial_rows = df.height
            df = df.drop_nulls(subset=["tpep_pickup_datetime"])
            final_rows = df.height
            if initial_rows != final_rows:
                logger.warning(
                    f"Dropped {initial_rows - final_rows} rows due to datetime parsing failures."
                )

            # 3. Save Artifact
            output_path = self.config.root_dir / "cleaned_trip_data.parquet"
            logger.info(f"Saving cleaned data to {output_path}")
            df.write_parquet(output_path)

            logger.info("Data Transformation completed successfully.")

        except Exception as e:
            raise CustomExceptionError(e, sys) from e
