"""
Data Ingestion Component for the NYC Taxi Tips Prediction pipeline.

This module contains the 'worker' logic for the Data Ingestion stage.
It handles reading the raw data from CSV files using Polars and joining
it with reference data (taxi zones) to produce an enriched dataset.
"""

import sys

import polars as pl

from src.entity.config_entity import DataIngestionConfig
from src.utils.exception import CustomExceptionError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataIngestion:
    """
    Handles the technical execution of data ingestion.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the DataIngestion component.

        Args:
            config (DataIngestionConfig): Configuration for data ingestion,
                including the data schema.
        """
        self.config = config

    def _get_polars_schema(self) -> dict[str, pl.DataType]:
        """
        Maps the dtypes defined in schema.yaml to Polars data types.

        Returns:
            dict[str, pl.DataType]: A dictionary mapping column names to Polars types.
        """
        dtype_mapping = {
            "Int64": pl.Int64,
            "Float64": pl.Float64,
            "Utf8": pl.Utf8,
            "Boolean": pl.Boolean,
        }

        polars_schema = {}
        # The schema is now under 'COLUMNS' in schema.yaml
        target_schema = self.config.all_schema.get("COLUMNS", {})

        for col_name, dtype_str in target_schema.items():
            if dtype_str in dtype_mapping:
                polars_schema[col_name] = dtype_mapping[dtype_str]
            else:
                logger.warning(
                    f"Unknown dtype '{dtype_str}' for column '{col_name}'. "
                    "Defaulting to pl.Utf8 (String)."
                )
                polars_schema[col_name] = pl.Utf8

        return polars_schema

    def _merge_taxi_zones(self, trips_df: pl.DataFrame, zones_df: pl.DataFrame) -> pl.DataFrame:
        """
        Merges taxi zone information into the trip DataFrame.

        Logic:
        - Joins with PU LocationID -> PU_Borough, PU_Zone
        - Joins with DO LocationID -> DO_Borough, DO_Zone

        Args:
            trips_df (pl.DataFrame): Trip data with 'PULocationID' and 'DOLocationID'.
            zones_df (pl.DataFrame): Zones data with 'LocationID', 'Borough', 'Zone'.

        Returns:
            pl.DataFrame: Enriched DataFrame.
        """
        logger.info("Enriching with Pickup Location metadata...")
        trips_df = trips_df.join(
            zones_df,
            left_on="PULocationID",
            right_on="LocationID",
            how="left",
        ).rename({"Borough": "PU_Borough", "Zone": "PU_Zone"})

        logger.info("Enriching with Dropoff Location metadata...")
        trips_df = trips_df.join(
            zones_df,
            left_on="DOLocationID",
            right_on="LocationID",
            how="left",
        ).rename({"Borough": "DO_Borough", "Zone": "DO_Zone"})

        return trips_df

    def initiate_data_ingestion(self) -> None:
        """
        Executes the data ingestion process:
        1. Reads the distilled trip data with specified schema.
        2. Reads the taxi zones data.
        3. Enriches trip data by joining with zones for pickup (PU) and dropoff (DO).
        4. Saves the enriched dataset as a Parquet file.

        Raises:
            CustomExceptionError: If any error occurs during ingestion.
        """
        logger.info("Starting Data Ingestion execution...")

        try:
            # Check if source file exists
            if not self.config.source_data_path.exists():
                raise FileNotFoundError(
                    f"Source data file not found at: {self.config.source_data_path}"
                )

            # 1. Read Trip Data
            logger.info(f"Reading trip data from: {self.config.source_data_path}")

            # schema_overrides dict for read_csv
            schema_overrides = self._get_polars_schema()

            # Polars CSV reading
            trips_df = pl.read_csv(
                self.config.source_data_path,
                separator=",",
                has_header=True,
                schema_overrides=schema_overrides,
                try_parse_dates=True,
                ignore_errors=True,
            )

            logger.info(f"Loaded Trip Data with shape: {trips_df.shape}")

            # 2. Read Taxi Zones
            logger.info(f"Reading taxi zones from: {self.config.taxi_zones_path}")
            if not self.config.taxi_zones_path.exists():
                raise FileNotFoundError(
                    f"Taxi zones file not found at: {self.config.taxi_zones_path}"
                )

            zones_df = pl.read_csv(self.config.taxi_zones_path)

            # Prepare Zones DataFrame for joining
            zones_lookup = zones_df.select(
                [
                    pl.col("Location ID").cast(pl.Int64).alias("LocationID"),
                    pl.col("Borough"),
                    pl.col("Zone"),
                ]
            ).unique(subset=["LocationID"], keep="first")

            # 3. Join logic
            trips_df = self._merge_taxi_zones(trips_df, zones_lookup)

            # 4. Save as Parquet
            output_file_path = self.config.output_data_path
            logger.info(f"Saving enriched data to: {output_file_path}")

            if not output_file_path.parent.exists():
                output_file_path.parent.mkdir(parents=True, exist_ok=True)

            trips_df.write_parquet(output_file_path)

            logger.info(f"Data Ingestion completed successfully. Shape: {trips_df.shape}")

        except Exception as e:
            raise CustomExceptionError(e, sys) from e
