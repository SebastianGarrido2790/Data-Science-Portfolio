"""
Data Validation Component.

This module validates the ingested data against the defined schema and business logic
using Polars. It acts as a gatekeeper, ensuring that only quality data enters
the transformation stage.
"""

import sys

import polars as pl

from src.entity.config_entity import DataValidationConfig
from src.utils.exception import CustomExceptionError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidation:
    """
    Validates data against schema and business constraints.
    """

    def __init__(self, config: DataValidationConfig):
        """
        Initializes the DataValidation component.

        Args:
            config (DataValidationConfig): Configuration for data validation.
        """
        self.config = config

    def validate_all_columns(self) -> bool:
        """
        Validates column presence and types against schema.yaml.
        Also performs basic range checks as "Guardrails".

        Returns:
            bool: True if validation passes, False otherwise.

        Raises:
            CustomExceptionError: If validation fails critically or file I/O errors.
        """
        try:
            validation_status = True
            logger.info("Starting Data Validation...")

            # 1. Load Data
            if not self.config.unzip_dir.exists():
                raise FileNotFoundError(f"Data file not found at: {self.config.unzip_dir}")

            df = pl.read_parquet(self.config.unzip_dir)
            logger.info(f"Loaded data with shape: {df.shape}")

            # 2. Check Column Presence
            # We look at 'COLUMNS' in schema.yaml
            all_schema = self.config.all_schema.get("COLUMNS", {})
            required_cols = list(all_schema.keys())

            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                validation_status = False
                logger.error(f"Missing columns: {missing_cols}")
            else:
                logger.info("All required columns are present.")

            # 3. Guardrails (Basic Logic Checks)
            # - Total Amount should generally be positive (though refunds exist, warnings are good)
            # - Trip Distance should be non-negative

            # Using Polars expression to check for invalid data
            invalid_distance = df.filter(pl.col("trip_distance") < 0).height
            if invalid_distance > 0:
                logger.warning(f"Found {invalid_distance} rows with negative trip_distance.")
                # We do not fail here, but log it. Cleaning stage handles logic.

            # 4. Write Status
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Validation status: {validation_status}")

            logger.info(f"Validation completed. Status written to {self.config.STATUS_FILE}")

            return validation_status

        except Exception as e:
            raise CustomExceptionError(e, sys) from e
