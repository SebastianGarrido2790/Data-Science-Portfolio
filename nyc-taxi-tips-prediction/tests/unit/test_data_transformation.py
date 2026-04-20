"""
Data Transformation Unit Tests.

This module tests the logic in `src/components/data_transformation.py`.
Scope:
1. Imputation (Handling nulls for airport_fee, congestion_surcharge, passenger_count).
2. Filtering (Removing negative values, impossible trip distances).
3. Column Dropping (Removing irrelevant features).
"""

from pathlib import Path

import polars as pl

from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataTransformationConfig

# Dummy configuration
config = DataTransformationConfig(
    root_dir=Path("./temp"),
    data_path=Path("./dummy_ingested.parquet"),
)


def test_initial_imputation(sample_trip_data):
    """Verifies null imputation rules."""
    dt = DataTransformation(config)
    result = dt._clean_data(sample_trip_data)

    # Expected retained rows:
    # Row 0: Valid.
    # Row 1: Dropped (Distance 0.0 < 0.5).
    # Row 2: Dropped (Negative Amount).
    # Row 3: Dropped (Amount > 1000).
    # Row 4: Valid (Distance 2.0).

    # Check airport_fee
    # Row 0 (1.25) + Row 4 (0.0 imputed) = 1.25
    assert result["airport_fee"].null_count() == 0, "Null airport_fee should be filled."
    assert result["airport_fee"].sum() == 1.25, "Airport fee sum mismatch."

    # Check congestion_surcharge
    # Row 0 (2.5) + Row 4 (0.0 imputed) = 2.5
    assert result["congestion_surcharge"].null_count() == 0, (
        "Null congestion_surcharge should be filled."
    )
    assert result["congestion_surcharge"].sum() == 2.5, "Congestion surcharge sum mismatch."

    # Check passenger_count
    # Row 0 (1) + Row 4 (0 -> 1) = 2. Mean = 1.0
    assert result["passenger_count"].null_count() == 0, "Null passenger_count should be filled."
    assert result["passenger_count"].mean() == 1.0, "Passenger count imputation mismatch."

    # Check RatecodeID
    assert result["RatecodeID"].null_count() == 0, "Null RatecodeID should be filled."
    assert result["RatecodeID"].max() == 99, (
        "Null RatecodeID should be 99."
    )  # Assuming 99 is max code in dummy data


def test_column_dropping(sample_trip_data):
    """Verifies store_and_fwd_flag is dropped."""
    dt = DataTransformation(config)
    result = dt._clean_data(sample_trip_data)

    assert "store_and_fwd_flag" not in result.columns, "store_and_fwd_flag should be dropped."


def test_filtering_refunds(sample_trip_data):
    """Verifies negative total_amount is filtered."""
    dt = DataTransformation(config)
    result = dt._clean_data(sample_trip_data)

    # The row with -5.0 should be gone
    assert result.filter(pl.col("total_amount") < 0).shape[0] == 0, (
        "Negative amounts should be filtered."
    )


def test_filtering_outliers_distance(sample_trip_data):
    """Verifies invalid trip distances are filtered."""
    dt = DataTransformation(config)
    result = dt._clean_data(sample_trip_data)

    # trip_distance must be > 0.5 and < 100
    # In sample_trip_data: 0.0 (too small), 150.0 (too big)
    assert (
        result.filter((pl.col("trip_distance") <= 0.5) | (pl.col("trip_distance") >= 100)).shape[0]
        == 0
    ), "Invalid distances should be filtered."


def test_filtering_outliers_amount(sample_trip_data):
    """Verifies excessive fares are filtered."""
    dt = DataTransformation(config)
    result = dt._clean_data(sample_trip_data)

    # total_amount must be <= 1000
    # In sample_trip_data: 1200.0 (too big)
    assert result.filter(pl.col("total_amount") > 1000).shape[0] == 0, (
        "Amounts > 1000 should be filtered."
    )


def test_data_consistency(sample_trip_data):
    """Ensures input data conforms to expected types post-cleaning."""
    dt = DataTransformation(config)
    result = dt._clean_data(sample_trip_data)

    # Verify that we have a DataFrame and filtering occurred.
    # Datetime type is verified in integration tests or Conductor tests.
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] < sample_trip_data.shape[0], (
        "No rows were filtered (expected filtering)."
    )
