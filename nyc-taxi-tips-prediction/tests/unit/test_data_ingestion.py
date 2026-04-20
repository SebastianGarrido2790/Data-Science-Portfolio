"""
Data Ingestion Unit Tests.

This module tests the logic in `src/components/data_ingestion.py`.
Scope:
1. Merge Logic: Validates that taxi zones are correctly joined to trip data (Pickup and Dropoff).
"""

from pathlib import Path

import polars as pl
import pytest

from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig

# Dummy configuration
config = DataIngestionConfig(
    root_dir=Path("./temp"),
    source_data_path=Path("./dummy_source.csv"),
    taxi_zones_path=Path("./dummy_zones.csv"),
    output_data_path=Path("./dummy_output.parquet"),
    all_schema={},
)


@pytest.fixture
def mock_trips_df():
    """Sample trip data with location IDs."""
    data = {
        "tpep_pickup_datetime": ["2023-01-01 10:00:00", "2023-01-01 10:30:00"],
        "PULocationID": [1, 2],
        "DOLocationID": [2, 999],  # 999 is invalid/missing
    }
    return pl.DataFrame(data)


@pytest.fixture
def mock_zones_df():
    """Sample zone data."""
    data = {
        "LocationID": [1, 2],
        "Borough": ["Manhattan", "Queens"],
        "Zone": ["Midtown", "JFK Airport"],
    }
    return pl.DataFrame(data)


def test_merge_taxi_zones(mock_trips_df, mock_zones_df):
    """Verifies that zones are correctly merged into trips."""
    di = DataIngestion(config)
    enriched_df = di._merge_taxi_zones(mock_trips_df, mock_zones_df)

    # Check Columns
    expected_cols = ["PU_Borough", "PU_Zone", "DO_Borough", "DO_Zone"]
    for col in expected_cols:
        assert col in enriched_df.columns, f"Missing enriched column: {col}"

    # Verify Correctness
    # Row 0: PU=1 (Manhattan/Midtown), DO=2 (Queens/JFK)
    row0 = enriched_df.row(0, named=True)
    assert row0["PU_Zone"] == "Midtown"
    assert row0["DO_Zone"] == "JFK Airport"

    # Verify Missing/Invalid Join
    # Row 1: DO=999 (should be null)
    row1 = enriched_df.row(1, named=True)
    assert row1["DO_Zone"] is None, "Invalid LocationID should result in null Zone."
