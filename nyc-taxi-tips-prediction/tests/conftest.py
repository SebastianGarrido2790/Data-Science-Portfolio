"""
Shared Pytest Fixtures.

This module provides common DataFrame fixtures for testing data processing components.
It includes:
1. `sample_trip_data`: Raw data with intentional anomalies for testing cleaning logic.
2. `cleaned_trip_data`: Cleaned data spanning multiple months for testing feature engineering.
"""

import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))


@pytest.fixture
def sample_trip_data():
    """
    Creates a sample DataFrame mimicking raw data for Data Transformation tests.
    Includes:
    - Normal row
    - Negative total_amount (should be dropped)
    - Zero/Negative trip_distance (should be dropped)
    - Outlier fare (>$1000) (should be dropped)
    - Null airport_fee (should be imputed to 0)
    """
    data = {
        "tpep_pickup_datetime": [
            "01/01/2023 12:00:00 AM",
            "01/01/2023 12:10:00 AM",
            "01/01/2023 12:20:00 AM",
            "01/01/2023 12:30:00 AM",
            "01/01/2023 12:40:00 AM",
        ],
        "trip_distance": [1.5, 0.0, 10.0, 150.0, 2.0],  # 0.0 and 150.0 are invalid
        "total_amount": [15.0, 20.0, -5.0, 1200.0, 10.0],  # -5.0 and 1200.0 are invalid
        "airport_fee": [1.25, None, 0.0, 0.0, 0.0],  # None should be 0
        "congestion_surcharge": [2.5, 2.5, 2.5, 2.5, None],  # None should be 0
        "passenger_count": [1, 2, 1, 1, 0],  # 0 should be 1
        "store_and_fwd_flag": ["N", "Y", "N", "N", "N"],  # Should be dropped
        "RatecodeID": [1, 99, 1, 1, None],  # None should be 99
    }
    return pl.DataFrame(data)


@pytest.fixture
def cleaned_trip_data():
    """
    Creates a sample DataFrame mimicking cleaned data for Feature Engineering tests.
    Includes timestamps covering different months to test splitting.
    """
    data = {
        "tpep_pickup_datetime": [
            datetime(2023, 1, 1, 0, 0),  # Train (Jan)
            datetime(2023, 4, 15, 12, 0),  # Train (Apr)
            datetime(2023, 9, 1, 8, 30),  # Val (Sept)
            datetime(2023, 11, 10, 18, 0),  # Test (Nov)
            datetime(2023, 12, 31, 23, 59),  # Test (Dec)
        ],
        "trip_distance": [2.5, 5.0, 1.2, 8.0, 3.3],
        "total_amount": [20.0, 35.0, 12.0, 50.0, 25.0],
    }
    return pl.DataFrame(data)
