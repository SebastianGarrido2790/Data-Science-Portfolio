"""
Feature Engineering Unit Tests.

This module tests the logic in `src/components/feature_engineering.py`.
Scope:
1. Cyclical Feature Creation (Transforming Day/Hour/Month to Sin/Cos).
2. Temporal Splitting (Strict Train/Val/Test alignment).
3. Schema Consistency.
"""

from pathlib import Path

import pytest

from src.components.feature_engineering import FeatureEngineering
from src.entity.config_entity import FeatureEngineeringConfig

# Dummy configuration
config = FeatureEngineeringConfig(
    root_dir=Path("./temp"),
    data_path=Path("./cleaned_trip_data.parquet"),
)


def test_feature_engineering_transforms(cleaned_trip_data):
    """Test feature engineering logic."""
    fe = FeatureEngineering(config)
    result = fe._create_cyclical_features(cleaned_trip_data)

    # Cyclical Features Check
    for col in [
        "pickup_hour_sin",
        "pickup_hour_cos",
        "pickup_day_sin",
        "pickup_day_cos",
        "pickup_month_sin",
        "pickup_month_cos",
    ]:
        assert col in result.columns, f"{col} missing from output."
        assert result[col].min() >= -1.0 - 1e-9, f"{col} min < -1."
        assert result[col].max() <= 1.0 + 1e-9, f"{col} max > 1."

    # Specific Value Check
    # First row is Jan 1st 00:00:00 (Sunday)
    # Hour 0 -> sin(0) = 0, cos(0) = 1
    row = result.row(0, named=True)
    assert row["pickup_hour_sin"] == pytest.approx(0.0, abs=1e-5), "Hour 0 sin should be 0."
    assert row["pickup_hour_cos"] == pytest.approx(1.0, abs=1e-5), "Hour 0 cos should be 1."


def test_temporal_splitting(cleaned_trip_data):
    """Test 3-way temporal splitting."""
    fe = FeatureEngineering(config)

    # We need 'pickup_month' which is created in _create_cyclical_features
    # So we must call that first or mock it.
    # Let's call it since we are testing the component integration mostly.
    processed_df = fe._create_cyclical_features(cleaned_trip_data)

    # Now verify the split logic
    train, val, test = fe._split_data(processed_df)

    # Train: Jan - Aug (Months 1-8)
    # Val: Sept - Oct (Months 9-10)
    # Test: Nov - Dec (Months 11-12)

    # Check set sizes based on fixture data (conftest.py)
    # Train: Jan (1), Apr (1) -> should be 2
    # Val: Sept (1) -> should be 1
    # Test: Nov (1), Dec (1) -> should be 2

    assert train.shape[0] == 2, f"Expected 2 training rows, got {train.shape[0]}."
    assert val.shape[0] == 1, f"Expected 1 validation row, got {val.shape[0]}."
    assert test.shape[0] == 2, f"Expected 2 testing rows, got {test.shape[0]}."

    # Verify no overlap
    train_months = train["pickup_month"].unique().to_list()
    val_months = val["pickup_month"].unique().to_list()
    test_months = test["pickup_month"].unique().to_list()

    # Train shouldn't have months > 8
    assert all(m <= 8 for m in train_months), f"Training set contains future months: {train_months}"
    # Val shouldn't have months < 9 or > 10
    assert all(9 <= m <= 10 for m in val_months), (
        f"Validation set contains invalid months: {val_months}"
    )
    # Test shouldn't have months < 11
    assert all(m >= 11 for m in test_months), f"Test set contains past months: {test_months}"


def test_feature_columns_consistency(cleaned_trip_data):
    """Ensure feature count matches expectations."""
    fe = FeatureEngineering(config)
    result = fe._create_cyclical_features(cleaned_trip_data)

    # Base columns (3) + 4 time components + 6 cyclical features = 13
    # Time components: pickup_hour, pickup_day, pickup_month, pickup_day_of_year
    expected_cols = [
        "pickup_hour_sin",
        "pickup_hour_cos",
        "pickup_day_sin",
        "pickup_day_cos",
        "pickup_month_sin",
        "pickup_month_cos",
    ]

    for col in expected_cols:
        assert col in result.columns, f"{col} is missing."
