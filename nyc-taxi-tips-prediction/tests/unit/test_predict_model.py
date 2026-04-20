"""
Unit tests for the Predict Model component (Batch inference).

These tests verify:
1. Correct loading of batched data.
2. Proper ingestion (handling data with or without the target feature).
3. Shape consistency of output predictions.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.components.predict_model import PredictModel
from src.entity.config_entity import PredictModelConfig


@pytest.fixture
def mock_predict_config(tmp_path):
    """Provides a mock configuration for PredictModel."""
    model_path = tmp_path / "model.joblib"
    model_path.touch()

    return PredictModelConfig(
        root_dir=tmp_path / "predictions",
        test_data_path=tmp_path / "test.parquet",
        model_path=model_path,
        output_filename="inference_results.csv",
        target_column="tip_amount",
    )


def test_predict_model_initialization(mock_predict_config):
    """Tests if PredictModel initializes correctly."""
    predictor = PredictModel(mock_predict_config)
    assert predictor.config.model_path.name == "model.joblib"


@patch("src.components.predict_model.joblib.load")
@patch("src.components.predict_model.pd.read_parquet")
def test_predict_model_inference_with_target(
    mock_read_parquet, mock_load_model, mock_predict_config, tmp_path
):
    """Tests inference simulating a test-set evaluation (with target column present)."""

    # Setup mock data WITH 'tip_amount' and 'VendorID'
    mock_df = pd.DataFrame(
        {
            "VendorID": [1, 2, 1],
            "feature1": [0.5, 0.2, 0.9],
            "tip_amount": [2.0, 1.5, 3.0],
        }
    )
    mock_read_parquet.return_value = mock_df

    # Mock model
    mock_model = MagicMock()
    # Return 3 predictions
    mock_model.predict.return_value = np.array([1.9, 1.6, 2.8])
    mock_load_model.return_value = mock_model

    predictor = PredictModel(mock_predict_config)
    predictor.perform_inference()

    # Asserts
    # Data load and preprocessing
    args, _ = mock_load_model.call_args
    assert args[0] == mock_predict_config.model_path

    args, _ = mock_model.predict.call_args
    x_passed = args[0]

    # Should drop tip_amount
    assert "tip_amount" not in x_passed.columns
    # VendorID is numeric, so it should be passed according to the current logic
    assert "VendorID" in x_passed.columns
    assert x_passed.shape == (3, 2)

    # Check output persistence
    output_file_path = mock_predict_config.root_dir / mock_predict_config.output_filename
    assert output_file_path.exists()

    result_df = pd.read_csv(output_file_path)
    assert "predicted_tip" in result_df.columns
    assert "VendorID" in result_df.columns
    assert result_df.shape == (3, 2)
    assert result_df["predicted_tip"].iloc[0] == 1.9


@patch("src.components.predict_model.joblib.load")
@patch("src.components.predict_model.pd.read_parquet")
def test_predict_model_inference_production_data(
    mock_read_parquet, mock_load_model, mock_predict_config, tmp_path
):
    """Tests inference with pure production data lacking the target 'tip_amount'."""

    # Setup mock data WITHOUT 'tip_amount'
    mock_df = pd.DataFrame(
        {
            "feature1": [0.5, 0.2, 0.9],
            "feature2": [10.0, 20.0, 30.0],
            "non_numeric": ["A", "B", "A"],
        }
    )
    mock_read_parquet.return_value = mock_df

    # Mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([2.5, 3.5, 4.5])
    mock_load_model.return_value = mock_model

    # Custom config for production data test
    mock_predict_config.root_dir = tmp_path / "predictions_prod"
    mock_predict_config.output_filename = "prod_results.csv"

    predictor = PredictModel(mock_predict_config)
    predictor.perform_inference()

    # Asserts
    args, _ = mock_model.predict.call_args
    x_passed = args[0]

    # Should only contain purely numeric features
    assert "non_numeric" not in x_passed.columns
    assert x_passed.shape == (3, 2)

    output_file_path = mock_predict_config.root_dir / mock_predict_config.output_filename
    assert output_file_path.exists()

    result_df = pd.read_csv(output_file_path)
    # Target wasn't in input, so output should only have predicted_tip since there is no VendorID
    assert "predicted_tip" in result_df.columns
    assert "VendorID" not in result_df.columns
    assert result_df.shape == (3, 1)
