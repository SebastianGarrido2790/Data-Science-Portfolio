"""
Unit tests for the Model Evaluation component.

These tests verify:
1. Model evaluation correctly loads data and model.
2. Generates predictions of correct shape.
3. Successfully logs to MLflow and persists metrics.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.components.model_evaluation import ModelEvaluation
from src.entity.config_entity import ModelEvaluationConfig


@pytest.fixture
def mock_eval_config(tmp_path):
    """Provides a mock configuration for ModelEvaluation."""
    model_path = tmp_path / "model.joblib"
    model_path.touch()

    return ModelEvaluationConfig(
        root_dir=tmp_path / "model_evaluation",
        test_data_path=tmp_path / "test.parquet",
        model_path=model_path,
        all_params={},
        metric_file_name=tmp_path / "metrics.json",
        mlflow_uri="file:./mlruns",
    )


def test_model_evaluation_initialization(mock_eval_config):
    """Tests if ModelEvaluation initializes correctly."""
    evaluator = ModelEvaluation(mock_eval_config)
    assert evaluator.config.mlflow_uri == "file:./mlruns"
    assert evaluator.config.test_data_path.name == "test.parquet"


@patch("mlflow.set_tracking_uri")
@patch("mlflow.set_experiment")
@patch("mlflow.start_run")
@patch("mlflow.log_metrics")
@patch("src.components.model_evaluation.joblib.load")
@patch("src.components.model_evaluation.pd.read_parquet")
@patch("src.components.model_evaluation.save_json")
def test_evaluate_workflow(
    mock_save_json,
    mock_read_parquet,
    mock_load_model,
    mock_log_metrics,
    mock_start_run,
    mock_set_exp,
    mock_set_uri,
    mock_eval_config,
):
    """Tests the full evaluation and MLflow logging workflow."""

    # Mock data setup
    mock_df = pd.DataFrame(
        {
            "feature1": np.random.rand(10),
            "feature2": np.random.rand(10),
            "tip_amount": np.random.rand(10),
        }
    )
    mock_read_parquet.return_value = mock_df

    # Mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = np.random.rand(10)
    mock_load_model.return_value = mock_model

    # Mock MLflow context manager
    mock_run = MagicMock()
    mock_start_run.return_value.__enter__.return_value = mock_run

    evaluator = ModelEvaluation(mock_eval_config)
    metrics = evaluator.evaluate()

    # Assertions
    # 1. Check data loading and preprocessing
    mock_read_parquet.assert_called_once_with(mock_eval_config.test_data_path)

    args, _ = mock_load_model.call_args
    assert args[0] == mock_eval_config.model_path

    # 2. Check model predict called with right shape (features only)
    args, _ = mock_model.predict.call_args
    x_test_passed = args[0]
    assert x_test_passed.shape == (10, 2)
    assert "tip_amount" not in x_test_passed.columns

    # 3. Check metrics dictionary
    assert "test_mae" in metrics
    assert "test_mse" in metrics
    assert "test_r2" in metrics

    # 4. Check MLflow and saving correctly
    mock_set_uri.assert_called_once_with("file:./mlruns")
    mock_set_exp.assert_called_once_with("NYC_Taxi_Tips_Evaluation")
    mock_log_metrics.assert_called_once_with(metrics)
    mock_save_json.assert_called_once()
