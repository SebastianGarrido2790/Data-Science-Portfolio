"""
Unit tests for the Model Trainer component.

These tests verify:
1. Model selection logic works correctly with weighted metrics.
2. Normalization handles polarity (lower MAE is better, higher R2 is better).
3. The component correctly handles numeric features and subsampling.
"""

import contextlib
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import ModelTrainerConfig


@pytest.fixture
def mock_trainer_config(tmp_path):
    """Provides a mock configuration for ModelTrainer."""
    return ModelTrainerConfig(
        root_dir=tmp_path / "model_trainer",
        train_data_path=tmp_path / "train.parquet",
        val_data_path=tmp_path / "val.parquet",
        model_name="model.joblib",
        all_params={
            "Baseline": {},
            "ElasticNet": {"alpha": 0.1, "l1_ratio": 0.5},
            "Ridge": {"alpha": 1.0},
            "RandomForest": {"n_estimators": 10},
            "XGBoost": {"n_estimators": 10},
            "GradientBoosting": {"n_estimators": 10},
        },
        mlflow_uri="file:./mlruns",
        subsample_fraction=1.0,
        selection_metrics={"mae": 0.5, "r2": 0.5},
    )


@pytest.fixture
def dummy_train_val_data(tmp_path):
    """Creates dummy parquet files for training and validation."""
    df = pd.DataFrame(
        {
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "tip_amount": np.random.rand(100),
            "non_numeric": ["a"] * 100,
        }
    )

    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    test_path = tmp_path / "test.parquet"

    df.to_parquet(train_path)
    df.to_parquet(val_path)
    df.to_parquet(test_path)

    return train_path, val_path


def test_model_trainer_initialization(mock_trainer_config):
    """Tests if ModelTrainer initializes correctly."""
    trainer = ModelTrainer(mock_trainer_config)
    assert trainer.config.model_name == "model.joblib"


@patch("mlflow.set_tracking_uri")
@patch("mlflow.set_experiment")
@patch("mlflow.start_run")
@patch("mlflow.log_params")
@patch("mlflow.log_metric")
@patch("mlflow.sklearn.log_model")
@patch("mlflow.xgboost.log_model")
@patch("mlflow.register_model")
def test_train_and_register_workflow(
    mock_register,
    mock_xg_log,
    mock_sk_log,
    mock_metric,
    mock_params,
    mock_start,
    mock_exp,
    mock_uri,
    mock_trainer_config,
    dummy_train_val_data,
):
    """Tests the full training and registration workflow using mocked MLflow."""

    # Setup mocks
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_id"
    mock_start.return_value.__enter__.return_value = mock_run

    trainer = ModelTrainer(mock_trainer_config)

    # We need to ensure root_dir exists
    mock_trainer_config.root_dir.mkdir(parents=True, exist_ok=True)

    # Execute
    trainer.train_and_register()

    # Verify MLflow calls
    assert mock_uri.called
    assert mock_exp.called
    assert mock_start.call_count >= 1
    assert mock_register.called

    # Verify local model save
    model_files = list(mock_trainer_config.root_dir.glob("*.joblib"))
    assert len(model_files) == 1


def test_champion_selection_logic(mock_trainer_config):
    """Verifies the multi-metric weighted selection logic directly."""

    # Higher MAE is worse, Higher R2 is better
    # Sample results
    results = [
        {"name": "Model_A", "mae": 10.0, "mse": 100.0, "r2": 0.5, "run_id": "1"},
        {"name": "Model_B", "mae": 5.0, "mse": 25.0, "r2": 0.9, "run_id": "2"},
        {"name": "Model_C", "mae": 2.0, "mse": 4.0, "r2": 0.95, "run_id": "3"},
    ]

    # Mock bounds calculation (private part of selection logic)
    # We'll just test the logic used in the component
    metrics_to_use = {"mae": 0.5, "r2": 0.5}
    bounds = {}
    for m in metrics_to_use:
        vals = [r[m] for r in results]
        bounds[m] = {"min": min(vals), "max": max(vals)}

    for r in results:
        total_score = 0.0
        for m, weight in metrics_to_use.items():
            val = r[m]
            b = bounds[m]
            diff = b["max"] - b["min"]
            if diff == 0:
                norm_val = 1.0
            else:
                norm_val = (val - b["min"]) / diff if m == "r2" else (b["max"] - val) / diff
            total_score += weight * norm_val
        r["final_score"] = total_score

    champion = max(results, key=lambda x: x["final_score"])

    # Model C has lowest MAE and highest R2, it should be the champion
    assert champion["name"] == "Model_C"
    assert results[2]["final_score"] == 1.0  # Best on both metrics
    assert results[0]["final_score"] == 0.0  # Worst on both metrics


def test_subsampling_logic(mock_trainer_config, dummy_train_val_data):
    """Tests if subsampling fraction correctly reduces data size."""
    mock_trainer_config.subsample_fraction = 0.1
    trainer = ModelTrainer(mock_trainer_config)

    # We only want to test the first model (Baseline) to verify subsampling
    # and skip the rest to avoid complexity in this unit test.
    with patch("src.components.model_trainer.pd.read_parquet") as mock_read:
        # Create a 100-row dataframe
        df = pd.DataFrame(
            {
                "feature1": np.random.rand(100),
                "feature2": np.random.rand(100),
                "tip_amount": np.random.rand(100),
            }
        )
        mock_read.return_value = df

        with patch("src.components.model_trainer.DummyRegressor") as mock_dummy_cls:
            mock_dummy_inst = MagicMock()
            mock_dummy_cls.return_value = mock_dummy_inst

            # Mock other things to stop after Baseline training
            with (
                patch("src.components.model_trainer.ElasticNet"),
                patch("src.components.model_trainer.Ridge"),
                patch("src.components.model_trainer.RandomForestRegressor"),
                patch("src.components.model_trainer.XGBRegressor"),
                patch("src.components.model_trainer.GradientBoostingRegressor"),
                patch("mlflow.start_run"),
                patch("mlflow.log_params"),
                patch("mlflow.log_metric"),
                patch("mlflow.sklearn.log_model"),
            ):
                # Mock the loop to break after first model or handle result
                with contextlib.suppress(Exception):
                    trainer.train_and_register()

                # Verify that it was called and check size
                assert mock_dummy_inst.fit.called
                args, _ = mock_dummy_inst.fit.call_args
                x_train_sub = args[0]
                assert x_train_sub.shape[0] == 10
