"""
Data entity definitions for project configuration.

This module defines the Pydantic models (data contracts) used throughout the pipeline.
Using Pydantic ensures that all configuration values are validated and fully typed with generics
(e.g., dict[str, Any] and dict[str, float]) providing type hints for nested configurations
across the pipelines, and type-presence before the pipeline begins execution.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class DataIngestionConfig(BaseModel):
    """
    Configuration for the data ingestion stage.

    Attributes:
        root_dir (Path): Directory where ingestion artifacts are stored.
        source_data_path (Path): Path to the raw distilled dataset.
        taxi_zones_path (Path): Path to the taxi zones CSV.
        output_data_path (Path): Path where the enriched Parquet file will be saved.
    """

    # ConfigDict(extra="forbid") prevents the addition of unexpected fields to the model.
    model_config = ConfigDict(extra="forbid")

    root_dir: Path
    source_data_path: Path
    taxi_zones_path: Path
    output_data_path: Path
    all_schema: dict[str, Any]


class DataValidationConfig(BaseModel):
    """
    Configuration for the data validation stage.

    Attributes:
        root_dir (Path): Directory for validation reports and status files.
        STATUS_FILE (str): Path to the text file indicating if validation passed.
        unzip_dir (Path): Path to the data file to be validated.
    """

    model_config = ConfigDict(extra="forbid")

    root_dir: Path
    STATUS_FILE: str
    unzip_dir: Path
    all_schema: dict[str, Any]


class DataTransformationConfig(BaseModel):
    """
    Configuration for the data transformation stage.

    Attributes:
        root_dir (Path): Directory for transformed feature storage.
        data_path (Path): Path to the input data for transformation.
        min_trip_distance (float): Minimum valid trip distance in miles.
        max_trip_distance (float): Maximum valid trip distance in miles.
        min_total_amount (float): Minimum total fare in USD (NYC base).
        max_total_amount (float): Maximum plausible total fare in USD.
    """

    model_config = ConfigDict(extra="forbid")

    root_dir: Path
    data_path: Path
    min_trip_distance: float = 0.5
    max_trip_distance: float = 100.0
    min_total_amount: float = 3.70
    max_total_amount: float = 1000.0


class FeatureEngineeringConfig(BaseModel):
    """
    Configuration for the feature engineering stage.

    Attributes:
        root_dir (Path): Directory where engineered features are saved.
        data_path (Path): Path to the cleaned input data.
        target_column (str): Name of the target variable (from schema.yaml).
        train_months_start (int): First month of training split.
        train_months_end (int): Last month of training split.
        val_months_start (int): First month of validation split.
        val_months_end (int): Last month of validation split.
        test_months_start (int): First month of test split.
        test_months_end (int): Last month of test split.
    """

    model_config = ConfigDict(extra="forbid")

    root_dir: Path
    data_path: Path
    target_column: str = "tip_amount"
    train_months_start: int = 1
    train_months_end: int = 8
    val_months_start: int = 9
    val_months_end: int = 10
    test_months_start: int = 11
    test_months_end: int = 12


class ModelTrainerConfig(BaseModel):
    """
    Configuration for the model training stage, including candidates.

    Attributes:
        root_dir (Path): Directory where model artifacts are saved.
        train_data_path (Path): Path to the training dataset.
        model_name (str): Filename for the saved champion model.
        all_params (dict): Dictionary containing hyperparameters for all candidate models.
        mlflow_uri (str): URI for the MLflow tracking server.
    """

    model_config = ConfigDict(extra="forbid")

    root_dir: Path
    train_data_path: Path
    val_data_path: Path
    model_name: str
    all_params: dict[str, dict[str, Any]]
    mlflow_uri: str
    subsample_fraction: float
    selection_metrics: dict[str, float]
    target_column: str = "tip_amount"


class ModelEvaluationConfig(BaseModel):
    """
    Configuration for the model evaluation stage and MLflow tracking.

    Attributes:
        root_dir (Path): Directory for evaluation metrics and artifacts.
        test_data_path (Path): Path to the test dataset for evaluation.
        model_path (Path): Path to the trained model artifact.
        all_params (dict): Dictionary of all hyperparameters logged to MLflow.
        metric_file_name (Path): Path to the JSON file where metrics are saved.
        mlflow_uri (str): URI for the MLflow tracking server.
    """

    model_config = ConfigDict(extra="forbid")

    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict[str, dict[str, Any]]
    metric_file_name: Path
    mlflow_uri: str
    target_column: str = "tip_amount"


class PredictModelConfig(BaseModel):
    """
    Configuration for the simulated batch inference stage.
    """

    model_config = ConfigDict(extra="forbid")

    root_dir: Path
    test_data_path: Path
    model_path: Path
    output_filename: str
    target_column: str = "tip_amount"
