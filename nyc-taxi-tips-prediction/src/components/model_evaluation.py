"""
Model Evaluation component for the NYC Taxi Tips Prediction pipeline.

This module evaluates the trained champion model on the test dataset,
calculates final performance metrics, logs them to MLflow, and saves
the metrics locally.
"""

import sys
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.entity.config_entity import ModelEvaluationConfig
from src.utils.common import save_json
from src.utils.exception import CustomExceptionError
from src.utils.logger import get_logger

logger = get_logger(__name__, headline="Model Evaluation")


class ModelEvaluation:
    """
    Evaluates the trained model against the hold-out test dataset.
    """

    def __init__(self, config: ModelEvaluationConfig):
        """
        Initializes the ModelEvaluation with configuration.

        Args:
            config (ModelEvaluationConfig): Configuration for evaluation.
        """
        self.config = config

    def evaluate(self) -> dict:
        """
        Loads the test data and model, generates predictions, calculates metrics,
        and logs to MLflow. Saves metrics as a JSON file.

        Returns:
            dict: The dictionary of calculated metrics.
        """
        try:
            # Set Tracking URI
            if hasattr(self.config, "mlflow_uri") and self.config.mlflow_uri:
                mlflow.set_tracking_uri(self.config.mlflow_uri)
                logger.info(f"MLflow Tracking URI set to: {self.config.mlflow_uri}")

            # Load test dataset
            target = self.config.target_column
            test_data = pd.read_parquet(self.config.test_data_path)

            x_test = test_data.drop([target], axis=1)
            y_test = test_data[target]

            # Drop non-numeric columns
            x_test = x_test.select_dtypes(include=["number"])
            logger.info("Loaded test dataset successfully.")

            # Load model
            model_dir = Path(self.config.model_path).parent
            model_files = list(model_dir.glob("*.joblib"))
            if not model_files:
                raise FileNotFoundError(f"No .joblib model found in {model_dir}")
            actual_model_path = model_files[0]
            model = joblib.load(actual_model_path)
            logger.info(f"Loaded champion model from {actual_model_path}")

            # Generate predictions
            y_pred = model.predict(x_test)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            metrics = {"test_mae": mae, "test_mse": mse, "test_r2": r2}

            logger.info(f"Test Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")

            # Save metrics locally
            save_json(Path(self.config.metric_file_name), data=metrics)

            # Log to MLflow
            mlflow.set_experiment("NYC_Taxi_Tips_Evaluation")
            with mlflow.start_run(run_name="Test_Set_Evaluation"):
                mlflow.log_metrics(metrics)
                logger.info("Metrics logged to MLflow successfully.")

            return metrics

        except Exception as e:
            raise CustomExceptionError(e, sys) from e
