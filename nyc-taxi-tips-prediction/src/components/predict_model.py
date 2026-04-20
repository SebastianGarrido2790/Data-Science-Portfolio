"""
Predict Model component for the NYC Taxi Tips Prediction pipeline.

This module simulates an inference pipeline by loading a batch of data,
generating predictions using the champion model, and saving the results.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

from src.entity.config_entity import PredictModelConfig
from src.utils.common import create_directories
from src.utils.exception import CustomExceptionError
from src.utils.logger import get_logger

logger = get_logger(__name__, headline="Model Inference")


class PredictModel:
    """
    Simulates batch inference using the champion model and fresh data.
    """

    def __init__(self, config: PredictModelConfig):
        """
        Initializes the PredictModel with the inference configuration.

        Args:
            config (PredictModelConfig): Configuration for model path and data.
        """
        self.config = config

    def perform_inference(self) -> None:
        """
        Loads the test data (as fresh batch), loads the model, generates predictions,
        and saves the output predictions to a CSV file.
        """
        try:
            # Ensure predictions directory exists
            create_directories([self.config.root_dir])

            # Load dataset representing the new batch of data
            # Here, we use the test set as our fresh data for simulation.
            logger.info(f"Loading incoming batch data from {self.config.test_data_path}")
            batch_data = pd.read_parquet(self.config.test_data_path)

            # Extract features (drop target as it would not be available in real inference)
            target = self.config.target_column
            if target in batch_data.columns:
                x_batch = batch_data.drop([target], axis=1)
            else:
                x_batch = batch_data.copy()

            # Process features for prediction (keep only numeric)
            x_batch_processed = x_batch.select_dtypes(include=["number"])

            # Load the model directly from local system
            # NOTE: Production systems might load from MLflow Model Registry like:
            # import mlflow.pyfunc
            # model_name = "champion_model_name"
            # model_version = "latest"
            # model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
            model_dir = Path(self.config.model_path).parent
            model_files = list(model_dir.glob("*.joblib"))
            if not model_files:
                raise FileNotFoundError(f"No .joblib model found in {model_dir}")
            actual_model_path = model_files[0]
            model = joblib.load(actual_model_path)
            logger.info(f"Loaded model from {actual_model_path}")

            # Predict
            predictions = model.predict(x_batch_processed)

            # Persist results
            # We join predictions with an identifier like VendorID or tpep_pickup_datetime
            results_df = pd.DataFrame()

            # Since VendorID might be categorical/encoded, we include what's available
            # Let's concatenate the predictions directly
            # Fallback to index if no distinct IDs are present to pass along
            results_df["predicted_tip"] = predictions
            if "VendorID" in batch_data.columns:
                results_df["VendorID"] = batch_data["VendorID"].values

            output_path = self.config.root_dir / self.config.output_filename
            results_df.to_csv(output_path, index=False)

            logger.info(f"Inference complete! Results saved to {output_path}")

        except Exception as e:
            raise CustomExceptionError(e, sys) from e
