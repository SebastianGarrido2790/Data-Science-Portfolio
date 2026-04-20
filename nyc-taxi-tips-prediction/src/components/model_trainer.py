"""
Model Trainer component for the NYC Taxi Tips Prediction pipeline.

This module implements the training of multiple candidate models,
benchmarking them using MLflow, and selecting/registering the champion model.

To view the results in your browser run:
    uv run mlflow ui
"""

import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.entity.config_entity import ModelTrainerConfig
from src.utils.exception import CustomExceptionError
from src.utils.logger import get_logger

logger = get_logger(__name__, headline="Component: Model Trainer")


class ModelTrainer:
    """
    Orchestrates the training and benchmarking of multiple regression models.
    """

    def __init__(self, config: ModelTrainerConfig):
        """
        Initializes the ModelTrainer with configuration.

        Args:
            config (ModelTrainerConfig): Configuration for training.
        """
        self.config = config

    def train_and_register(self) -> None:
        """
        Trains baseline and candidate models, logs to MLflow, and registers the champion.
        """
        try:
            # Set Tracking URI
            if hasattr(self.config, "mlflow_uri") and self.config.mlflow_uri:
                mlflow.set_tracking_uri(self.config.mlflow_uri)
                logger.info(f"MLflow Tracking URI set to: {self.config.mlflow_uri}")

            # Load datasets
            train_data = pd.read_parquet(self.config.train_data_path)
            val_data = pd.read_parquet(self.config.val_data_path)

            # Optional Subsampling (Local Fast Mode)
            if self.config.subsample_fraction < 1.0:
                logger.info(
                    f"Subsampling training data to {self.config.subsample_fraction * 100}%..."
                )
                train_data = train_data.sample(frac=self.config.subsample_fraction, random_state=42)
                logger.info(f"New training set shape: {train_data.shape}")

            # Separate features and target
            target = self.config.target_column
            x_train = train_data.drop([target], axis=1)
            y_train = train_data[target]
            x_val = val_data.drop([target], axis=1)
            y_val = val_data[target]

            # Drop non-numeric columns (like Timestamps) which algorithms can't handle
            x_train = x_train.select_dtypes(include=["number"])
            x_val = x_val.select_dtypes(include=["number"])

            logger.info(f"Numeric features selected: {list(x_train.columns)}")

            # Define candidate models
            models = {
                "Baseline": DummyRegressor(**self.config.all_params["Baseline"]),
                "ElasticNet": ElasticNet(**self.config.all_params["ElasticNet"]),
                "Ridge": Ridge(**self.config.all_params["Ridge"]),
                "RandomForest": RandomForestRegressor(**self.config.all_params["RandomForest"]),
                "XGBoost": XGBRegressor(**self.config.all_params["XGBoost"]),
                "GradientBoosting": GradientBoostingRegressor(
                    **self.config.all_params["GradientBoosting"]
                ),
            }

            results = []
            mlflow.set_experiment("NYC_Taxi_Tips_Training")

            for model_name, model in models.items():
                run_name = f"Run_{model_name}"
                with mlflow.start_run(run_name=run_name) as run:
                    logger.info(f"Training {model_name}...")

                    # Train
                    model.fit(x_train, y_train)

                    # Predict on Validation data
                    y_pred = model.predict(x_val)

                    # Metrics
                    mae = mean_absolute_error(y_val, y_pred)
                    mse = mean_squared_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)

                    logger.info(
                        f"{model_name} (Val) - MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}"
                    )

                    # Log params and metrics
                    mlflow.log_params(self.config.all_params.get(model_name, {}))
                    mlflow.log_metric("val_mae", mae)
                    mlflow.log_metric("val_mse", mse)
                    mlflow.log_metric("val_r2", r2)

                    # Log model
                    if model_name == "XGBoost":
                        mlflow.xgboost.log_model(model, artifact_path="model")  # type: ignore
                    else:
                        mlflow.sklearn.log_model(model, artifact_path="model")  # type: ignore

                    results.append(
                        {
                            "name": model_name,
                            "mae": mae,
                            "mse": mse,
                            "r2": r2,
                            "model": model,
                            "run_id": run.info.run_id,
                        }
                    )

            # --- Multi-Metric Champion Selection ---
            logger.info("Computing multi-metric weighted score for champion selection...")

            # Normalize metrics and compute score
            metrics_to_use = self.config.selection_metrics

            # Find min/max for each metric across all models for normalization
            bounds = {}
            for m in metrics_to_use:
                vals = [r[m] for r in results]
                bounds[m] = {"min": min(vals), "max": max(vals)}

            for r in results:
                total_score = 0.0
                for m, weight in metrics_to_use.items():
                    val = r[m]
                    b = bounds[m]
                    # Avoid division by zero if all models have same metric value
                    diff = b["max"] - b["min"]
                    if diff == 0:
                        norm_val = 1.0
                    else:
                        norm_val = (val - b["min"]) / diff if m == "r2" else (b["max"] - val) / diff

                    total_score += weight * norm_val

                r["final_score"] = total_score

            # Select model with highest weighted score
            champion = max(results, key=lambda x: x["final_score"])

            logger.info(
                f"Champion Model Selected! Name: {champion['name']} | "
                f"Final Weighted Score: {champion['final_score']:.4f}"
            )
            # Log individual metrics of the champion for reference
            metric_details = " | ".join([f"{k.upper()}: {champion[k]:.4f}" for k in metrics_to_use])
            logger.info(f"Selected metrics: {metrics_to_use}")
            logger.info(f"Champion breakdown: {metric_details}")

            # Clear existing model files locally to avoid clutter
            for f in os.listdir(self.config.root_dir):
                if f.endswith(".joblib"):
                    os.remove(os.path.join(self.config.root_dir, f))

            # Save champion model locally
            model_filename = f"{champion['name']}.joblib"
            model_path = os.path.join(self.config.root_dir, model_filename)
            joblib.dump(champion["model"], model_path)
            logger.info(f"Champion model saved locally to {model_path}")

            # Register Champion
            model_uri = f"runs:/{champion['run_id']}/model"
            mlflow.register_model(model_uri, champion["name"])
            logger.info(f"Registered champion model as '{champion['name']}'")

        except Exception as e:
            raise CustomExceptionError(e, sys) from e
