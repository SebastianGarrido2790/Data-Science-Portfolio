"""
This module handles the evaluation stage and MLflow tracking.
It logs parameters, metrics, and artifacts to MLflow for experiment tracking.

Usage:
    Run the entire pipeline (including evaluation) with:
        uv run python main.py
    Or run only the evaluation stage with:
        uv run python -m src.pipeline.stage_06_evaluation
    To view the tracked experiments, you can launch the MLflow UI:
        uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
        NOTE: This is what you need to run, because we configured the code to write
        to this database in the .env file.
"""

import sys
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import pandas as pd

from src.entity.config_entity import ModelEvaluationConfig
from src.utils.common import save_json
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluation:
    """
    This class is responsible for calculating evaluation metrics, saving them locally,
    and tracking the experiment using MLflow. It integrates with the project's
    configuration management to ensure all parameters and results are logged
    for reproducibility.

    Attributes:
        config (ModelEvaluationConfig): Configuration entity containing data paths,
            MLflow tracking URI, and all pipeline parameters.

    """

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def log_into_mlflow(self) -> None:
        """
        Logs parameters and basic metrics to MLflow.

        This method performs the following steps:
        1. Reads the test dataset from the specified path.
        2. Calculates basic metrics (e.g., number of records).
        3. Saves these metrics to a local JSON file.
        4. Configures MLflow tracking, setting the registry URI and experiment name.
        5. Starts a new MLflow run and logs all pipeline parameters.
        6. Logs the calculated metrics.
        7. Handles model logging (currently commented out for ChromaDB integration).

        Raises:
            CustomException: If any error occurs during the logging process.
        """
        try:
            test_data = pd.read_csv(self.config.data_path)

            # Simple metrics for a recommender index
            metrics = {
                "num_test_records": float(len(test_data)),
            }

            # Save metrics locally
            save_json(path=Path(self.config.root_dir) / "metrics.json", data=metrics)

            # MLflow configuration
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            mlflow.set_experiment(self.config.experiment_name)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            # all_params is a dict at type-check time, so use bracket access
            provider = self.config.all_params["model_trainer"]["embedding_provider"]
            model_full_name = self.config.all_params["model_trainer"]["model_name"]
            model_short_name = str(model_full_name).split("/")[-1]
            run_name = f"{provider}_{model_short_name}_{timestamp}"

            with mlflow.start_run(run_name=run_name):
                # Log all parameters from the pipeline
                # flattened_params = self._flatten_dict(self.config.all_params)
                # Since all_params is a ConfigBox, we can just log sections or iterate
                for section, content in self.config.all_params.items():
                    if isinstance(content, dict):
                        for key, value in content.items():
                            mlflow.log_param(f"{section}_{key}", value)

                mlflow.log_metrics(metrics)

                # Model logging (if applicable)
                # For ChromaDB, we track the directory as an artifact rather than a 'model'
                if tracking_url_type_store != "file":
                    # Registering the model is only supported for certain flavors
                    # and usually requires a model signature.
                    # For now, we just log the parameters.
                    pass
                else:
                    # mlflow.sklearn.log_model(model, "model", registered_model_name="BookRecommender")
                    pass

            logger.info("Successfully logged metrics and parameters to MLflow.")

        except Exception as e:
            raise CustomException(e, sys)

    def _flatten_dict(self, d, parent_key="", sep="_") -> dict:
        """Helper to flatten nested config/params dictionaries."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
