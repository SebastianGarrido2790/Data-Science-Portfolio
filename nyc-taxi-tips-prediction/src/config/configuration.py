"""
Configuration Management module for the MLOps pipeline.

This module is the 'Brain' of the system. It orchestrates the reading of
YAML configurations and parameters, transforms them into validated
Pydantic entities, and ensures that the necessary artifact directories
are created before any pipeline stage begins.
"""

from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    FeatureEngineeringConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
    PredictModelConfig,
)
from src.utils.common import create_directories, read_yaml
from src.utils.mlflow_config import get_mlflow_uri


class ConfigurationManager:
    """
    Manages the lifecycle of project configurations.

    This class centralizes the loading of 'config.yaml', 'params.yaml',
    and 'schema.yaml' and provides specialized methods to retrieve
    configuration objects for each pipeline stage.
    """

    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH,
    ):
        """
        Initializes the ConfigurationManager.

        Args:
            config_filepath (Path): Path to the system configuration file.
            params_filepath (Path): Path to the model hyperparameters file.
            schema_filepath (Path): Path to the data schema file.
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        # Ensure the root artifacts directory exists immediately
        create_directories([self.config["artifacts_root"]])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieves the configuration for the Data Ingestion stage.

        Returns:
            DataIngestionConfig: Validated ingestion configuration object.
        """
        config = self.config["data_ingestion"]
        create_directories([config["root_dir"]])

        return DataIngestionConfig(
            root_dir=config["root_dir"],
            source_data_path=config["source_data_path"],
            taxi_zones_path=config["taxi_zones_path"],
            output_data_path=config["output_data_path"],
            all_schema=self.schema,
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Retrieves the configuration for the Data Validation stage.

        Returns:
            DataValidationConfig: Validated validation configuration object.
        """
        config = self.config["data_validation"]
        create_directories([config["root_dir"]])

        return DataValidationConfig(
            root_dir=config["root_dir"],
            STATUS_FILE=config["STATUS_FILE"],
            unzip_dir=config["unzip_dir"],
            all_schema=self.schema,
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Retrieves the configuration for the Data Transformation stage.

        Returns:
            DataTransformationConfig: Validated transformation configuration object.
        """
        config = self.config["data_transformation"]
        create_directories([config["root_dir"]])

        cleaning = self.params.get("DataCleaning", {})
        return DataTransformationConfig(
            root_dir=config["root_dir"],
            data_path=config["data_path"],
            min_trip_distance=cleaning.get("min_trip_distance", 0.5),
            max_trip_distance=cleaning.get("max_trip_distance", 100.0),
            min_total_amount=cleaning.get("min_total_amount", 3.70),
            max_total_amount=cleaning.get("max_total_amount", 1000.0),
        )

    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:
        """
        Retrieves the configuration for the Feature Engineering stage.

        Returns:
            FeatureEngineeringConfig: Validated feature engineering configuration object.
        """
        config = self.config["feature_engineering"]
        create_directories([config["root_dir"]])

        fe = self.params.get("FeatureEngineering", {})
        return FeatureEngineeringConfig(
            root_dir=config["root_dir"],
            data_path=config["data_path"],
            target_column=self.schema["TARGET_COLUMN"]["name"],
            train_months_start=fe.get("train_months_start", 1),
            train_months_end=fe.get("train_months_end", 8),
            val_months_start=fe.get("val_months_start", 9),
            val_months_end=fe.get("val_months_end", 10),
            test_months_start=fe.get("test_months_start", 11),
            test_months_end=fe.get("test_months_end", 12),
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Retrieves the configuration for the Model Training stage.

        Returns:
            ModelTrainerConfig: Validated training configuration object.
        """
        config = self.config["model_trainer"]
        create_directories([config["root_dir"]])

        return ModelTrainerConfig(
            root_dir=config["root_dir"],
            train_data_path=config["train_data_path"],
            val_data_path=config["val_data_path"],
            model_name=config["model_name"],
            all_params=self.params,
            mlflow_uri=get_mlflow_uri(),
            subsample_fraction=self.params.get("Training", {}).get("subsample_fraction", 1.0),
            selection_metrics=self.params.get("Training", {}).get(
                "selection_metrics", {"mae": 1.0}
            ),
            target_column=self.schema["TARGET_COLUMN"]["name"],
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Retrieves the configuration for the Model Evaluation stage.

        Returns:
            ModelEvaluationConfig: Validated evaluation configuration object.
        """
        config = self.config["model_evaluation"]
        create_directories([config["root_dir"]])

        return ModelEvaluationConfig(
            root_dir=config["root_dir"],
            test_data_path=config["test_data_path"],
            model_path=config["model_path"],
            all_params=self.params,
            metric_file_name=config["metric_file_name"],
            mlflow_uri=get_mlflow_uri(),
            target_column=self.schema["TARGET_COLUMN"]["name"],
        )

    def get_predict_model_config(self) -> PredictModelConfig:
        """
        Retrieves the configuration for the Predict Model (Batch Inference) stage.

        Returns:
            PredictModelConfig: Validated inference configuration object.
        """
        config = self.config["predict_model"]
        eval_config = self.config["model_evaluation"]
        create_directories([config["root_dir"]])

        return PredictModelConfig(
            root_dir=config["root_dir"],
            test_data_path=eval_config["test_data_path"],
            model_path=eval_config["model_path"],
            output_filename=config["output_filename"],
            target_column=self.schema["TARGET_COLUMN"]["name"],
        )

    def get_fare_constants(self) -> dict[str, float]:
        """
        Retrieves NYC-specific fare constants.

        Returns:
            dict[str, float]: Dictionary of fare constants (mta_tax, extra, etc.)
        """
        return self.config.get("fare_constants", {})
