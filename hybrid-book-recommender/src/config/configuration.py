"""
Configuration Manager for the MLOps Pipeline.

- This module serves as the 'Brain' of the system, responsible for coordinating
configurations and parameters across the pipeline.
- It centralizes the orchestration of configurations and parameters,
integrating with DVC for data versioning.
- It transforms raw YAML inputs into strictly-typed Configuration Entities,
providing a robust and reproducible interface for all downstream pipeline components.
"""

import os
import sys
from pathlib import Path

import dvc.api
from box import ConfigBox

from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, PROJECT_ROOT, SCHEMA_FILE_PATH
from src.entity.config_entity import (
    AgentConfig,
    BatchPredictionConfig,
    DataEnrichmentConfig,
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    InferenceConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
    SchemaConfig,
    ToneAnalysisConfig,
)
from src.utils.common import create_directories, read_yaml
from src.utils.exception import CustomException
from src.utils.mlflow_config import get_mlflow_uri


class ConfigurationManager:
    """
    Manages project configurations, parameters, and schema for the MLOps pipeline.

    This class centralizes the orchestration of configurations and parameters,
    integrating with DVC for data versioning. It transforms raw YAML inputs into
    strictly-typed Pydantic Configuration Entities, providing a robust and reproducible
    interface for all downstream pipeline components.

    Attributes:
        config (ConfigBox): Static configuration settings.
        params (ConfigBox): Pipeline parameters, integrated with DVC for versioning.
        schema (ConfigBox): Data schema and contract mapping.
    """

    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
        schema_filepath: Path = SCHEMA_FILE_PATH,
    ) -> None:
        """
        Initializes the ConfigurationManager.

        Args:
            config_filepath (Path): Path to the static configuration file.
            params_filepath (Path): Path to the parameters file (tracked by DVC).
            schema_filepath (Path): Path to the schema definition file.
        """
        try:
            self.config = read_yaml(config_filepath)
            self.schema = read_yaml(schema_filepath)

            # DVC API expects paths relative to the project root
            rel_params_path = os.path.relpath(params_filepath, PROJECT_ROOT)

            try:
                params_dict = dvc.api.params_show(rel_params_path)
                self.params = ConfigBox(params_dict)
            except Exception:
                # Fallback for production/Docker environments without .git/.dvc
                self.params = read_yaml(params_filepath)

            create_directories([self.config.artifacts_root])

        except Exception as e:
            raise CustomException(e, sys)

    def get_schema_config(self) -> SchemaConfig:
        """
        Creates the Schema configuration entity for data contracts.

        Returns:
            SchemaConfig: Configuration object with column mapping and data types.
        """
        try:
            # We convert ConfigBox to dict for Pydantic instantiation
            return SchemaConfig(**self.schema.to_dict())
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Creates the Data Ingestion configuration entity.

        Returns:
            DataIngestionConfig: Configuration for the data ingestion component.
        """
        try:
            config = self.config.data_ingestion
            create_directories([config.root_dir])

            # Use Pydantic for validation
            return DataIngestionConfig(**config.to_dict())

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Creates the Data Validation configuration entity.

        Returns:
            DataValidationConfig: Configuration object containing paths and parameters
            for the data validation stage.
        """
        try:
            config = self.config.data_validation
            params = self.params.data_validation

            create_directories([config.root_dir])

            data_validation_config = DataValidationConfig(
                root_dir=Path(config.root_dir),
                unzip_data_dir=Path(config.unzip_data_dir),
                STATUS_FILE=Path(config.STATUS_FILE),
                cleaned_data_file=Path(config.cleaned_data_file),
                min_desc_len=int(params.min_desc_len),
                categories_min_len=int(params.categories_min_len),
            )

            return data_validation_config
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Creates the Data Transformation configuration entity.

        Returns:
            DataTransformationConfig: Configuration object containing paths, split ratios,
            and parameters for the data transformation stage.
        """
        try:
            config = self.config.data_transformation
            params = self.params.data_transformation

            create_directories([config.root_dir])

            data_transformation_config = DataTransformationConfig(
                root_dir=Path(config.root_dir),
                data_path=Path(config.data_path),
                test_size=float(params.test_size),
                val_size=float(params.val_size),
                random_state=int(params.random_state),
            )

            return data_transformation_config
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_enrichment_config(self) -> DataEnrichmentConfig:
        """
        Creates the Data Enrichment (Zero-Shot) configuration entity.

        Returns:
            DataEnrichmentConfig: Configuration object containing paths and parameters
            for the data enrichment stage.
        """
        try:
            config = self.config.data_enrichment
            params = self.params.data_enrichment

            create_directories([config.root_dir])

            data_enrichment_config = DataEnrichmentConfig(
                root_dir=Path(config.root_dir),
                data_path=Path(config.data_path),
                enriched_data_path=Path(config.enriched_data_path),
                model_name=str(params.model_name),
                candidate_labels=list(params.candidate_labels),
                batch_size=int(params.batch_size),
            )

            return data_enrichment_config
        except Exception as e:
            raise CustomException(e, sys)

    def get_tone_analysis_config(self) -> ToneAnalysisConfig:
        """
        Creates the Tone Analysis (Sentiment) configuration entity.

        Returns:
            ToneAnalysisConfig: Configuration object containing paths and parameters
            for the tone analysis stage.
        """
        try:
            config = self.config.tone_analysis
            params = self.params.tone_analysis

            create_directories([config.root_dir])

            source_data = Path(config.data_path)
            if not source_data.exists():
                source_data = Path(self.config.data_validation.cleaned_data_file)

            tone_analysis_config = ToneAnalysisConfig(
                root_dir=Path(config.root_dir),
                data_path=source_data,
                output_path=Path(config.output_path),
                model_name=str(params.model_name),
                target_emotions=list(params.target_emotions),
                batch_size=int(params.batch_size),
                min_sentence_len=int(params.min_sentence_len),
                max_sentences_per_book=int(params.max_sentences_per_book),
                detection_threshold=float(params.detection_threshold),
            )

            return tone_analysis_config
        except Exception as e:
            raise CustomException(e, sys)

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Creates the Model Trainer (Vector DB) configuration entity.

        Returns:
            ModelTrainerConfig: Configuration object for vector database creation.
        """
        try:
            config = self.config.model_trainer
            params = self.params.model_trainer

            create_directories([config.root_dir])

            model_trainer_config = ModelTrainerConfig(
                root_dir=Path(config.root_dir),
                data_path=Path(config.data_path),
                db_path=Path(config.db_path),
                model_name=str(params.model_name),
                collection_name=str(params.collection_name),
                embedding_provider=str(params.embedding_provider),
                batch_size=int(params.batch_size),
            )

            return model_trainer_config
        except Exception as e:
            raise CustomException(e, sys)

    def get_inference_config(self) -> InferenceConfig:
        """
        Creates the Inference configuration entity.

        Returns:
            InferenceConfig: Configuration object containing paths and parameters
            for the inference stage.
        """
        try:
            model_trainer_config = self.config.model_trainer
            params_inference = self.params.inference

            data_path = PROJECT_ROOT / self.config.tone_analysis.output_path

            if not data_path.exists():
                data_path = PROJECT_ROOT / self.config.data_enrichment.enriched_data_path

            if not data_path.exists():
                data_path = PROJECT_ROOT / self.config.data_validation.cleaned_data_file

            inference_config = InferenceConfig(
                model_name=str(params_inference.model_name),
                embedding_provider=str(params_inference.embedding_provider),
                chroma_db_dir=PROJECT_ROOT / model_trainer_config.db_path,
                data_path=data_path,
                collection_name=str(params_inference.collection_name),
                top_k=int(params_inference.top_k),
                popularity_weight=float(params_inference.popularity_weight),
                search_buffer_multiplier=int(params_inference.search_buffer_multiplier),
                filtered_search_boost=int(params_inference.filtered_search_boost),
            )

            return inference_config
        except Exception as e:
            raise CustomException(e, sys)

    def get_batch_prediction_config(self) -> BatchPredictionConfig:
        """
        Creates the Batch Prediction configuration entity.

        Returns:
            BatchPredictionConfig: Configuration object for batch predictions.
        """
        try:
            config = self.config.prediction
            create_directories([config.root_dir])

            return BatchPredictionConfig(
                root_dir=Path(config.root_dir),
                results_file=Path(config.results_file),
            )

        except Exception as e:
            raise CustomException(e, sys)

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Creates the Model Evaluation configuration entity.

        Returns:
            ModelEvaluationConfig: Configuration object for model evaluation and MLflow tracking.
        """
        try:
            config = self.config.model_evaluation
            params = self.params.to_dict()

            create_directories([config.root_dir])

            model_evaluation_config = ModelEvaluationConfig(
                root_dir=Path(config.root_dir),
                data_path=Path(config.data_path),
                model_path=Path(config.model_path),
                all_params=params,
                mlflow_uri=get_mlflow_uri(),
                experiment_name=str(self.params.mlflow.experiment_name),
            )

            return model_evaluation_config
        except Exception as e:
            raise CustomException(e, sys)

    def get_agent_config(self) -> AgentConfig:
        """Creates the Agent configuration entity.

        Returns:
            AgentConfig: Configuration object for the agentic layer.
        """
        try:
            params = self.params.agent

            return AgentConfig(
                model_name=str(params.model_name),
                temperature=float(params.temperature),
                max_results_per_search=int(params.max_results_per_search),
            )
        except Exception as e:
            raise CustomException(e, sys)
