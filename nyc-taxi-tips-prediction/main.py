"""
Main entry point for the NYC Taxi Tips Prediction pipeline.

This script allows for manual orchestration of the pipeline stages for debugging
and development purposes.

Usage:
    uv run python main.py
"""

import sys

from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.pipeline.stage_04_feature_engineering import FeatureEngineeringTrainingPipeline
from src.pipeline.stage_05_model_trainer import ModelTrainerPipeline
from src.pipeline.stage_06_model_evaluation import ModelEvaluationPipeline
from src.utils.exception import CustomExceptionError
from src.utils.logger import get_logger, log_spacer

logger = get_logger(__name__, headline="main.py")

try:
    STAGE_NAME = "Data Ingestion"
    logger.info(f"Stage: {STAGE_NAME} started")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f"Stage: {STAGE_NAME} completed")
    log_spacer()

    STAGE_NAME = "Data Validation"
    logger.info(f"Stage: {STAGE_NAME} started")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f"Stage: {STAGE_NAME} completed")
    log_spacer()

    STAGE_NAME = "Data Transformation"
    logger.info(f"Stage: {STAGE_NAME} started")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f"Stage: {STAGE_NAME} completed")
    log_spacer()

    STAGE_NAME = "Feature Engineering"
    logger.info(f"Stage: {STAGE_NAME} started")
    feature_engineering = FeatureEngineeringTrainingPipeline()
    feature_engineering.main()
    logger.info(f"Stage: {STAGE_NAME} completed")
    log_spacer()

    STAGE_NAME = "Model Training"
    logger.info(f"Stage: {STAGE_NAME} started")
    model_trainer = ModelTrainerPipeline()
    model_trainer.main()
    logger.info(f"Stage: {STAGE_NAME} completed")
    log_spacer()

    STAGE_NAME = "Model Evaluation"
    logger.info(f"Stage: {STAGE_NAME} started")
    model_evaluation = ModelEvaluationPipeline()
    model_evaluation.main()
    logger.info(f"Stage: {STAGE_NAME} completed")

except Exception as e:
    raise CustomExceptionError(e, sys) from e
