"""
This module acts as the entry point for the hybrid book recommender system.
It orchestrates the project's pipeline stages in order for debugging and testing to be easier.

Usage:
    uv run python main.py
"""

import sys

from src.pipeline.stage_01_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_02_validation import DataValidationTrainingPipeline
from src.pipeline.stage_03_enrichment import DataEnrichmentTrainingPipeline
from src.pipeline.stage_04_tone_analysis import ToneAnalysisTrainingPipeline
from src.pipeline.stage_05_transformation import DataTransformationTrainingPipeline
from src.pipeline.stage_06_training import ModelTrainingPipeline
from src.pipeline.stage_07_prediction import PredictionPipeline
from src.pipeline.stage_08_evaluation import ModelEvaluationPipeline
from src.utils.exception import CustomException
from src.utils.logger import get_logger, log_spacer

logger = get_logger(headline="main.py")


try:
    # 1. Data Ingestion
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    log_spacer()

    # 2. Data Validation
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    log_spacer()

    # 3. Data Enrichment
    data_enrichment = DataEnrichmentTrainingPipeline()
    data_enrichment.main()
    log_spacer()

    # 4. Tone Analysis
    tone_analysis = ToneAnalysisTrainingPipeline()
    tone_analysis.main()
    log_spacer()

    # 5. Data Transformation
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    log_spacer()

    # 6. Model Training
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    log_spacer()

    # 7. Prediction
    # This acts as a smoke test locally
    prediction = PredictionPipeline()
    prediction.main()
    log_spacer()

    # 8. Model Evaluation
    model_evaluation = ModelEvaluationPipeline()
    model_evaluation.main()

except Exception as e:
    raise CustomException(e, sys)
