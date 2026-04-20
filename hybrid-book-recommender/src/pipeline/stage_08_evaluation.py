"""
This module serves as the 'Conductor' for the Model Evaluation Stage.
It orchestrates the execution of MLflow tracking and metric logging.
"""

import sys

from src.components.model_evaluation import ModelEvaluation
from src.config.configuration import ConfigurationManager
from src.utils.exception import CustomException
from src.utils.logger import get_logger

STAGE_NAME = "Model Evaluation Stage"
logger = get_logger(headline=STAGE_NAME)


class ModelEvaluationPipeline:
    """
    Orchestration class for the Model Evaluation pipeline stage.
    """

    def __init__(self):
        pass

    def main(self) -> None:
        """
        Main execution flow for the Model Evaluation stage.

        Flow:
        1. Load Evaluation Config.
        2. Initialize ModelEvaluation Component.
        3. Log parameters and metrics to MLflow.

        Raises:
            CustomException: If evaluation fails.
        """
        try:
            logger.info("🚀 Starting Model Evaluation Pipeline 🚀")

            # 1. Load Config
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()

            # 2. Init Component
            model_evaluation = ModelEvaluation(config=model_evaluation_config)

            # 3. Run Logic
            model_evaluation.log_into_mlflow()

            logger.info("✅ Model Evaluation Pipeline Completed ✅")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        obj = ModelEvaluationPipeline()
        obj.main()
    except Exception as e:
        raise CustomException(e, sys)
