"""
Pipeline stage for model training and benchmarking.

This stage loads the processed data, trains multiple candidate models,
compares them using MLflow, and saves the best model.
"""

import sys

from src.components.model_trainer import ModelTrainer
from src.config.configuration import ConfigurationManager
from src.utils.exception import CustomExceptionError
from src.utils.logger import get_logger

STAGE_NAME = "Model Training"
logger = get_logger(__name__, headline="Stage: Model Training")


class ModelTrainerPipeline:
    """
    Orchestrates the model training stage.
    """

    def __init__(self) -> None:
        """
        Initializes the pipeline stage.
        """
        self.config = ConfigurationManager()

    def main(self) -> None:
        """
        Executes the model training stage logic for the model trainer.
        """
        try:
            model_trainer_config = self.config.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.train_and_register()

        except Exception as e:
            raise CustomExceptionError(e, sys) from e


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<<")
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
