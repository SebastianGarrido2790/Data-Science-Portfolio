"""
Feature Engineering Pipeline Stage (Conductor).

This module orchestrates the Feature Engineering process, including cyclical feature
creation and temporal splitting into Train, Validation, and Test sets.
"""

import sys

from src.components.feature_engineering import FeatureEngineering
from src.config.configuration import ConfigurationManager
from src.utils.exception import CustomExceptionError
from src.utils.logger import get_logger

STAGE_NAME = "Feature Engineering"
logger = get_logger(__name__, headline="Stage: Feature Engineering")


class FeatureEngineeringTrainingPipeline:
    """
    Orchestrates the feature engineering stage.
    """

    def __init__(self) -> None:
        """
        Initializes the pipeline stage.
        """
        self.config = ConfigurationManager()

    def main(self) -> None:
        """
        Executes the main orchestration logic for the feature engineering stage.
        """
        try:
            feature_engineering_config = self.config.get_feature_engineering_config()
            feature_engineering = FeatureEngineering(config=feature_engineering_config)
            feature_engineering.initiate_feature_engineering()

        except Exception as e:
            raise CustomExceptionError(e, sys) from e


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<<")
        obj = FeatureEngineeringTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
