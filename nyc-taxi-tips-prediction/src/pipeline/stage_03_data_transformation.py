"""
Data Transformation Pipeline Stage (Conductor).

This module orchestrates the Data Transformation process, ensuring that raw data is
cleaned and split according to business rules before feature engineering.
"""

import sys

from src.components.data_transformation import DataTransformation
from src.config.configuration import ConfigurationManager
from src.utils.exception import CustomExceptionError
from src.utils.logger import get_logger

STAGE_NAME = "Data Transformation"
logger = get_logger(__name__, headline="Stage: Data Transformation")


class DataTransformationTrainingPipeline:
    """
    Orchestrates the data transformation stage.
    """

    def __init__(self) -> None:
        """
        Initializes the pipeline stage.
        """
        self.config = ConfigurationManager()

    def main(self) -> None:
        """
        Executes the main orchestration logic for the transformation stage.
        """
        try:
            data_transformation_config = self.config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.initiate_data_transformation()

        except Exception as e:
            raise CustomExceptionError(e, sys) from e


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
