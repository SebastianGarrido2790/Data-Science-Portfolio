"""
This module serves as the 'Conductor' for the Data Transformation Stage of the pipeline.
It orchestrates the execution of data cleaning and splitting by managing configurations
and triggering the DataTransformation component.
"""

import sys

from src.components.data_transformation import DataTransformation
from src.config.configuration import ConfigurationManager
from src.utils.exception import CustomException
from src.utils.logger import get_logger

STAGE_NAME = "Data Transformation Stage"
logger = get_logger(headline=STAGE_NAME)


class DataTransformationTrainingPipeline:
    """
    Orchestration class for the Data Transformation pipeline stage.

    Handles the sequence of operations: loading configuration, initializing the
    transformation component, and executing the transformation logic.
    """

    def __init__(self):
        pass

    def main(self) -> None:
        """
        Main execution flow for the Data Transformation stage.

        Flow:
        1. Load Config.
        2. Initialize DataTransformation Component.
        3. Trigger initiate_data_transformation logic.

        Raises:
            CustomException: If execution fails.
        """
        try:
            logger.info("🚀 Starting Data Transformation Pipeline 🚀")

            # 1. Load Config
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()

            # 2. Init Component
            data_transformation = DataTransformation(config=data_transformation_config)

            # 3. Run Logic
            data_transformation.initiate_data_transformation()

            logger.info("✅ Data Transformation Pipeline Completed ✅")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        obj = DataTransformationTrainingPipeline()
        obj.main()
    except Exception as e:
        raise CustomException(e, sys)
