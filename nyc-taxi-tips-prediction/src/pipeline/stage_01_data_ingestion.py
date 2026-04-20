"""
Data Ingestion Pipeline Stage (Conductor) for the NYC Taxi Tips Prediction project.

This module orchestrates the Data Ingestion stage. It handles the configuration
management and calls the technical components to perform the work.
"""

import sys

from src.components.data_ingestion import DataIngestion
from src.config.configuration import ConfigurationManager
from src.utils.exception import CustomExceptionError
from src.utils.logger import get_logger

STAGE_NAME = "Data Ingestion"
logger = get_logger(__name__, headline="Stage: Data Ingestion")


class DataIngestionTrainingPipeline:
    """
    Orchestrates the Data Ingestion process.
    """

    def __init__(self) -> None:
        """
        Initializes the pipeline stage.
        """
        self.config = ConfigurationManager()

    def main(self) -> None:
        """
        Executes the main orchestration logic for the ingestion stage.
        """
        try:
            data_ingestion_config = self.config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.initiate_data_ingestion()

        except Exception as e:
            raise CustomExceptionError(e, sys) from e


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
