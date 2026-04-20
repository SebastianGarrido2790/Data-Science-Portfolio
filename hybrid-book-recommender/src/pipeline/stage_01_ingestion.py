"""
This module serves as the 'Conductor' for the Data Ingestion Stage of the pipeline.
It manages the transition from raw data configuration to local availability
by orchestrating the DataIngestion component.
"""

import sys

from src.components.data_ingestion import DataIngestion
from src.config.configuration import ConfigurationManager
from src.utils.exception import CustomException
from src.utils.logger import get_logger

STAGE_NAME = "Data Ingestion Stage"
logger = get_logger(headline=STAGE_NAME)


class DataIngestionTrainingPipeline:
    """
    Orchestration class for the Data Ingestion pipeline stage.

    Handles configuration loading, component initialization, and the execution
    of data acquisition steps (downloading and extracting).
    """

    def __init__(self):
        pass

    def main(self) -> None:
        """
        Main execution flow for the Data Ingestion stage.

        This method:
        1. Initializes the ConfigurationManager to retrieve ingestion settings.
        2. Instantiates the DataIngestion component.
        3. Executes the download and extraction processes.

        Raises:
            CustomException: If an error occurs during the ingestion process.
        """
        try:
            logger.info("🚀 Starting Ingestion Pipeline 🚀")

            # 1. Initialize Configuration
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()

            # 2. Initialize Component
            data_ingestion = DataIngestion(config=data_ingestion_config)

            # 3. Execute Actions
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()

            logger.info("✅ Ingestion Pipeline Completed ✅")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        obj = DataIngestionTrainingPipeline()
        obj.main()
    except Exception as e:
        raise CustomException(e, sys)
