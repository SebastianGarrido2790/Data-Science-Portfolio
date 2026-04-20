"""
This module serves as the 'Conductor' for the Data Enrichment Stage of the pipeline.
It orchestrates the DataEnrichment component using zero-shot classification.
"""

import sys

from src.components.data_enrichment import DataEnrichment
from src.config.configuration import ConfigurationManager
from src.utils.exception import CustomException
from src.utils.logger import get_logger

STAGE_NAME = "Data Enrichment Stage"
logger = get_logger(headline=STAGE_NAME)


class DataEnrichmentTrainingPipeline:
    """
    Coordinates the execution of the DataEnrichment component.
    """

    def __init__(self):
        pass

    def main(self) -> None:
        """
        Executes the Data Enrichment pipeline.
        """
        try:
            logger.info("🚀 Starting Enrichment Pipeline 🚀")
            config = ConfigurationManager()
            enrichment_config = config.get_data_enrichment_config()
            schema_config = config.get_schema_config()

            enrichment = DataEnrichment(config=enrichment_config, schema=schema_config)
            enrichment.initiate_data_enrichment()

            logger.info("✅ Enrichment Pipeline Completed ✅")
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        obj = DataEnrichmentTrainingPipeline()
        obj.main()
    except Exception as e:
        raise CustomException(e, sys)
