"""
Module for executing the offline data enrichment pipeline.

This script initializes the configuration manager, retrieves the data enrichment
configuration, and executes the enrichment process using the DataEnrichment component.

Usage:
    uv run python -m src.scripts.run_enrichment
"""

import sys

from src.components.data_enrichment import DataEnrichment
from src.config.configuration import ConfigurationManager
from src.utils.exception import CustomException
from src.utils.logger import get_logger

STAGE_NAME = "Data Enrichment Script"
logger = get_logger(headline=STAGE_NAME)


def main() -> None:
    """
    Main execution flow for the offline data enrichment script.

    Flow:
    1. Initialize ConfigurationManager.
    2. Retrieve DataEnrichmentConfig.
    3. Initialize DataEnrichment component.
    4. Execute initiate_data_enrichment().

    Raises:
        CustomException: If enrichment fails.
    """
    try:
        logger.info("🚀 Starting Offline Data Enrichment Process 🚀")

        config_manager = ConfigurationManager()
        enrichment_config = config_manager.get_data_enrichment_config()
        schema_config = config_manager.get_schema_config()

        enricher = DataEnrichment(config=enrichment_config, schema=schema_config)
        enricher.initiate_data_enrichment()

        logger.info("✅ Enrichment Process Completed Successfully ✅")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
