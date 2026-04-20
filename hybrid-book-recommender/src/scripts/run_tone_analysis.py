"""
Module for executing the offline tone analysis pipeline.

This script initializes the configuration manager, retrieves the tone analysis
configuration, and executes the tone analysis process using the ToneAnalysis component.

Usage:
    uv run python -m src.scripts.run_tone_analysis
"""

import sys

from src.components.tone_analysis import ToneAnalysis
from src.config.configuration import ConfigurationManager
from src.utils.exception import CustomException
from src.utils.logger import get_logger

STAGE_NAME = "Tone Analysis Script"
logger = get_logger(headline=STAGE_NAME)


def main() -> None:
    """
    Main execution flow for the offline tone analysis script.

    Flow:
    1. Initialize ConfigurationManager.
    2. Retrieve ToneAnalysisConfig.
    3. Initialize ToneAnalysis component.
    4. Execute initiate_tone_analysis().

    Raises:
        CustomException: If tone analysis fails.
    """
    try:
        logger.info("🚀 Starting Offline Tone Analysis Process 🚀")

        config_manager = ConfigurationManager()
        tone_config = config_manager.get_tone_analysis_config()
        schema_config = config_manager.get_schema_config()

        analyzer = ToneAnalysis(config=tone_config, schema=schema_config)
        analyzer.initiate_tone_analysis()

        logger.info("✅ Tone Analysis Completed Successfully ✅")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
