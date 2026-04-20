"""
This module serves as the 'Conductor' for the Tone Analysis Stage of the pipeline.
It orchestrates the ToneAnalysis component using sentiment analysis.
"""

import sys

from src.components.tone_analysis import ToneAnalysis
from src.config.configuration import ConfigurationManager
from src.utils.exception import CustomException
from src.utils.logger import get_logger

STAGE_NAME = "Tone Analysis Stage"
logger = get_logger(headline=STAGE_NAME)


class ToneAnalysisTrainingPipeline:
    """
    Coordinates the execution of the ToneAnalysis component.
    """

    def __init__(self):
        pass

    def main(self) -> None:
        """
        Executes the Tone Analysis pipeline.
        """
        try:
            logger.info("🚀 Starting Tone Analysis Pipeline 🚀")
            config = ConfigurationManager()
            tone_config = config.get_tone_analysis_config()
            schema_config = config.get_schema_config()

            tone_analysis = ToneAnalysis(config=tone_config, schema=schema_config)
            tone_analysis.initiate_tone_analysis()

            logger.info("✅ Tone Analysis Pipeline Completed ✅")
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        obj = ToneAnalysisTrainingPipeline()
        obj.main()
    except Exception as e:
        raise CustomException(e, sys)
