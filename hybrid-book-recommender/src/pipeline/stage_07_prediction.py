"""
This script is used to perform predictions using the HybridRecommender model.
It loads the model from the specified configuration and runs a set of test queries.
Results are saved to a text file in the artifacts/prediction directory.
"""

import sys

from src.components.batch_prediction import BatchPrediction
from src.config.configuration import ConfigurationManager
from src.utils.exception import CustomException
from src.utils.logger import get_logger

STAGE_NAME = "Prediction Pipeline"
logger = get_logger(headline=STAGE_NAME)


class PredictionPipeline:
    """
    Encapsulates the logic for the prediction stage of the ML pipeline.

    This stage is responsible for loading the trained HybridRecommender model
    and running inference on a set of test queries to validate the system's
    recommendation capabilities.
    """

    def __init__(self):
        pass

    def main(self) -> None:
        """
        Main execution flow for the Prediction Pipeline.

        Loads the HybridRecommender and runs a set of predefined test queries
        to verify model performance locally.

        Raises:
            CustomException: If prediction flow fails.
        """
        try:
            logger.info("🚀 Starting Prediction Pipeline 🚀")

            # 1. Load Configuration
            config_manager = ConfigurationManager()
            inference_config = config_manager.get_inference_config()
            batch_config = config_manager.get_batch_prediction_config()
            schema_config = config_manager.get_schema_config()

            # 2. Init Component
            batch_predictor = BatchPrediction(
                batch_config=batch_config,
                inference_config=inference_config,
                schema=schema_config,
            )

            # Test cases to verify different aspects (semantic vs hybrid)
            queries = [
                "A cyberpunk novel about artificial intelligence",
                "A historical fiction about Roman Empire",
                "A book about machine learning",
            ]

            # 3. Run Batch Predictions
            batch_predictor.run_batch_predictions(queries)

            logger.info("✅ Prediction Pipeline Completed ✅")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        obj = PredictionPipeline()
        obj.main()
    except Exception as e:
        raise CustomException(e, sys)
