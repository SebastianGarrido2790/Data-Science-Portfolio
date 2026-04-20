"""
This module contains the BatchPrediction component, which runs inference on a set of queries
and saves the results to a file, satisfying the 'heavy lifting' requirement.
"""

import sys

from src.entity.config_entity import BatchPredictionConfig, InferenceConfig, SchemaConfig
from src.models.hybrid_recommender import HybridRecommender
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BatchPrediction:
    """
    Component responsible for running batch predictions using the HybridRecommender.
    """

    def __init__(
        self,
        batch_config: BatchPredictionConfig,
        inference_config: InferenceConfig,
        schema: SchemaConfig,
    ):
        """
        Initializes the BatchPrediction component.

        Args:
            batch_config (BatchPredictionConfig): Configuration for batch output.
            inference_config (InferenceConfig): Configuration for the recommender model.
            schema (SchemaConfig): Data contract mapping logical -> physical column names.
        """
        try:
            self.batch_config = batch_config
            self.recommender = HybridRecommender(config=inference_config, schema=schema)
        except Exception as e:
            raise CustomException(e, sys)

    def run_batch_predictions(self, queries: list) -> None:
        """
        Runs predictions for the provided queries and saves results to the configured file.

        Args:
            queries (list): List of natural language queries to test.
        """
        try:
            output_file = self.batch_config.results_file

            # Ensure parent dir exists (though create_directories usually handles this)
            # output_file.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Starting batch prediction for {len(queries)} queries.")
            logger.info(f"Saving results to: {output_file}")

            with open(output_file, "w", encoding="utf-8") as f:
                for query in queries:
                    logger.info(f"🔍 Testing Query: {query}")
                    logger.info("-" * 60)
                    f.write(f"\n🔍 Query: {query}\n")
                    f.write("-" * 60 + "\n")

                    try:
                        results = self.recommender.recommend(query)
                    except Exception as e:
                        logger.error(f"Failed to get recommendations for query '{query}': {e}")
                        f.write(f"Error: {e}\n")
                        continue

                    for i, book in enumerate(results):
                        result_str = (
                            f"{i + 1}. {book.title} | "
                            f"{book.category} "
                            f"(Rating: {book.rating}, Score: {book.score:.3f})"
                        )

                        logger.info(result_str)
                        logger.info(f"   Author: {book.authors}")

                        f.write(result_str + "\n")
                        f.write(f"   Author: {book.authors}\n")

                        desc = book.description
                        desc_preview = (
                            desc[:100].replace("\n", " ") + "..." if desc else "No description"
                        )
                        logger.info(f"   Desc: {desc_preview}")
                        f.write(f"   Desc: {desc_preview}\n")

            logger.info(f"Batch prediction for {len(queries)} queries completed successfully.")

        except Exception as e:
            raise CustomException(e, sys)
