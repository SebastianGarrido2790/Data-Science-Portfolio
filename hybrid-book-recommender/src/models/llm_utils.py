"""
This module implements a factory pattern for generating embedding functions based on configuration.
It ensures that both the Trainer (creating the DB) and the Recommender (querying the DB) define the embedding model exactly the same way, preventing a common "Training-Serving Skew" issue.
"""

import os
import sys
from typing import Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingFactory:
    """
    Factory class to generate embedding functions based on configuration.
    Ensures consistent embedding models are used across Training and Inference.
    """

    @staticmethod
    def get_embedding_function(provider: str, model_name: str) -> Any:
        """
        Returns the LangChain embedding function based on the provider.

        Args:
            provider (str): 'huggingface' or 'gemini'
            model_name (str): The specific model ID (e.g., 'sentence-transformers/all-MiniLM-L6-v2')

        Returns:
            Embeddings: A LangChain Embeddings interface.
        """
        try:
            provider = provider.lower()

            if provider == "huggingface":
                logger.info(f"Using HuggingFace Embeddings: {model_name}")
                return HuggingFaceEmbeddings(model_name=model_name)
            elif provider == "gemini":
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError(
                        "GOOGLE_API_KEY not found in environment variables. Please check your .env file."
                    )
                logger.info(f"Using Google Gemini Embeddings: {model_name}")
                return GoogleGenerativeAIEmbeddings(model=model_name)
            else:
                raise ValueError(
                    f"Unsupported embedding provider: {provider}. Options: 'huggingface', 'gemini'."
                )
        except Exception as e:
            raise CustomException(e, sys)
