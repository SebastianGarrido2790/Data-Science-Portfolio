"""
This module serves as the 'Worker' for the Model Training Stage.
It handles the generation of Vector Embeddings using a configurable provider (HuggingFace or Gemini) and indexing them into ChromaDB.
"""

import os
import shutil
import sys
import time
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

from src.entity.config_entity import ModelTrainerConfig, SchemaConfig
from src.models.llm_utils import EmbeddingFactory
from src.utils.exception import CustomException
from src.utils.logger import get_logger

# Load environment variables (API Keys)
load_dotenv()

logger = get_logger(__name__)


class ModelTrainer:
    """
    This class handles the creation of the semantic search engine.
    It reads the cleaned dataset, generates embeddings via a configurable provider (HuggingFace or Gemini),
    and persists them into a ChromaDB vector store.

    Attributes:
        config (ModelTrainerConfig): Configuration entity containing data paths,
            model name, and batch size for the enrichment process.
        schema (SchemaConfig): Data contract mapping logical -> physical column names.
    """

    def __init__(self, config: ModelTrainerConfig, schema: SchemaConfig):
        self.config = config
        self.schema = schema

    def get_embedding_function(self) -> Any:
        """
        Factory method to get the embedding function based on configuration.
        """
        return EmbeddingFactory.get_embedding_function(
            provider=self.config.embedding_provider, model_name=self.config.model_name
        )

    def initiate_model_training(self) -> None:
        """
        Executes the embedding generation and indexing process.

        Steps:
        1. Loads the clean dataset.
        2. Converts rows into LangChain 'Document' objects with rich context.
        3. Initializes the Embedding model (Provider agnostic).
        4. Resets and repopulates the ChromaDB collection.

        Raises:
            CustomException: If any error occurs during the training process.
        """
        try:
            logger.info("--- Loading Data ---")
            if not os.path.exists(self.config.data_path):
                raise FileNotFoundError(f"Data not found at {self.config.data_path}")

            df = pd.read_csv(self.config.data_path)
            # Fill NaN values to prevent embedding errors
            df = df.fillna("")

            # Standardize columns to lower case for consistent access
            df.columns = df.columns.str.lower().str.strip()

            logger.info(f"Loaded {len(df)} books for indexing.")

            # Prepare Documents for LangChain
            documents = []
            cols = self.schema.columns
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing Context Documents"):
                # Safety checks for columns - using schema mapping
                title = row.get(cols["title"], "Unknown Title")
                authors = row.get(cols["authors"], "Unknown Author")
                desc = row.get(cols["description"], "")
                cats = row.get(cols["categories"], "Uncategorized")

                # Content: The text used for semantic similarity matching
                content = (
                    f"Title: {title}\nAuthor: {authors}\nDescription: {desc}\nCategories: {cats}"
                )

                # Metadata: The data returned when a match is found
                metadata = {
                    "isbn": str(
                        row.get(cols["isbn"], "")
                    ),  # Map physical column to logical key (unique identifier)
                    "title": str(title),
                    "authors": str(authors),
                    "categories": str(cats),
                    "description": str(desc)[:500],
                }
                documents.append(Document(page_content=content, metadata=metadata))

            # Initialize Embedding Model
            embedding_fn = self.get_embedding_function()

            # Define Persistence Path and Collection
            persist_dir = str(self.config.db_path)
            collection_name = self.config.collection_name

            # Reset VectorDB for a fresh build
            # Since this is a training pipeline, we usually want to start fresh to avoid duplicates or stale data.
            if os.path.exists(persist_dir):
                logger.warning(f"Removing existing VectorDB at {persist_dir} for fresh index.")
                shutil.rmtree(persist_dir)

            logger.info(f"Creating VectorDB at {persist_dir}...")

            # Initialize Chroma with auto-persistence
            vector_store = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding_fn,
                collection_name=collection_name,
            )

            # Batch add documents to manage memory and rate limits
            batch_size = self.config.batch_size
            logger.info(f"Indexing {len(documents)} documents in batches of {batch_size}...")

            for i in tqdm(range(0, len(documents), batch_size), desc="Indexing Batches"):
                batch = documents[i : i + batch_size]

                # Robust retry logic for API Rate Limits (429)
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        vector_store.add_documents(batch)
                        break  # Success, move to next batch
                    except Exception as e:
                        error_msg = str(e)
                        if (
                            "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg
                        ) and attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 15  # Incremental wait: 15s, 30s, 45s...
                            logger.warning(
                                f"⚠️ Rate limit hit (429). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}..."
                            )
                            time.sleep(wait_time)
                        else:
                            logger.error(f"❌ Permanent error during indexing batch {i}: {e}")
                            raise e

                # Mandatory pause for Gemini Free Tier to stabilize TPM (Tokens Per Minute)
                if self.config.embedding_provider.lower() == "gemini":
                    time.sleep(2)

            logger.info(f"✅ Successfully indexed {len(documents)} books into VectorDB.")

        except Exception as e:
            raise CustomException(e, sys)
