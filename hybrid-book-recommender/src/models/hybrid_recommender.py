"""
This module implements a hybrid recommendation system that combines semantic search with collaborative filtering.
It uses a vector database (ChromaDB) for semantic search and a collaborative filtering proxy (ratings) to boost books with higher average ratings.
"""

import sys

import pandas as pd
from langchain_chroma import Chroma

from src.constants import PROJECT_ROOT
from src.entity.config_entity import InferenceConfig, SchemaConfig
from src.entity.recommendation_entity import RecommendationResult
from src.models.llm_utils import EmbeddingFactory
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRecommender:
    """
    The 'Brain' of the recommendation system.

    It combines:
    1. Semantic Search (ChromaDB): Finds books with similar content/topics.
    2. Collaborative Filtering Proxy (Ratings): Boosts books with higher average ratings.
    """

    def __init__(self, config: InferenceConfig, schema: SchemaConfig):
        """
        Initializes the HybridRecommender components, including the embedding function,
        vector store (ChromaDB), and book metadata.

        Args:
            config (InferenceConfig): The inference configuration entity containing
                embedding provider details, database paths, and data source locations.
            schema (SchemaConfig): Data contract mapping logical -> physical column names.

        Raises:
            CustomException: If any component fails to initialize or data cannot be loaded.
        """
        try:
            self.config = config
            self.schema = schema

            self.embedding_fn = EmbeddingFactory.get_embedding_function(
                provider=self.config.embedding_provider,
                model_name=self.config.model_name,
            )

            self.vector_store = Chroma(
                persist_directory=str(self.config.chroma_db_dir),
                embedding_function=self.embedding_fn,
                collection_name=self.config.collection_name,
            )

            # We read 'clean_books.csv' specifically to get ALL ratings, not just the training set.
            self.books_metadata = pd.read_csv(self.config.data_path)

            cols = self.schema.columns
            # Ensure index is set correctly using schema mapping
            self.books_metadata.set_index(cols["isbn"], inplace=True)

            logger.info(
                f"Hybrid Recommender initialized. DB: {self.config.chroma_db_dir.relative_to(PROJECT_ROOT)}"
            )
        except Exception as e:
            raise CustomException(e, sys)

    def recommend(
        self, query: str, category_filter: str | None = None, tone_filter: str | None = None
    ) -> list[RecommendationResult]:
        """
        Retrieves and ranks book recommendations based on a hybrid score.

        Args:
            query (str): The natural language search query (e.g., "Space opera with aliens").
            category_filter (str, optional): A category to filter results by (e.g., "Non-Fiction").
            tone_filter (str, optional): A tone to filter results by (e.g., "joy", "fear").

        Returns:
            List[RecommendationResult]: A list of RecommendationResult objects, each containing book metadata
                                   and the calculated 'score'.
        """
        top_k = self.config.top_k
        popularity_weight = self.config.popularity_weight

        try:
            logger.info(
                f"Processing query: '{query}' (Category: {category_filter}, Tone: {tone_filter})"
            )

            # 1. Semantic Search (Fetch more to allow for filtering)
            # Increase fetch buffer if filters are applied to find matches in the long tail.
            # Since ChromaDB metadata might not match the specific filters (categories/tone), we rely on post-filtering.
            # We need a large window to ensure we find enough candidates that match both semantic relevance and the hard filters.
            # We use the configured search_buffer_multiplier.
            multiplier = self.config.search_buffer_multiplier
            if category_filter or tone_filter:
                multiplier *= (
                    self.config.filtered_search_boost
                )  # Additional boost for filtered queries

            fetch_k = top_k * multiplier
            logger.info(f"Fetching {fetch_k} candidates from VectorDB for filtering...")

            results = self.vector_store.similarity_search_with_score(query, k=fetch_k)

        except Exception as e:
            # Critical failure in vector search (e.g., DB disconnected)
            raise CustomException(e, sys)

        recommendations = []
        cols = self.schema.columns
        enriched_cols = self.schema.enriched_columns

        for doc, score in results:
            try:
                # --- ISBN CASTING ---
                # logical key in vector DB metadata was 'isbn'
                isbn_str = doc.metadata.get("isbn", "0")
                isbn_clean = "".join(filter(str.isdigit, str(isbn_str)))
                isbn = int(isbn_clean) if isbn_clean else 0

                # Check if we have rating data for this book
                if isbn in self.books_metadata.index:
                    book_row = self.books_metadata.loc[isbn]
                    if isinstance(book_row, pd.DataFrame):
                        book_row = book_row.iloc[0]

                    # --- CATEGORICAL FILTERING ---
                    # We check for broad category first, then fallback to original
                    category = book_row.get(enriched_cols["simple_category"])
                    if category is None or pd.isna(category):
                        category = book_row.get(cols["categories"], "Uncategorized")

                    if category_filter and str(category_filter).lower() != str(category).lower():
                        continue

                    # --- TONE FILTERING ---
                    tone = book_row.get(
                        enriched_cols["dominant_tone"], "neutral"
                    )  # default to neutral if missing

                    if tone_filter and str(tone_filter).lower() != str(tone).lower():
                        continue

                    # --- TONE PROBABILITIES ---
                    # If a tone filter is applied, we capture its specific probability for sorting.
                    tone_prob = 0.0
                    if tone_filter and tone_filter in book_row:
                        tone_prob = float(book_row.get(tone_filter, 0.0))

                    rating_val = book_row.get(cols["rating"], 0.0)
                    rating = 0.0 if pd.isna(rating_val) else float(rating_val)

                    # Safely conversion for ratings_count
                    raw_count = book_row.get(cols["ratings_count"], 0)
                    try:
                        ratings_count = int(raw_count)
                    except (ValueError, TypeError):
                        ratings_count = 0

                    # 2. Hybrid Scoring
                    similarity_score = 1 - score
                    # rating / 5.0 as ratings are usually 0-5
                    hybrid_score = similarity_score + ((rating / 5.0) * popularity_weight)

                    recommendations.append(
                        RecommendationResult(
                            isbn=isbn,
                            title=doc.metadata.get("title", ""),
                            authors=doc.metadata.get("authors", ""),
                            description=doc.metadata.get("description", ""),
                            category=str(category),
                            tone=str(tone),
                            tone_prob=tone_prob,
                            rating=rating,
                            ratings_count=ratings_count,
                            thumbnail=book_row.get(cols["thumbnail"]),
                            score=hybrid_score,
                            match_reason="Hybrid Match",
                        )
                    )
            except Exception as e:
                logger.warning(f"Skipping book due to error: {e}")
                continue

        # 3. Sort logic
        if tone_filter:
            recommendations = sorted(recommendations, key=lambda x: x.tone_prob, reverse=True)
        else:
            recommendations = sorted(recommendations, key=lambda x: x.score, reverse=True)
        return recommendations[:top_k]
