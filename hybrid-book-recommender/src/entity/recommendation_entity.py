"""
Data classes for recommendation results.

This prevents "stringly-typed" errors in the UI layer and ensures deep code intelligence
(autocomplete/type checking) throughout the application.
"""

from dataclasses import dataclass


@dataclass
class RecommendationResult:
    """
    Represents a single book recommendation with its metadata and score.

    Attributes:
        isbn (int): The ISBN of the book.
        title (str): The title of the book.
        authors (str): The authors of the book.
        description (str): The description of the book.
        category (str): The category of the book.
        tone (str): The tone of the book.
        tone_prob (float): The probability of the tone.
        rating (float): The rating of the book.
        ratings_count (int): The number of ratings for the book.
        thumbnail (str | None): The thumbnail of the book.
        score (float): The score of the book.
        match_reason (str): The reason for the match.
    """

    isbn: int
    title: str
    authors: str
    description: str
    category: str
    tone: str
    tone_prob: float
    rating: float
    ratings_count: int
    thumbnail: str | None
    score: float
    match_reason: str
