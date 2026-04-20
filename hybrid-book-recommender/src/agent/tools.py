"""
Deterministic tools for the Book Recommender Agent.

These are the 'Brawn' — pure deterministic functions that the Agent 'Brain' calls
to interact with the HybridRecommender engine (Brain vs. Brawn).

Each tool has a rich docstring so the LLM understands its capabilities.
Input validation is handled via typed parameters with Pydantic validation.
"""

from dataclasses import dataclass

from pydantic_ai import RunContext

from src.agent.schemas import BookRecommendation
from src.models.hybrid_recommender import HybridRecommender
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentDependencies:
    """Runtime dependencies injected into the agent at execution time.

    Attributes:
        recommender: The initialized HybridRecommender engine.
        categories: Available book categories from params.yaml.
        tones: Available emotional tones from params.yaml.
        tone_map: Mapping of display tone names to internal tone labels.
        max_results: Maximum number of results to return per search.
    """

    recommender: HybridRecommender
    categories: list[str]
    tones: list[str]
    tone_map: dict[str, str]
    max_results: int
    model_name: str  # Dynamically loaded from params.yaml


def search_books(
    ctx: RunContext[AgentDependencies],
    query: str,
    category: str | None = None,
    tone: str | None = None,
) -> list[BookRecommendation]:
    """Search the book database using natural language with optional filters.

    Use this tool to find books matching a user's description. The query should
    capture the themes, topics, or style the user is looking for.

    Args:
        ctx: The runtime context containing the recommender engine.
        query: Natural language description of the desired book
            (e.g., "A dark mystery set in Victorian London").
        category: Optional genre filter. Must be one of the available categories
            (use `get_available_categories` to see options). Pass None for no filter.
        tone: Optional emotional tone filter using the DISPLAY name
            (e.g., "Happy", "Sad", "Suspenseful"). Pass None for no filter.

    Returns:
        A list of BookRecommendation objects with title, authors, description,
        rating, mood_score, and category.
    """
    deps = ctx.deps
    recommender = deps.recommender

    # Map display tone name to internal label (e.g., "Happy" -> "joy")
    tone_filter: str | None = None
    if tone:
        tone_filter = deps.tone_map.get(tone)

    cat_filter: str | None = None
    if category and category != "All":
        cat_filter = category

    logger.info(
        f"Agent tool call: search_books(query='{query}', "
        f"category='{cat_filter}', tone='{tone_filter}')"
    )

    results = recommender.recommend(
        query=query, category_filter=cat_filter, tone_filter=tone_filter
    )

    results = results[: deps.max_results]

    recommendations: list[BookRecommendation] = []
    for book in results:
        try:
            mood_score = f"{float(book.tone_prob):.2f}"
        except (ValueError, TypeError):
            mood_score = "N/A"

        recommendations.append(
            BookRecommendation(
                title=book.title,
                authors=book.authors,
                description=book.description if book.description else "No description available.",
                rating=book.rating,
                mood_score=mood_score,
                category=book.category,
            )
        )

    logger.info(f"Agent tool returned {len(recommendations)} recommendations")
    return recommendations


def get_available_categories(ctx: RunContext[AgentDependencies]) -> list[str]:
    """Get the list of available book categories for filtering.

    Use this tool to discover which categories can be passed to the `search_books`
    tool's `category` parameter.

    Returns:
        A list of category names (e.g., ["All", "Fiction", "Non-Fiction", "Thriller", ...]).
    """
    return ctx.deps.categories


def get_available_tones(ctx: RunContext[AgentDependencies]) -> list[str]:
    """Get the list of available emotional tones for filtering.

    Use this tool to discover which tones can be passed to the `search_books`
    tool's `tone` parameter.

    Returns:
        A list of tone display names (e.g., ["All", "Happy", "Sad", "Suspenseful", ...]).
    """
    return ctx.deps.tones
