"""
Unit tests for the Agentic Layer.

This module validates that the agent's tools and schemas work correctly
with mocked dependencies — no API keys or LLM calls required.

Usage:
    uv run pytest tests/unit/test_agent.py -v
"""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.agent.schemas import AgentResponse, BookRecommendation
from src.agent.tools import AgentDependencies, get_available_categories, get_available_tones

# --- Fixtures ---


@pytest.fixture
def mock_deps() -> AgentDependencies:
    """Provides mock agent dependencies."""
    return AgentDependencies(
        model_name="test-model",
        recommender=MagicMock(),
        categories=["All", "Fiction", "Non-Fiction", "Thriller"],
        tones=["All", "Happy", "Sad", "Suspenseful"],
        tone_map={"Happy": "joy", "Sad": "sadness", "Suspenseful": "fear"},
        max_results=5,
    )


# --- Schema Tests ---


def test_book_recommendation_valid() -> None:
    """Verifies BookRecommendation accepts well-formed data."""
    book = BookRecommendation(
        title="Test Book",
        authors="Test Author",
        description="A test description.",
        rating=4.5,
        mood_score="0.85",
        category="Fiction",
    )
    assert book.title == "Test Book"
    assert book.rating == 4.5


def test_book_recommendation_rejects_extra_fields() -> None:
    """Verifies BookRecommendation enforces extra='forbid'."""
    with pytest.raises(ValidationError):
        BookRecommendation(
            title="Test",
            authors="Author",
            description="Desc",
            rating=4.0,
            mood_score="0.5",
            category="Fiction",
            extra_field="should fail",  # type: ignore[call-arg]
        )


def test_agent_response_valid() -> None:
    """Verifies AgentResponse accepts well-formed structured output."""
    response = AgentResponse(
        message="Here are some books for you!",
        recommendations=[
            BookRecommendation(
                title="Book A",
                authors="Author A",
                description="Description A",
                rating=4.0,
                mood_score="0.75",
                category="Fiction",
            )
        ],
        follow_up_suggestions=["Try thrillers", "Look for sad books"],
    )
    assert len(response.recommendations) == 1
    assert len(response.follow_up_suggestions) == 2
    assert response.message.startswith("Here are")


def test_agent_response_rejects_extra_fields() -> None:
    """Verifies AgentResponse enforces extra='forbid'."""
    with pytest.raises(ValidationError):
        AgentResponse(
            message="Hello",
            recommendations=[],
            follow_up_suggestions=[],
            secret="should fail",  # type: ignore[call-arg]
        )


def test_agent_response_defaults() -> None:
    """Verifies AgentResponse defaults for optional list fields."""
    response = AgentResponse(message="No books found.")
    assert response.recommendations == []
    assert response.follow_up_suggestions == []


# --- Tool Tests ---


def test_get_categories_returns_config(mock_deps: AgentDependencies) -> None:
    """Verifies get_available_categories returns the configured list."""
    ctx = MagicMock()
    ctx.deps = mock_deps

    result = get_available_categories(ctx)
    assert result == ["All", "Fiction", "Non-Fiction", "Thriller"]


def test_get_tones_returns_config(mock_deps: AgentDependencies) -> None:
    """Verifies get_available_tones returns the configured list."""
    ctx = MagicMock()
    ctx.deps = mock_deps

    result = get_available_tones(ctx)
    assert result == ["All", "Happy", "Sad", "Suspenseful"]
