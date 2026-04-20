"""
Structured output models for the Book Recommender Agent.

These Pydantic models enforce structured output, ensuring the agent
communicates with the UI layer via validated JSON — never free text.
All models enforce `extra="forbid"` per project convention.
"""

from pydantic import BaseModel, ConfigDict, Field


class BookRecommendation(BaseModel):
    """A single book recommendation returned by the agent."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., description="Title of the book")
    authors: str = Field(..., description="Authors of the book")
    description: str = Field(..., description="Brief summary of the book")
    rating: float = Field(..., description="Average rating (0-5)")
    mood_score: str = Field(..., description="Dominant mood probability score")
    category: str = Field(..., description="Genre category of the book")


class AgentResponse(BaseModel):
    """Structured response from the Book Recommender Agent."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., description="Conversational response to the user")
    recommendations: list[BookRecommendation] = Field(
        default_factory=list,
        description="List of book recommendations",
    )
    follow_up_suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up queries the user might ask",
    )
