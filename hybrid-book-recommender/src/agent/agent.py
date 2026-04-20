"""
Core Agent definition for the Book Recommender.

This module wires together the LLM model, system prompt, tools, and structured output
into a single pydantic-ai Agent instance. It follows the 'Agent as Tool' pattern
where the agent retains full control over state and uses tools as
stateless functions.

The agent uses Google Gemini Flash for cost-efficient reasoning.
"""

import sys

from pydantic_ai import Agent

from src.agent.prompts import BOOK_RECOMMENDER_SYSTEM_PROMPT
from src.agent.schemas import AgentResponse
from src.agent.tools import (
    AgentDependencies,
    get_available_categories,
    get_available_tones,
    search_books,
)
from src.app.data_loaders import get_app_config, init_recommender
from src.config.configuration import ConfigurationManager
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- Agent Definition ---
# The agent is defined without a default model here to allow runtime configuration.
# We use the generic 'google-gla:gemini-1.5-flash' as a default if not overridden.

book_agent: Agent[AgentDependencies, AgentResponse] = Agent(
    model=None,  # Set at runtime in run_sync
    deps_type=AgentDependencies,
    output_type=AgentResponse,
    system_prompt=BOOK_RECOMMENDER_SYSTEM_PROMPT,
    tools=[search_books, get_available_categories, get_available_tones],
)


def create_agent_dependencies() -> AgentDependencies | None:
    """Initializes the runtime dependencies for the agent.

    Returns:
        AgentDependencies | None: The initialized dependencies or None on failure.
    """
    try:
        recommender = init_recommender()
        if not recommender:
            logger.error("Failed to initialize recommender for agent")
            return None

        categories, tones, tone_map, _ = get_app_config()

        config = ConfigurationManager()
        agent_params = config.params.agent
        max_agent_results = int(agent_params.max_results_per_search)
        model_name = agent_params.model_name

        # Enforce pydantic-ai provider prefix if missing
        if ":" not in model_name:
            model_name = f"google-gla:{model_name}"

        return AgentDependencies(
            recommender=recommender,
            categories=categories,
            tones=tones,
            tone_map=tone_map,
            max_results=max_agent_results,
            model_name=model_name,
        )
    except Exception as e:
        logger.error(f"Failed to create agent dependencies: {CustomException(e, sys)}")
        return None


def chat(user_message: str, deps: AgentDependencies) -> AgentResponse:
    """Sends a message to the book recommender agent and returns a structured response.

    Args:
        user_message: The user's natural language message.
        deps: The runtime dependencies for the agent.

    Returns:
        AgentResponse: The agent's structured response with message,
        recommendations, and follow-up suggestions.

    Raises:
        CustomException: If the agent fails to produce a valid response.
    """
    try:
        # Pass the model from deps to override the default
        result = book_agent.run_sync(user_message, deps=deps, model=deps.model_name)
        return result.output
    except Exception as e:
        logger.error(f"Agent chat failed: {CustomException(e, sys)}")
        return AgentResponse(
            message="I'm sorry, I encountered an error while processing your request. "
            "Please try again or use the Search tab for direct recommendations.",
            recommendations=[],
            follow_up_suggestions=["Try the Search tab", "Rephrase your question"],
        )
