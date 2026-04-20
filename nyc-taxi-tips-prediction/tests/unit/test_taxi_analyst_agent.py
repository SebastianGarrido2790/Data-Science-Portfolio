"""
Unit Tests for the LangGraph Agentic Taxi Analyst.

Tests the agent factory function and @tool wrapper without making any real
LLM or API calls — all external calls are mocked.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.agents.taxi_analyst_agent import (
    AgentConfigError,
    get_taxi_analyst_agent,
    predict_taxi_tip,
)
from src.tools.taxi_prediction_tool import PredictionToolError

# ---------------------------------------------------------------------------
# Test 1: Tool schema integrity
# ---------------------------------------------------------------------------


def test_tool_schema_contains_required_fields() -> None:
    """
    Verify the @tool-decorated `predict_taxi_tip` exposes an args_schema
    whose JSON schema contains the 'rides' input field expected by the agent.
    """
    schema = predict_taxi_tip.args_schema
    assert schema is not None, "Tool must have an args_schema"

    json_schema = schema.model_json_schema()
    assert "rides" in json_schema.get("properties", {}), (
        "Tool schema must expose a 'rides' property so the LLM can populate it."
    )


# ---------------------------------------------------------------------------
# Test 2: Agent factory — smoke test (no live LLM)
# ---------------------------------------------------------------------------


@patch("src.agents.taxi_analyst_agent.ChatGoogleGenerativeAI")
@patch.dict("os.environ", {"GOOGLE_API_KEY": "AIza-test-fake-key-for-unit-tests"})
def test_get_agent_returns_runnable(mock_chat_genai: MagicMock) -> None:
    """
    Verify that `get_taxi_analyst_agent()` returns a compiled LangGraph
    graph with an `.invoke` interface — no real Gemini API call is made.
    """
    mock_chat_genai.return_value = MagicMock()

    agent = get_taxi_analyst_agent()

    assert hasattr(agent, "invoke"), "Compiled LangGraph graph must expose an `.invoke()` method."


# ---------------------------------------------------------------------------
# Test 3: AgentConfigError on missing API key
# ---------------------------------------------------------------------------


@patch.dict("os.environ", {}, clear=True)  # strip all env vars
def test_get_agent_raises_on_missing_key() -> None:
    """
    Verify that `get_taxi_analyst_agent()` raises `AgentConfigError`
    when GOOGLE_API_KEY is absent from the environment.
    """
    with pytest.raises(AgentConfigError) as exc_info:
        get_taxi_analyst_agent()

    assert "GOOGLE_API_KEY" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 4: PredictionToolError propagates through the @tool wrapper
# ---------------------------------------------------------------------------


@patch("src.agents.taxi_analyst_agent.TaxiPredictionTool")
def test_tool_propagates_prediction_tool_error(mock_taxi_tool_cls: MagicMock) -> None:
    """
    Verify that a `PredictionToolError` raised by the backend propagates
    cleanly out of the `predict_taxi_tip` @tool — the LangGraph agent
    can then handle or surface it.
    """
    mock_tool_instance = mock_taxi_tool_cls.return_value
    mock_tool_instance.predict_tips.side_effect = PredictionToolError(
        "Model Serving API timed out after 5 seconds."
    )

    valid_ride: dict[str, Any] = {
        "trip_distance": 3.0,
        "total_amount": 18.0,
        "passenger_count": 1,
        "ratecode_id": 1,
        "airport_fee": 0.0,
        "congestion_surcharge": 2.5,
        "tolls_amount": 0.0,
        "hour": 10,
        "day": 15,
        "month": 6,
    }

    with pytest.raises(PredictionToolError) as exc_info:
        predict_taxi_tip.invoke({"rides": [valid_ride]})

    assert "timed out" in str(exc_info.value)
