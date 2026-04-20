"""
Agentic Taxi Analyst — LangGraph ReAct Agent.

This module constructs the LangGraph-based "Agentic Taxi Analyst" that powers the
natural language chat interface on Page 2 of the Streamlit dashboard.

Architecture (Brain vs. Brawn separation):
- Brain: ChatGoogleGenerativeAI (gemini-2.5-flash) - reasoning, routing,
  natural language generation.
- Brawn: `predict_taxi_tip` @tool - deterministic HTTP call to the FastAPI serving layer.

Usage:
    >>> agent = get_taxi_analyst_agent()
    >>> result = agent.invoke({"messages": [("human", "Predict a tip for a 3-mile trip at 2 PM.")]})
    >>> print(result["messages"][-1].content)
"""

import os
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from src.agents.prompts import TAXI_ANALYST_SYSTEM_PROMPT
from src.tools.taxi_prediction_tool import (
    TaxiPredictionTool,
    TaxiRideInput,
)


class AgentConfigError(Exception):
    """Raised when the agent cannot be initialised due to missing configuration."""

    pass


@tool
def predict_taxi_tip(rides: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Calls the NYC Taxi Tip ML model to predict tip amounts for one or more rides.

    Use this tool whenever the user requests a tip prediction and you have extracted
    all required ride fields. Pass the data as a list of ride dictionaries matching
    the TaxiRideInput schema.

    Required fields per ride:
        - trip_distance (float, > 0): Total distance in miles.
        - total_amount (float, > 0): Total fare in USD, excluding the tip.
        - passenger_count (int, 1-6): Number of passengers.
        - ratecode_id (int): Rate code - 1 Standard, 2 JFK, 3 Newark, 4 Nassau/Westchester,
          5 Negotiated, 6 Group.
        - hour (int, 0-23): Pickup hour in 24-hour format.
        - day (int, 1-31): Day of the month.
        - month (int, 1-12): Month of the year.
        - airport_fee (float, optional, default 0.0): Airport surcharge in USD.
        - congestion_surcharge (float, optional, default 0.0): Congestion zone fee in USD.
        - tolls_amount (float, optional, default 0.0): Total tolls paid in USD.

    Args:
        rides: A list of ride dictionaries, each with the fields described above.

    Returns:
        A list of dictionaries each containing 'predicted_tip' (float, USD).

    Raises:
        PredictionToolError: If the ML serving API is unreachable or returns an error.
        ValueError: If ride data fails Pydantic validation.
    """
    parsed_rides = [TaxiRideInput(**r) for r in rides]
    taxi_tool = TaxiPredictionTool(api_url=os.getenv("API_URL", "http://localhost:8000"))
    return taxi_tool.predict_tips(parsed_rides)


def get_taxi_analyst_agent() -> Any:
    """
    Factory function that builds and returns a compiled LangGraph ReAct agent.

    The agent binds the `predict_taxi_tip` tool to GPT-4o-mini and uses
    the versioned TAXI_ANALYST_SYSTEM_PROMPT as its persona.

    Returns:
        A compiled LangGraph StateGraph (CompiledGraph) with `.invoke()` and
        `.stream()` interfaces.

    Raises:
        AgentConfigError: If GOOGLE_API_KEY is not set in the environment.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise AgentConfigError(
            "GOOGLE_API_KEY is not configured. Get a free key at "
            "https://aistudio.google.com/app/apikey and add it to the .env file."
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,  # Deterministic outputs — critical for structured tool calls
        google_api_key=api_key,
    )

    system_message = SystemMessage(content=TAXI_ANALYST_SYSTEM_PROMPT)

    return create_react_agent(
        model=llm,
        tools=[predict_taxi_tip],
        prompt=system_message,
    )
