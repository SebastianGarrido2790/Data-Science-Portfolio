"""
Agent Tool Abstraction: NYC Taxi Tip Predictor.

This module provides a deterministic, strongly-typed tool interface designed
specifically for Agentic (LLM) Systems to interact with the FTI Model Serving layer.
"""

import os
from typing import Any

import requests
from pydantic import BaseModel, Field


class PredictionToolError(Exception):
    """Domain-specific exception for Agentic Tool failures to prevent silent errors."""

    pass


class TaxiRideInput(BaseModel):
    """
    Strict Pydantic schema representing a single NYC Taxi ride.
    The LLM agent reads these descriptions to understand how to format its output.
    """

    trip_distance: float = Field(
        ...,
        gt=0,
        description="Total distance of the ride in miles. Must be greater than 0.",
    )
    total_amount: float = Field(
        ...,
        gt=0,
        description=(
            "Total amount charged to the passenger in USD, excluding the tip. "
            "Must be greater than 0."
        ),
    )
    passenger_count: int = Field(
        ..., ge=1, le=6, description="Number of passengers in the vehicle (1 to 6)."
    )
    ratecode_id: int = Field(
        ...,
        description=(
            "Final rate code in effect (e.g., 1 for Standard, 2 for JFK, 3 for Newark, etc.)."
        ),
    )
    airport_fee: float = Field(
        0.0, ge=0, description="Additional fee for airport trips in USD, if applicable."
    )
    congestion_surcharge: float = Field(
        0.0,
        ge=0,
        description="Surcharge for entering the Manhattan congestion zone in USD.",
    )
    tolls_amount: float = Field(
        0.0, ge=0, description="Total amount of all tolls paid in trip in USD."
    )
    hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of the day when the meter was engaged in 24-hour format (0 to 23).",
    )
    day: int = Field(
        ...,
        ge=1,
        le=31,
        description="Day of the month when the trip started (1 to 31).",
    )
    month: int = Field(..., ge=1, le=12, description="Month of the year (1 to 12).")


class TaxiPredictionTool:
    """
    Deterministic Tool for LLM Agents to fetch taxi tip predictions.

    This class handles the rigid HTTP execution, serialization, and error wrapping
    so the Agent (Brain) focuses strictly on reasoning and orchestrating workflow.
    """

    def __init__(self, api_url: str | None = None):
        """
        Initializes the Tool with the target ML Serving endpoint.

        Args:
            api_url: The base URL of the FastAPI service.
                     Defaults to the environment variable or localhost.
        """
        self.api_url = api_url or os.getenv("API_URL", "http://localhost:8000")
        self.predict_endpoint = f"{self.api_url}/v1/predict"
        self.health_endpoint = f"{self.api_url}/v1/health"

    def predict_tips(self, rides: list[TaxiRideInput]) -> list[dict[str, Any]]:
        """
        Calculates the expected tip amount for a batch of NYC Taxi rides.

        Args:
            rides: A list of TaxiRideInput objects detailing the ride parameters.

        Returns:
            A list of dictionary results containing the 'predicted_tip' in USD.

        Raises:
            PredictionToolError: If the backend is unreachable or returns validation/server errors.
        """
        if not rides:
            raise PredictionToolError("No rides provided for prediction. The list is empty.")

        # Serialize Pydantic objects to JSON-friendly dictionaries
        payload = [ride.model_dump() for ride in rides]

        try:
            response = requests.post(self.predict_endpoint, json=payload, timeout=5)
            response.raise_for_status()

            # Fast-Fail Logic: Ensure API contract is met
            data = response.json()
            if not isinstance(data, list) or len(data) == 0 or "predicted_tip" not in data[0]:
                raise PredictionToolError(f"Unexpected response format from Model API: {data}")

            return data

        except requests.exceptions.Timeout as e:
            raise PredictionToolError(
                f"Model Serving API timed out after 5 seconds at {self.predict_endpoint}. "
                "Ensure the deployment is healthy."
            ) from e
        except requests.exceptions.HTTPError as e:
            raise PredictionToolError(
                f"Model Serving API returned HTTP error {e.response.status_code}: {e.response.text}"
            ) from e
        except requests.exceptions.RequestException as e:
            raise PredictionToolError(
                f"Network error communicating with the Model Serving API: {e}"
            ) from e

    def check_health(self) -> dict[str, Any]:
        """
        Verifies if the Model Serving Pipeline is online and ready for inferences.
        """
        try:
            response = requests.get(self.health_endpoint, timeout=2)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            msg = f"Model capability is currently offline or unreachable: {e}"
            raise PredictionToolError(msg) from e
