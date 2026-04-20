"""
Unit Tests for the Agent Tool Abstractions.

Verifies that the `TaxiPredictionTool` correctly validates LLM input via Pydantic
and deterministically handles both successful and failed HTTP responses from the backend.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from requests.exceptions import HTTPError, ReadTimeout

from src.tools.taxi_prediction_tool import (
    PredictionToolError,
    TaxiPredictionTool,
    TaxiRideInput,
)

# --- Validation Tests ---


def test_pydantic_schema_validation_success():
    """Ensure valid Agent inputs are successfully parsed into the Pydantic schema."""
    valid_data = {
        "trip_distance": 2.5,
        "total_amount": 15.0,
        "passenger_count": 2,
        "ratecode_id": 1,
        "airport_fee": 1.25,
        "congestion_surcharge": 2.5,
        "tolls_amount": 0.0,
        "hour": 14,
        "day": 5,
        "month": 10,
    }

    ride = TaxiRideInput(**valid_data)
    assert ride.trip_distance == 2.5
    assert ride.passenger_count == 2


def test_pydantic_schema_validation_failure():
    """Ensure invalid LLM hallucinations trigger Pydantic ValidationErrors before network calls."""
    invalid_data = {
        "trip_distance": -5.0,  # Cannot be negative
        "total_amount": 15.0,
        "passenger_count": 9,  # Exceeds max
        "ratecode_id": 1,
        "airport_fee": 0.0,
        "congestion_surcharge": 0.0,
        "tolls_amount": 0.0,
        "hour": 25,  # Invalid hour
        "day": 5,
        "month": 10,
    }

    with pytest.raises(ValidationError) as exc_info:
        TaxiRideInput(**invalid_data)

    errors = exc_info.value.errors()
    assert len(errors) == 3

    error_fields = [err["loc"][0] for err in errors]
    assert "trip_distance" in error_fields
    assert "passenger_count" in error_fields
    assert "hour" in error_fields


# --- Tool Network Execution Tests ---


@pytest.fixture
def mock_tool():
    """Provides a fresh instance of the tool."""
    return TaxiPredictionTool(api_url="http://fake-backend:8000")


@pytest.fixture
def valid_rides():
    """Provides valid input data."""
    return [
        TaxiRideInput(
            trip_distance=2.5,
            total_amount=15.0,
            passenger_count=1,
            ratecode_id=1,
            airport_fee=0.0,
            congestion_surcharge=2.5,
            tolls_amount=0.0,
            hour=12,
            day=15,
            month=1,
        )
    ]


@patch("src.tools.taxi_prediction_tool.requests.post")
def test_tool_predict_success(mock_post, mock_tool, valid_rides):
    """Verifies that the tool correctly parses a 200 OK inference response."""
    # Mock a successful API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"predicted_tip": 3.45}]
    mock_post.return_value = mock_response

    results = mock_tool.predict_tips(valid_rides)

    assert isinstance(results, list)
    assert len(results) == 1
    assert "predicted_tip" in results[0]
    assert results[0]["predicted_tip"] == 3.45

    # Ensure the URL is hit properly
    assert mock_post.call_args[0][0] == "http://fake-backend:8000/v1/predict"


@patch("src.tools.taxi_prediction_tool.requests.post")
def test_tool_predict_timeout(mock_post, mock_tool, valid_rides):
    """Verifies that the tool intercepts ReadTimeouts and wraps them in custom exception."""
    mock_post.side_effect = ReadTimeout("Request timed out")

    with pytest.raises(PredictionToolError) as exc_info:
        mock_tool.predict_tips(valid_rides)

    assert "timed out after 5 seconds" in str(exc_info.value)


@patch("src.tools.taxi_prediction_tool.requests.post")
def test_tool_predict_http_error(mock_post, mock_tool, valid_rides):
    """Verifies that the tool intercepts 500/400 errors and wraps them in custom exception."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    mock_http_error = HTTPError()
    mock_http_error.response = mock_response

    mock_post.side_effect = mock_http_error

    with pytest.raises(PredictionToolError) as exc_info:
        mock_tool.predict_tips(valid_rides)

    assert "returned HTTP error 500" in str(exc_info.value)
    assert "Internal Server Error" in str(exc_info.value)


def test_tool_predict_empty_list(mock_tool):
    """Verifies fast-fail logic if the agent provides no rides."""
    with pytest.raises(PredictionToolError) as exc_info:
        mock_tool.predict_tips([])

    assert "The list is empty" in str(exc_info.value)
