"""
Data models for the FastAPI serving layer.

This module defines the Pydantic schemas used for API requests and responses,
ensuring strict data validation at the API boundaries.
"""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """
    Schema for incoming tip prediction requests.
    Validates the raw trip characteristics inputted by the user or client system.
    """

    trip_distance: float = Field(
        ...,
        gt=0,
        description="Total distance of the ride in miles.",
        examples=[2.5],
    )
    total_amount: float = Field(
        ...,
        gt=0,
        description="Total amount charged to the passenger, excluding tip.",
        examples=[15.0],
    )
    passenger_count: int = Field(
        ...,
        gt=0,
        le=10,
        description="Number of passengers in the vehicle.",
        examples=[1],
    )
    ratecode_id: int = Field(
        ...,
        description="Final rate code in effect (e.g., 1 for Standard, 2 for JFK).",
        examples=[1],
    )
    airport_fee: float = Field(
        ...,
        ge=0,
        description="Additional fee for airport trips.",
        examples=[0.0],
    )
    congestion_surcharge: float = Field(
        ...,
        ge=0,
        description="Surcharge for entering the Manhattan congestion zone.",
        examples=[2.5],
    )
    tolls_amount: float = Field(
        ...,
        ge=0,
        description="Total amount of all tolls paid in trip.",
        examples=[0.0],
    )
    hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Pickup hour (0-23).",
        examples=[12],
    )
    day: int = Field(
        ...,
        ge=1,
        le=31,
        description="Day of the month when the trip started.",
        examples=[15],
    )
    month: int = Field(
        ...,
        ge=1,
        le=12,
        description="Month of the year (1-12).",
        examples=[1],
    )


class PredictResponse(BaseModel):
    """
    Schema for the prediction response returned by the API.
    """

    predicted_tip: float = Field(
        ...,
        description="The estimated tip amount in USD.",
        examples=[2.50],
    )
    model_version: str = Field(
        default="unknown",
        description="The version/name of the model used for prediction.",
        examples=["xgboost"],
    )
