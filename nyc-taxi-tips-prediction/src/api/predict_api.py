"""
FastAPI Microservice for NYC Taxi Tips Prediction.

This module exposes the trained machine learning model as a REST API,
adhering to the FTI (Feature, Training, Inference) MLOps pattern by
decoupling model serving from the frontend application.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException

from src.config.configuration import ConfigurationManager
from src.entity.api_entity import PredictRequest, PredictResponse
from src.utils.feature_utils import encode_cyclical
from src.utils.logger import get_logger
from src.utils.model_utils import get_feature_importances

logger = get_logger(__name__, headline="API: Predict")

# Initialize Configuration Manager
config_manager = ConfigurationManager()
predict_config = config_manager.get_model_trainer_config()
fare_constants = config_manager.get_fare_constants()

# Global variable to store the loaded model
MODEL_REGISTRY = {}
MODEL_DIR = Path(predict_config.root_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event manager to load the ML model into memory exactly once
    at application startup, and clean up at shutdown.
    """
    logger.info("Starting up FastAPI Inference Server...")

    if not MODEL_DIR.exists():
        logger.warning(f"Model directory not found at {MODEL_DIR}")
    else:
        model_files = list(MODEL_DIR.glob("*.joblib"))
        if not model_files:
            logger.warning(f"No .joblib models found in {MODEL_DIR}")
        else:
            try:
                model_path = model_files[0]
                MODEL_REGISTRY["champion"] = joblib.load(model_path)
                MODEL_REGISTRY["model_version"] = model_path.stem
                logger.info(f"Successfully loaded model: {model_path.stem}")
            except Exception as e:
                logger.error(f"Failed to load model from {MODEL_DIR}: {e}")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down Inference Server. Clearing model registry.")
    MODEL_REGISTRY.clear()


app = FastAPI(
    title="NYC Taxi Tips Prediction API",
    description="Production-ready FTI architecture serving predictions via REST.",
    version="1.0.0",
    lifespan=lifespan,
)

# /v1 versioned router — all endpoints live here for backward-compat future
router = APIRouter(prefix="/v1")


def _preprocess_request(req: PredictRequest) -> dict:
    """
    Transforms the raw API PredictRequest into the feature dictionary
    expected by the trained ML model.
    """
    hour = req.hour
    day = req.day
    month = req.month

    # Cyclical feature engineering using shared utility (training-serving parity)
    # Hour (0-23, period=24) - same as feature_engineering.py
    hour_sin, hour_cos = encode_cyclical(float(hour), 24.0)
    # Day of Week: the API receives raw day (1-31). We map it to day-of-week
    # via a 7-day modular approximation consistent with training's weekday encoding.
    # NOTE: training uses actual weekday from datetime; here we use the user-supplied
    # day as a proxy (1-31 % 7) to maintain the same 7-period encoding.
    day_sin, day_cos = encode_cyclical(float((day - 1) % 7), 7.0)
    # Month (1-12 -> shifted 0-11, period=12) - same as feature_engineering.py
    month_sin, month_cos = encode_cyclical(float(month - 1), 12.0)

    return {
        "VendorID": 1.0,
        "passenger_count": float(req.passenger_count),
        "trip_distance": float(req.trip_distance),
        "RatecodeID": float(req.ratecode_id),
        "PULocationID": 132.0,  # Default fallback based on app logic
        "DOLocationID": 236.0,  # Default fallback based on app logic
        "payment_type": 1.0,
        "fare_amount": req.total_amount
        - req.airport_fee
        - req.congestion_surcharge
        - req.tolls_amount,
        "extra": fare_constants.get("extra", 0.0),
        "mta_tax": fare_constants.get("mta_tax", 0.5),
        "tolls_amount": float(req.tolls_amount),
        "improvement_surcharge": fare_constants.get("improvement_surcharge", 0.3),
        "total_amount": float(req.total_amount),
        "congestion_surcharge": float(req.congestion_surcharge),
        "Airport_fee": float(req.airport_fee),
        "pickup_hour_sin": hour_sin,
        "pickup_hour_cos": hour_cos,
        "pickup_day_sin": day_sin,
        "pickup_day_cos": day_cos,
        "pickup_month_sin": month_sin,
        "pickup_month_cos": month_cos,
    }


@router.get("/health", tags=["System"])
def health_check():
    """Simple healthcheck to verify API is active."""
    return {
        "status": "healthy",
        "model_loaded": "champion" in MODEL_REGISTRY,
        "model_version": MODEL_REGISTRY.get("model_version", "unknown"),
    }


@router.post("/predict", response_model=list[PredictResponse], tags=["Inference"])
def predict_tips(requests: list[PredictRequest]):
    """
    Accepts a batch of ride characteristics and returns predicted tips.
    """
    if "champion" not in MODEL_REGISTRY:
        raise HTTPException(status_code=503, detail="Model is not loaded or available.")

    model = MODEL_REGISTRY["champion"]
    version = MODEL_REGISTRY.get("model_version", "unknown")

    # 1. Preprocess all requests
    processed_dicts = [_preprocess_request(req) for req in requests]
    model_input_df = pd.DataFrame(processed_dicts)

    # 2. Ensure column alignment securely
    if hasattr(model, "feature_names_in_"):
        expected_cols = model.feature_names_in_
        for col in expected_cols:
            if col not in model_input_df.columns:
                model_input_df[col] = 0.0
        model_input_df = model_input_df[expected_cols]

    # 3. Predict
    try:
        preds = model.predict(model_input_df)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction execution failed.") from e

    # 4. Format responses
    responses = [
        PredictResponse(predicted_tip=float(pred), model_version=version) for pred in preds
    ]
    return responses


@router.get("/feature-importance", tags=["Inference"])
def feature_importance():
    """
    Returns the feature importance relative to the current champion model.
    """
    if "champion" not in MODEL_REGISTRY:
        raise HTTPException(status_code=503, detail="Model is not loaded or available.")

    model = MODEL_REGISTRY["champion"]
    feature_names, importances = get_feature_importances(model)

    if feature_names is None or importances is None:
        raise HTTPException(
            status_code=400,
            detail="Feature importance not supported for this model type.",
        )

    return {"features": feature_names, "importances": importances}


# Register the versioned router — all API endpoints are now under /v1
app.include_router(router)
