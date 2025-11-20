import os
import pandas as pd
import numpy as np
import logging
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prophet import Prophet
import holidays
from datetime import timedelta
from src.utility.helper import load_sales_data, save_forecasts
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/prophet.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def forecast_prophet(
    feature: str,
    df: pd.DataFrame,
    params: dict,
    horizon: int = 30,
    confidence: float = 0.95,
    plot: bool = True,
) -> dict:
    """
    Generate forecast using Prophet for a single feature.

    Args:
        feature (str): Feature name.
        df (pd.DataFrame): Data with datetime index and feature column.
        params (dict): Tuned hyperparameters.
        horizon (int): Forecasting horizon in days.
        confidence (float): Confidence level for prediction intervals.
        plot (bool): If True, save forecast plot.

    Returns:
        dict: Dictionary with 'y_pred' (Series of predictions) and 'ci' (DataFrame of confidence intervals).

    Raises:
        ValueError: If data is invalid or empty.
        Exception: For other errors during forecasting.
    """
    try:
        logger.info(f"Generating forecast for feature: {feature}")
        # Prepare DataFrame for Prophet
        df_prophet = (
            df[[feature]]
            .reset_index()
            .rename(columns={feature: "y", "order_date": "ds"})
        )
        df_prophet["y"] = pd.to_numeric(df_prophet["y"], errors="coerce")
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])
        df_prophet = df_prophet.dropna()

        if df_prophet.empty:
            raise ValueError(f"No valid data for feature {feature}")

        # Prepare holidays
        us_holidays = holidays.US(years=range(2015, 2026))
        holiday_df = pd.DataFrame(
            {
                "holiday": "US_Holidays",
                "ds": pd.to_datetime(list(us_holidays.keys())),
                "lower_window": -1,
                "upper_window": 1,
            }
        )

        # Model
        model = Prophet(
            changepoint_prior_scale=params.get("changepoint_prior_scale", 0.05),
            seasonality_prior_scale=params.get("seasonality_prior_scale", 1.0),
            holidays_prior_scale=params.get("holidays_prior_scale", 1.0),
            seasonality_mode=params.get("seasonality_mode", "multiplicative"),
            changepoint_range=params.get("changepoint_range", 0.8),
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            holidays=holiday_df,
            interval_width=confidence,
        )
        model.fit(df_prophet)

        # Forecast
        future = model.make_future_dataframe(periods=horizon)
        forecast_df = model.predict(future)
        # Filter future forecast
        forecast_start_date = df_prophet["ds"].max()
        forecast_df = forecast_df[forecast_df["ds"] >= forecast_start_date]

        # Prepare output
        result = {
            "y_pred": forecast_df["yhat"],
            "ci": forecast_df[["yhat_lower", "yhat_upper"]],
        }

        # Plotting
        if plot:
            try:
                os.makedirs("../../../reports/figures/prophet/", exist_ok=True)
                plt.figure(figsize=(12, 6))
                plt.plot(
                    df_prophet["ds"],
                    df_prophet["y"],
                    label="Historical Sales",
                    color="black",
                    alpha=0.5,
                )
                plt.plot(
                    forecast_df["ds"],
                    forecast_df["yhat"],
                    label="Forecast",
                    color="green",
                    linestyle="--",
                )
                plt.fill_between(
                    forecast_df["ds"],
                    forecast_df["yhat_lower"],
                    forecast_df["yhat_upper"],
                    color="green",
                    alpha=0.1,
                    label="95% Confidence Interval",
                )
                plt.title(f"Prophet Forecast - {horizon}-step - {feature}")
                plt.xlabel("Date")
                plt.ylabel("Sales")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(
                    f"../../../reports/figures/prophet/prophet_forecast_{feature}.png"
                )
                plt.close()
                logger.info(
                    f"Saved plot for {feature} to ../../../reports/figures/prophet/prophet_forecast_{feature}.png"
                )
            except Exception as e:
                logger.warning(f"Plotting failed for {feature}: {str(e)}")

        return result
    except Exception as e:
        logger.error(f"Error forecasting for {feature}: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Load data
        data_path = "../../../data/processed/sales_for_fc.csv"
        abs_data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), data_path)
        )
        if not os.path.exists(abs_data_path):
            logger.error(f"Data file not found at: {abs_data_path}")
            raise FileNotFoundError(f"Data file not found at: {abs_data_path}")
        logger.info(f"Loading data from: {abs_data_path}")
        df = load_sales_data(data_path)

        # Load parameters
        params_path = "../../../models/prophet/params.pkl"
        abs_params_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), params_path)
        )
        if not os.path.exists(abs_params_path):
            logger.error(f"Parameters file not found at: {abs_params_path}")
            raise FileNotFoundError(f"Parameters file not found at: {abs_params_path}")
        try:
            with open(params_path, "rb") as f:
                params_dict = pickle.load(f)
            logger.info(f"Loaded parameters from: {abs_params_path}")
        except Exception as e:
            logger.error(f"Error loading parameters: {str(e)}")
            raise

        # Forecast for each feature
        all_forecasts = {}
        for feature in df.columns:
            if feature in params_dict:
                all_forecasts[feature] = forecast_prophet(
                    feature,
                    df,
                    params_dict[feature],
                    horizon=30,
                    confidence=0.95,
                    plot=True,
                )
            else:
                logger.warning(f"No parameters found for feature: {feature}. Skipping.")

        # Save forecasts
        forecasts_path = "../../../models/prophet/forecasts.csv"
        os.makedirs(os.path.dirname(forecasts_path), exist_ok=True)
        save_forecasts(all_forecasts, output_path=forecasts_path, model="prophet")
        logger.info(f"Saved forecasts to: {forecasts_path}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
