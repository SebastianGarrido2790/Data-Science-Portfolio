import os
import pandas as pd
import logging
import pickle
from prophet import Prophet
import holidays
from datetime import timedelta
from src.utility.helper import load_sales_data, compute_metrics, save_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/backtesting_prophet.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def backtest_prophet(
    feature: str, df: pd.DataFrame, params: dict, horizon: int = 30
) -> dict:
    """
    Backtest Prophet model for a single feature.

    Args:
        feature (str): Feature name.
        df (pd.DataFrame): Data with datetime index and feature column.
        params (dict): Tuned hyperparameters.
        horizon (int): Forecasting horizon in days.

    Returns:
        dict: Metrics (MAE, RMSE, MAPE, SMAPE).

    Raises:
        ValueError: If data is invalid or insufficient.
        Exception: For other errors during backtesting.
    """
    try:
        logger.info(f"Backtesting Prophet for feature: {feature}")
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
            raise ValueError(f"No valid data for feature {feature} after preprocessing")

        # Check for date continuity
        date_range = pd.date_range(
            start=df_prophet["ds"].min(), end=df_prophet["ds"].max(), freq="D"
        )
        if len(date_range) != len(df_prophet):
            logger.warning(
                f"Non-continuous dates detected for {feature}. Expected {len(date_range)} days, got {len(df_prophet)}"
            )

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

        # Split data
        forecast_start_date = df_prophet["ds"].max() - timedelta(days=horizon)
        train_df = df_prophet[df_prophet["ds"] <= forecast_start_date]
        test_df = df_prophet[df_prophet["ds"] > forecast_start_date]

        if len(test_df) < horizon:
            logger.warning(
                f"Test data for {feature} has {len(test_df)} days, expected {horizon}"
            )
        if train_df.empty or test_df.empty:
            raise ValueError(
                f"Insufficient data for {feature}: train={len(train_df)}, test={len(test_df)}"
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
        )
        model.fit(train_df)

        # Predict
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        # Filter predictions for test period and ensure alignment
        pred_df = forecast[forecast["ds"].isin(test_df["ds"])][["ds", "yhat"]]
        pred_df = pred_df.merge(test_df[["ds", "y"]], on="ds", how="inner")

        if pred_df.empty:
            raise ValueError(
                f"No overlapping dates between forecast and test data for {feature}"
            )

        if pred_df["y"].isna().any() or pred_df["yhat"].isna().any():
            logger.warning(
                f"NaNs detected in pred_df for {feature}: y={pred_df['y'].isna().sum()}, yhat={pred_df['yhat'].isna().sum()}"
            )
            pred_df = pred_df.dropna()

        if pred_df.empty:
            raise ValueError(f"No valid data after NaN removal for {feature}")

        # Compute metrics
        metrics = compute_metrics(pred_df["y"], pred_df["yhat"])
        logger.info(f"Metrics for {feature}: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Backtesting failed for {feature}: {str(e)}")
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

        # Backtest for each feature
        all_metrics = {}
        for feature in df.columns:
            if feature in params_dict:
                all_metrics[feature] = {
                    "prophet": backtest_prophet(
                        feature, df, params_dict[feature], horizon=30
                    )
                }
            else:
                logger.warning(f"No parameters found for feature: {feature}. Skipping.")

        # Save metrics
        metrics_path = "../../../models/prophet/metrics.json"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        save_metrics(all_metrics, output_path=metrics_path, model="prophet")
        logger.info(f"Saved metrics to: {metrics_path}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
