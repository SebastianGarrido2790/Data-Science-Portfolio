import pandas as pd
import numpy as np
import logging
import os
import sktime
from sktime.forecasting.base import ForecastingHorizon, BaseForecaster
from sktime.forecasting.model_selection import (
    temporal_train_test_split,
    ForecastingGridSearchCV,
)
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.forecasting.fbprophet import Prophet
from sktime.performance_metrics.forecasting import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sktime.utils.plotting import plot_series
from src.utility.helper import (
    mape,
    smape,
    load_subcategory_data,
    compute_metrics,
    save_metrics,
    save_forecasts,
)
import matplotlib.pyplot as plt
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/sktime_forecasting.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Log sktime version for debugging
logger.info(f"Using sktime version: {sktime.__version__}")

# Suppress warnings for cleaner output
warnings.simplefilter("ignore")


def tune_forecaster(
    forecaster: BaseForecaster, y_train: pd.Series, param_grid: dict
) -> BaseForecaster:
    """Tune forecaster hyperparameters using ForecastingGridSearchCV."""
    try:
        cv = SlidingWindowSplitter(
            window_length=len(y_train) // 2, step_length=len(y_train) // 10, fh=30
        )
        gscv = ForecastingGridSearchCV(
            forecaster=forecaster,
            cv=cv,
            param_grid=param_grid,
            scoring=mean_absolute_error,
            n_jobs=4,
        )
        gscv.fit(y_train)
        logger.info(f"Best parameters: {gscv.best_params_}")
        return gscv.best_forecaster_
    except Exception as e:
        logger.error(f"Error tuning forecaster: {str(e)}")
        raise
    finally:
        try:
            from joblib.externals.loky import get_reusable_executor

            get_reusable_executor().shutdown(wait=True)
        except Exception as e:
            logger.warning(f"Joblib cleanup failed: {str(e)}")


def sktime_forecast(
    dataset: pd.Series,
    horizon: int,
    forecaster: BaseForecaster,
    param_grid: dict = None,
    validation: bool = False,
    confidence: float = 0.95,
    frequency: str = "D",
    plot: bool = False,
    n_hist: int = None,
    subcategory: str = "sales",  # Added parameter for subcategory name
) -> dict:
    """
    Train and forecast for a single time series using an sktime forecaster.

    Args:
        dataset (pd.Series): Time series data with a datetime index.
        horizon (int): Number of periods to forecast.
        forecaster (BaseForecaster): Configured sktime forecaster instance.
        param_grid (dict, optional): Hyperparameters for tuning.
        validation (bool, optional): If True, split data for validation; else forecast beyond data. Defaults to False.
        confidence (float, optional): Confidence level for prediction intervals (0-1). Defaults to 0.95.
        frequency (str, optional): Resampling frequency (e.g., 'D' for daily). Defaults to "D".
        plot (bool, optional): If True, plot the results. Defaults to False.
        n_hist (int, optional): Number of historical periods to plot. Defaults to horizon*3.
        subcategory (str, optional): Subcategory name for plot filename. Defaults to "sales".

    Returns:
        dict: Results with keys 'y_pred', 'ci', and metrics (if validation=True).
    """
    try:
        # Input Validation
        if not isinstance(dataset.index, pd.DatetimeIndex):
            raise ValueError("Dataset must have a datetime index.")

        # Resample and handle missing values
        forecast_df = (
            dataset.resample(frequency)
            .sum()
            .interpolate(method="time")
            .fillna(method="bfill")
            .fillna(method="ffill")
        )

        if forecast_df.isna().any():
            logger.warning("NaNs remain after interpolation; consider reviewing data.")

        results = {}

        if validation:
            # Split for validation
            y_train, y_test = temporal_train_test_split(forecast_df, test_size=horizon)

            # Tune forecaster if param_grid is provided
            tuned_forecaster = (
                tune_forecaster(forecaster, y_train, param_grid)
                if param_grid
                else forecaster
            )
            tuned_forecaster.fit(y_train)

            # Predict and compute metrics
            fh = ForecastingHorizon(y_test.index, is_relative=False)
            y_pred = tuned_forecaster.predict(fh)
            ci = (
                tuned_forecaster.predict_interval(fh, coverage=confidence)
                if hasattr(tuned_forecaster, "predict_interval")
                else None
            )

            results = {
                "y_pred": y_pred,
                "ci": ci,
                "y_test": y_test,
                **compute_metrics(y_test, y_pred),
            }
        else:
            # Forecast beyond dataset
            tuned_forecaster = (
                tune_forecaster(forecaster, forecast_df, param_grid)
                if param_grid
                else forecaster
            )
            tuned_forecaster.fit(forecast_df)
            last_date = forecast_df.index.max()
            fh = ForecastingHorizon(
                pd.date_range(last_date, periods=horizon + 1, freq=frequency)[1:],
                is_relative=False,
            )
            y_pred = tuned_forecaster.predict(fh)
            ci = (
                tuned_forecaster.predict_interval(fh, coverage=confidence)
                if hasattr(tuned_forecaster, "predict_interval")
                else None
            )
            results = {"y_pred": y_pred, "ci": ci}

        # Plotting with error handling
        if plot:
            try:
                forecaster_name = (
                    tuned_forecaster.__class__.__name__
                    if param_grid
                    else forecaster.__class__.__name__
                )
                n_hist = n_hist or horizon * 3
                os.makedirs("reports/figures/sktime", exist_ok=True)
                if validation:
                    plot_series(
                        y_train[-n_hist:],
                        y_test,
                        y_pred,
                        labels=["Train", "Test", "Pred"],
                    )
                else:
                    plot_series(
                        forecast_df[-n_hist:], y_pred, labels=["Historical", "Forecast"]
                    )
                if ci is not None:
                    plt.fill_between(
                        y_pred.index, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.1
                    )
                plt.grid(False)
                plt.title(
                    f"{forecaster_name} - {horizon}-step {'Validation' if validation else 'Forecast'}"
                    + (f" (MAE: {results['mae']:.2f})" if validation else "")
                    + f" - {subcategory}"
                )
                plt.savefig(
                    f"reports/figures/sktime/sktime_{forecaster_name}_{subcategory}.png"
                )
                plt.close()
            except ImportError as e:
                logger.warning(
                    f"Plotting failed due to missing dependency: {str(e)}. Continuing without plots."
                )
            except Exception as e:
                logger.warning(
                    f"Plotting failed for {subcategory}: {str(e)}. Continuing without plots."
                )

        return results
    except Exception as e:
        logger.error(f"Forecasting failed for {subcategory}: {str(e)}")
        raise


if __name__ == "__main__":
    # Define forecaster and hyperparameter grid
    forecaster = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    param_grid = {
        "changepoint_prior_scale": [0.01, 0.05, 0.1],
        "seasonality_prior_scale": [0.1, 1.0, 10.0],
    }

    # Load subcategory data
    subcategory_data = load_subcategory_data()

    # Initialize results storage
    all_metrics = {}
    all_forecasts = {}

    # Process each subcategory
    for subcategory, df in subcategory_data.items():
        logger.info(f"Processing subcategory: {subcategory}")
        try:
            # Validation mode: evaluate last 30 days
            results_val = sktime_forecast(
                dataset=df["sales"],
                horizon=30,
                forecaster=forecaster,
                param_grid=param_grid,
                validation=True,
                confidence=0.95,
                frequency="D",
                plot=True,
                subcategory=subcategory,  # Pass subcategory name
            )
            all_metrics[subcategory] = {
                "sktime_Prophet": {
                    "mae": results_val["mae"],
                    "rmse": results_val["rmse"],
                    "mape": results_val["mape"],
                    "smape": results_val["smape"],
                }
            }

            # Forecasting mode: predict next 30 days
            results_fc = sktime_forecast(
                dataset=df["sales"],
                horizon=30,
                forecaster=forecaster,
                param_grid=param_grid,
                validation=False,
                confidence=0.95,
                frequency="D",
                plot=True,
                subcategory=subcategory,  # Pass subcategory name
            )
            all_forecasts[subcategory] = results_fc

            logger.info(f"Completed processing for {subcategory}")
        except Exception as e:
            logger.error(f"Failed processing {subcategory}: {str(e)}")
            continue

    # Save metrics and forecasts
    save_metrics(
        all_metrics, output_path="models/sktime/metrics.json", model="sktime_Prophet"
    )
    save_forecasts(
        all_forecasts, output_path="models/sktime/forecasts.csv", model="sktime_Prophet"
    )

    # Print summary
    for subcategory, metrics in all_metrics.items():
        print(f"{subcategory} - sktime_Prophet Metrics:")
        for metric, value in metrics["sktime_Prophet"].items():
            print(f"  {metric.upper()}: {value:.2f}")
