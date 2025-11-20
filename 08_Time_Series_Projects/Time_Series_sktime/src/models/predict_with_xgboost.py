import os
import pandas as pd
import numpy as np
import logging
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from joblib import Parallel, delayed
import matplotlib

matplotlib.use("Agg")  # Set non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import holidays
import warnings

from src.utility.helper import (
    mape,
    smape,
    load_subcategory_data,
    compute_metrics,
    save_metrics,
    save_forecasts,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/xgboost_logs.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Log xgboost version
logger.info(f"Using xgboost version: {xgb.__version__}")

# Suppress warnings
warnings.simplefilter("ignore")


def prepare_features(df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    """Engineer features for time series forecasting."""
    try:
        df = df.copy()
        # Handle outliers
        # Q1 = df["sales"].quantile(0.25)
        # Q3 = df["sales"].quantile(0.75)
        # IQR = Q3 - Q1
        # lower_bound = Q1 - 1.5 * IQR
        # upper_bound = Q3 + IQR
        # df["sales"] = df["sales"].clip(lower=lower_bound, upper=upper_bound)

        # Lagged features
        df["lag_1"] = df["sales"].shift(1)
        df["lag_7"] = df["sales"].shift(7)
        df["lag_30"] = df["sales"].shift(30)

        # Temporal features
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)
        df["week_of_year"] = df.index.isocalendar().week

        # Rolling statistics
        df["rolling_mean_7"] = df["sales"].rolling(window=7).mean()
        df["rolling_std_7"] = df["sales"].rolling(window=7).std()

        # Holiday feature
        us_holidays = holidays.US()
        df["is_holiday"] = df.index.map(lambda x: 1 if x in us_holidays else 0)

        # Drop rows with NaNs
        df = df.dropna()
        return df
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    """Tune XGBoost hyperparameters using RandomizedSearchCV."""
    try:
        model = xgb.XGBRegressor(objective="reg:squarederror")
        param_dist = {
            "max_depth": [3, 5, 7, 10],
            "n_estimators": [50, 100, 200, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.7, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 5, 10],
            "gamma": [0, 0.1, 0.5],
            "lambda": [0.1, 1, 10],
            "alpha": [0.1, 1, 10],
        }
        tscv = TimeSeriesSplit(n_splits=5)
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=25,  # Reduced iterations
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=4,  # Limit parallel jobs
            random_state=42,
        )
        random_search.fit(X_train, y_train)
        logger.info(f"Best parameters: {random_search.best_params_}")
        return random_search.best_estimator_
    except Exception as e:
        logger.error(f"Error tuning XGBoost: {str(e)}")
        raise
    finally:
        # Cleanup joblib resources
        try:
            from joblib.externals.loky import get_reusable_executor

            get_reusable_executor().shutdown(wait=True)
        except Exception as e:
            logger.warning(f"Joblib cleanup failed: {str(e)}")


def xgboost_forecast(
    dataset: pd.DataFrame,
    horizon: int = 30,
    features: list = None,
    validation: bool = False,
    plot: bool = True,
    n_hist: int = None,
    subcategory: str = "sales",
) -> dict:
    """Train and forecast for a single time series using XGBoost."""
    try:
        # Input validation
        if not isinstance(dataset.index, pd.DatetimeIndex):
            raise ValueError("Dataset must have a datetime index.")
        if "sales" not in dataset.columns:
            raise ValueError("Dataset must have a 'sales' column.")

        # Prepare features
        df = prepare_features(dataset, horizon)
        features = features or [
            "lag_1",
            "lag_7",
            "lag_30",
            "day_of_week",
            "month",
            "is_weekend",
            "rolling_mean_7",
            "rolling_std_7",
            "is_holiday",
            "week_of_year",
        ]

        results = {}

        if validation:
            # Split for validation
            train_df = df.iloc[:-horizon]
            test_df = df.iloc[-horizon:]
            X_train = train_df[features]
            y_train = train_df["sales"]
            X_test = test_df[features]
            y_test = test_df["sales"]

            # Train model
            model = tune_xgboost(X_train, y_train)
            y_pred = model.predict(X_test)

            # Compute metrics

            results = {
                "y_pred": pd.Series(y_pred, index=y_test.index),
                "y_test": y_test,
                **compute_metrics(y_test, y_pred),
            }
        else:
            # Forecast beyond dataset
            X_train = df[features]
            y_train = df["sales"]
            model = tune_xgboost(X_train, y_train)

            # Generate future dates
            last_date = df.index.max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
            )
            future_df = pd.DataFrame(index=future_dates)
            last_data = df.tail(horizon).copy()
            predictions = []

            for date in future_dates:
                next_day = pd.DataFrame(index=[date])
                next_day["lag_1"] = last_data["sales"].iloc[-1]
                next_day["lag_7"] = (
                    last_data["sales"].iloc[-7]
                    if len(last_data) >= 7
                    else last_data["sales"].iloc[0]
                )
                next_day["lag_30"] = (
                    last_data["sales"].iloc[-30]
                    if len(last_data) >= 30
                    else last_data["sales"].iloc[0]
                )
                next_day["day_of_week"] = date.dayofweek
                next_day["month"] = date.month
                next_day["is_weekend"] = int(date.dayofweek in [5, 6])
                next_day["rolling_mean_7"] = last_data["sales"].tail(7).mean()
                next_day["rolling_std_7"] = last_data["sales"].tail(7).std()
                next_day["is_holiday"] = int(date in holidays.US())
                next_day["week_of_year"] = date.isocalendar().week

                pred = model.predict(next_day[features])[0]
                predictions.append(pred)

                new_row = next_day.copy()
                new_row["sales"] = pred
                last_data = pd.concat([last_data, new_row])

            results = {"y_pred": pd.Series(predictions, index=future_dates)}

        # Plotting
        if plot:
            try:
                n_hist = n_hist or horizon * 3
                os.makedirs("reports/figures/xgboost", exist_ok=True)
                plt.figure(figsize=(12, 6))
                plt.plot(
                    dataset["sales"][-n_hist:],
                    label="Historical Sales",
                    color="black",
                    alpha=0.5,
                )
                if validation:
                    plt.plot(
                        results["y_test"],
                        label="Actual (Test)",
                        color="blue",
                        linewidth=2,
                    )
                    plt.plot(
                        results["y_pred"],
                        label="Predicted (Test)",
                        color="red",
                        linestyle="--",
                        linewidth=2,
                    )
                else:
                    plt.plot(
                        results["y_pred"],
                        label="Forecasted",
                        color="green",
                        linestyle="--",
                        linewidth=2,
                    )
                plt.title(
                    f"XGBoost - {horizon}-step {'Validation' if validation else 'Forecast'} "
                    + (f"(MAE: {results['mae']:.2f}) " if validation else "")
                    + f"- {subcategory}"
                )
                plt.xlabel("Date")
                plt.ylabel("Sales")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"reports/figures/xgboost/xgboost_{subcategory}.png")
                plt.close()
            except Exception as e:
                logger.warning(f"Plotting failed for {subcategory}: {str(e)}")

        return results
    except Exception as e:
        logger.error(f"Forecasting failed for {subcategory}: {str(e)}")
        raise


if __name__ == "__main__":
    # Load subcategory data
    subcategory_data = load_subcategory_data()

    # Initialize results storage
    all_metrics = {}
    all_forecasts = {}

    # Process each subcategory
    for subcategory, df in subcategory_data.items():
        logger.info(f"Processing subcategory: {subcategory}")
        try:
            # Validation mode
            results_val = xgboost_forecast(
                dataset=df,
                horizon=30,
                validation=True,
                plot=True,
                subcategory=subcategory,
            )
            all_metrics[subcategory] = {
                "xgboost": {
                    "mae": results_val["mae"],
                    "rmse": results_val["rmse"],
                    "mape": results_val["mape"],
                    "smape": results_val["smape"],
                }
            }

            # Forecasting mode
            results_fc = xgboost_forecast(
                dataset=df,
                horizon=30,
                validation=False,
                plot=True,
                subcategory=subcategory,
            )
            all_forecasts[subcategory] = results_fc

            logger.info(f"Completed processing for {subcategory}")
        except Exception as e:
            logger.error(f"Failed processing {subcategory}: {str(e)}")
            continue

    # Save metrics and forecasts
    save_metrics(
        all_metrics, output_path="models/xgboost/metrics.json", model="xgboost"
    )
    save_forecasts(
        all_forecasts, output_path="models/xgboost/forecasts.csv", model="xgboost"
    )

    # Print summary
    for subcategory, metrics in all_metrics.items():
        print(f"{subcategory.upper()} - XGBoost Metrics:")
        for metric, value in metrics["xgboost"].items():
            print(f"  {metric.capitalize()}: {value:.2f}")
