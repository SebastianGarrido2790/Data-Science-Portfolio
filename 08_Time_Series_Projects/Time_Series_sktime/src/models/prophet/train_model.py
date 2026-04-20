import os
import pandas as pd
import numpy as np
import logging
import itertools
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from joblib import Parallel, delayed
import pickle
import warnings
from src.utility.helper import load_sales_data

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

# Suppress warnings
warnings.simplefilter("ignore")


def tune_prophet(feature: str, df: pd.DataFrame, param_grid: dict) -> dict:
    """
    Tune Prophet hyperparameters for a single feature using cross-validation.

    Args:
        feature (str): Column name in the dataset.
        df (pd.DataFrame): Time series data with datetime index and feature columns.
        param_grid (dict): Hyperparameter grid for tuning.

    Returns:
        dict: Best parameters and SMAPE score.
    """
    try:
        logger.info(f"Tuning Prophet for feature: {feature}")
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

        all_params = [
            dict(zip(param_grid.keys(), v))
            for v in itertools.product(*param_grid.values())
        ]
        all_params = np.random.choice(
            all_params, size=min(25, len(all_params)), replace=False
        )

        def evaluate_params(params):
            try:
                model = Prophet(
                    **params,
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=True,
                )
                model.fit(df_prophet)
                df_cv = cross_validation(
                    model, initial="730 days", period="30 days", horizon="30 days"
                )
                df_p = performance_metrics(df_cv, rolling_window=1)
                return params, df_p["smape"].values[0]
            except Exception as e:
                logger.warning(f"Failed parameters {params}: {str(e)}")
                return params, np.inf

        results = Parallel(n_jobs=4)(
            delayed(evaluate_params)(params) for params in all_params
        )
        smapes = [r[1] for r in results]
        best_params = min(results, key=lambda x: x[1])[0]
        best_params["smape"] = min(smapes)
        best_params["feature"] = feature
        logger.info(f"Best parameters for {feature}: {best_params}")
        return best_params
    except Exception as e:
        logger.error(f"Error tuning Prophet for {feature}: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Define data path relative to sales-forecasting root
        data_path = "../../../data/processed/sales_for_fc.csv"
        # Validate file existence
        abs_data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), data_path)
        )
        if not os.path.exists(abs_data_path):
            logger.error(f"Data file not found at: {abs_data_path}")
            raise FileNotFoundError(f"Data file not found at: {abs_data_path}")

        # Load data
        logger.info(f"Loading data from: {data_path}")
        df = load_sales_data(data_path)
        # Ensure datetime index (load_sales_data already sets this, but reinforce for safety)
        if not isinstance(df.index, pd.DatetimeIndex):
            df["order_date"] = pd.to_datetime(df["order_date"])
            df.set_index("order_date", inplace=True)
        df = df.drop(columns=["Unnamed: 0"], errors="ignore")

        # Define hyperparameter grid
        param_grid = {
            "changepoint_prior_scale": [0.01, 0.05, 0.1, 0.5],
            "seasonality_prior_scale": [0.1, 1.0, 10.0],
            "holidays_prior_scale": [0.1, 1.0, 10.0],
            "seasonality_mode": ["additive", "multiplicative"],
            "changepoint_range": [0.8, 0.9],
        }

        # Tune parameters for each feature
        params_dict = {}
        for feature in df.columns:
            params_dict[feature] = tune_prophet(feature, df, param_grid)

        # Save parameters
        params_path = "../../../models/prophet/params.pkl"
        os.makedirs(os.path.dirname(params_path), exist_ok=True)
        with open(params_path, "wb") as f:
            pickle.dump(params_dict, f)
        logger.info(f"Saved parameters to: {params_path}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
    finally:
        try:
            from joblib.externals.loky import get_reusable_executor

            get_reusable_executor().shutdown(wait=True)
        except Exception as e:
            logger.warning(f"Joblib cleanup failed: {str(e)}")
