import os
import pandas as pd
import numpy as np
import glob
import json
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_subcategory_data(data_dir: str = "data/processed/subcategories") -> dict:
    """
    Load all subcategory CSV files into a dictionary of DataFrames.
    Note: Used for subcategory-based workflows (e.g., sktime, XGBoost), not for aggregated data like sales_for_fc.csv.

    Args:
        data_dir (str): Directory containing subcategory CSV files. Defaults to "data/processed/subcategories".

    Returns:
        dict: Dictionary mapping subcategory names to DataFrames with 'sales' column and datetime index.

    Raises:
        FileNotFoundError: If no CSV files are found in the specified directory.
        Exception: For other data loading errors.
    """
    logger = logging.getLogger(__name__)
    try:
        data_dir = os.path.abspath(data_dir)
        logger.info(f"Loading subcategory files from: {data_dir}")
        csv_files = glob.glob(os.path.join(data_dir, "subcategory_*.csv"))
        if not csv_files:
            logger.error(f"No subcategory files found in {data_dir}")
            raise FileNotFoundError(f"No subcategory files found in {data_dir}")

        subcategory_data = {}
        for file in csv_files:
            subcategory = os.path.basename(file).replace(".csv", "")
            df = pd.read_csv(file, index_col="order_date", parse_dates=True)
            if df.empty:
                logger.warning(f"Empty DataFrame for {subcategory}")
                continue
            # Assume first column is the target (sales)
            col_name = df.columns[0]
            subcategory_data[subcategory] = df[[col_name]].rename(
                columns={col_name: "sales"}
            )
            logger.info(f"Loaded {subcategory} with shape {df.shape}")
        return subcategory_data
    except Exception as e:
        logger.error(f"Error loading subcategory data: {str(e)}")
        raise


def load_sales_data(data_path: str = "data/processed/sales_for_fc.csv") -> pd.DataFrame:
    """
    Load sales_for_fc.csv into a DataFrame with datetime index.

    Args:
        data_path (str): Path to sales_for_fc.csv.

    Returns:
        pd.DataFrame: DataFrame with datetime index and sales columns.

    Raises:
        FileNotFoundError: If data_path doesn't exist.
        ValueError: If data is empty or malformed.
    """
    logger = logging.getLogger(__name__)
    try:
        data_path = os.path.abspath(data_path)
        if not os.path.exists(data_path):
            logger.error(f"Data file not found at: {data_path}")
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        logger.info(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path, parse_dates=["order_date"])
        df.set_index("order_date", inplace=True)
        df = df.drop(columns=["Unnamed: 0"], errors="ignore")
        if df.empty:
            logger.error(f"Empty DataFrame at {data_path}")
            raise ValueError(f"Empty DataFrame at {data_path}")
        logger.info(f"Loaded sales data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading sales data: {str(e)}")
        raise


def mape(actual, pred):
    """
    Mean Absolute Percentage Error (MAPE) Function

    Args:
        actual: list/series of actual values
        pred: list/series of predicted values

    Returns:
        float: MAPE value as a percentage
    """
    actual, pred = np.array(actual), np.array(pred)
    mask = actual != 0  # Exclude zeros to avoid division by zero
    if not mask.any():
        logging.getLogger(__name__).warning(
            "MAPE undefined due to all zero actual values"
        )
        return np.nan
    return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100


def smape(actual, pred):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE) Function

    Args:
        actual: list/series of actual values
        pred: list/series of predicted values

    Returns:
        float: SMAPE value as a percentage
    """
    actual, pred = np.array(actual), np.array(pred)
    denominator = np.abs(actual) + np.abs(pred)
    if not denominator.any():
        logging.getLogger(__name__).warning(
            "SMAPE undefined due to zero sum of absolute values"
        )
        return np.nan
    return 100 * np.mean(2 * np.abs(pred - actual) / denominator)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Compute MAE, RMSE, MAPE, and SMAPE metrics.

    Args:
        y_true (pd.Series): Actual values
        y_pred (pd.Series): Predicted values

    Returns:
        dict: Dictionary containing 'mae', 'rmse', 'mape', and 'smape' metrics

    Raises:
        Exception: For errors during metric computation
    """
    logger = logging.getLogger(__name__)
    try:
        metrics = {}
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics["mape"] = mape(y_true, y_pred)
        metrics["smape"] = smape(y_true, y_pred)
        return metrics
    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        raise


def save_metrics(metrics: dict, output_path: str, model: str = "model") -> None:
    """
    Save metrics to a JSON file.

    Args:
        metrics (dict): Dictionary of metrics to save
        output_path (str): Path to save the JSON file (relative to project root)
        model (str): Model name for logging purposes. Defaults to "model"

    Raises:
        Exception: For errors during file saving
    """
    logger = logging.getLogger(__name__)
    try:
        # Resolve path relative to project root
        output_path = os.path.join(os.path.dirname(__file__), "..", "..", output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"{model} metrics saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving {model} metrics: {str(e)}")
        raise


def save_forecasts(forecasts: dict, output_path: str, model: str = "model") -> None:
    """
    Save forecasts to a CSV file.

    Args:
        forecasts (dict): Dictionary mapping subcategories to forecast results
        output_path (str): Path to save the CSV file (relative to project root)
        model (str): Model name for the 'model' column and logging. Defaults to "model"

    Raises:
        Exception: For errors during file saving
    """
    logger = logging.getLogger(__name__)
    try:
        # Resolve path relative to project root
        output_path = os.path.join(os.path.dirname(__file__), "..", "..", output_path)
        forecast_dfs = []
        for subcategory, result in forecasts.items():
            y_pred = result["y_pred"]
            df = pd.DataFrame(
                {
                    "subcategory": subcategory,
                    "date": y_pred.index,
                    "model": model,
                    "forecast": y_pred.values,
                    "lower_ci": (
                        result.get(
                            "ci",
                            pd.DataFrame(np.nan, index=y_pred.index, columns=[0, 1]),
                        ).iloc[:, 0]
                        if "ci" in result
                        else np.nan
                    ),
                    "upper_ci": (
                        result.get(
                            "ci",
                            pd.DataFrame(np.nan, index=y_pred.index, columns=[0, 1]),
                        ).iloc[:, 1]
                        if "ci" in result
                        else np.nan
                    ),
                }
            )
            forecast_dfs.append(df)
        forecast_df = pd.concat(forecast_dfs, ignore_index=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        forecast_df.to_csv(output_path, index=False)
        logger.info(f"{model} forecasts saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving {model} forecasts: {str(e)}")
        raise
