import os
import logging
import pandas as pd
import numpy as np

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_data(df, expected_columns):
    try:
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(
                f"Missing values detected:\n{missing_values[missing_values > 0]}"
            )
            df = df.dropna()
            logger.info("Dropped rows with missing values.")

        if not pd.to_numeric(df["age"], errors="coerce").notnull().all():
            logger.error("Invalid 'age' values: must be numeric.")
            raise ValueError("Invalid 'age' values: must be numeric.")
        if (df["age"] < 0).any():
            logger.error("Invalid 'age' values: must be non-negative.")
            raise ValueError("Invalid 'age' values: must be non-negative.")

        if not pd.to_numeric(df["tenure"], errors="coerce").notnull().all():
            logger.error("Invalid 'tenure' values: must be numeric.")
            raise ValueError("Invalid 'tenure' values: must be numeric.")
        if (df["tenure"] < 0).any():
            logger.error("Invalid 'tenure' values: must be non-negative.")
            raise ValueError("Invalid 'tenure' values: must be non-negative.")

        if not pd.to_numeric(df["spend_rate"], errors="coerce").notnull().all():
            logger.error("Invalid 'spend_rate' values: must be numeric.")
            raise ValueError("Invalid 'spend_rate' values: must be numeric.")
        if (df["spend_rate"] < 0).any():
            logger.error("Invalid 'spend_rate' values: must be non-negative.")
            raise ValueError("Invalid 'spend_rate' values: must be non-negative.")

        if not df["plan_type"].apply(lambda x: isinstance(x, str)).all():
            logger.error("Invalid 'plan_type' values: must be strings.")
            raise ValueError("Invalid 'plan_type' values: must be strings.")

        if not df["churn"].isin([0, 1]).all():
            logger.error("Invalid 'churn' values: must be 0 or 1.")
            raise ValueError("Invalid 'churn' values: must be 0 or 1.")

        if not df["ticket_notes"].apply(lambda x: isinstance(x, str)).all():
            logger.error("Invalid 'ticket_notes' values: must be strings.")
            raise ValueError("Invalid 'ticket_notes' values: must be strings.")

        logger.info("Data validation passed successfully.")
        return df
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise


def make_dataset(raw_data_path, output_path):
    try:
        logger.info("Loading raw dataset...")
        df = pd.read_csv(raw_data_path)

        expected_columns = [
            "customer_id",
            "age",
            "tenure",
            "spend_rate",
            "plan_type",
            "churn",
            "ticket_notes",
        ]
        df = validate_data(df, expected_columns)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Cleaned dataset saved to {output_path}")
        return df
    except Exception as e:
        logger.error(f"Error in make_dataset: {e}")
        raise


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    raw_data_path = os.path.join(project_root, "data", "raw", "customer_churn.csv")
    output_path = os.path.join(
        project_root, "data", "processed", "cleaned_customer_churn.csv"
    )
    make_dataset(raw_data_path, output_path)
