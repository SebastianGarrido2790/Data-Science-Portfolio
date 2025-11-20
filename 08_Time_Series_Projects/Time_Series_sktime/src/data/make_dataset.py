import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from src.data.data_ingestor import DataIngestorFactory
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/data_processing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def missing_data(input_data: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with total and percentage of nulls per column and data types."""
    total = input_data.isnull().sum()
    percent = input_data.isnull().sum() / input_data.isnull().count() * 100
    table = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
    types = [str(input_data[col].dtype) for col in input_data.columns]
    table["Types"] = types
    return table


def process_data(
    zip_path: str, csv_path: str, output_dir: str = "data/processed/subcategories"
) -> None:
    """Process raw data, perform feature engineering, and save to subcategories directory."""
    try:
        # Resolve paths to absolute paths
        zip_path = os.path.abspath(zip_path)
        csv_path = os.path.abspath(csv_path)
        output_dir = os.path.abspath(output_dir)

        logger.info(f"Resolved zip_path: {zip_path}")
        logger.info(f"Resolved csv_path: {csv_path}")
        logger.info(f"Resolved output_dir: {output_dir}")

        # Check if extracted CSV exists; if not, ingest from ZIP
        if os.path.exists(csv_path):
            logger.info(f"Using existing CSV file: {csv_path}")
            ingestor = DataIngestorFactory.get_data_ingestor(csv_path)
            df = ingestor.ingest(csv_path)
        else:
            logger.info(f"CSV not found, ingesting from: {zip_path}")
            ingestor = DataIngestorFactory.get_data_ingestor(zip_path)
            df = ingestor.ingest(zip_path)
        logger.info(f"Data ingested successfully, shape: {df.shape}")

        # Basic info and missing data analysis
        logger.info("Analyzing missing data")
        print(df.info())
        missing_df = missing_data(df)
        print(missing_df)

        # Rename columns: replace spaces and hyphens with underscores, and lowercase
        df.columns = df.columns.str.replace(" ", "_").str.replace("-", "_").str.lower()
        logger.info(f"Renamed columns: {list(df.columns)}")

        # Drop irrelevant column
        df = df.drop(columns=["postal_code"], errors="ignore")

        # Convert dates to datetime
        df["order_date"] = pd.to_datetime(
            df["order_date"], format="%d/%m/%Y", errors="coerce"
        )
        if df["order_date"].isnull().any():
            logger.warning("Some order dates failed to parse")
        logger.info(
            f"Date range: {min(df['order_date']):%Y-%m-%d} to {max(df['order_date']):%Y-%m-%d}"
        )

        # Category grouping and pivot
        logger.info("Aggregating sales by sub_category")
        agg_categories = (
            df.groupby(["order_date", "sub_category"])
            .agg({"sales": "sum"})
            .reset_index()
        )
        total_sales_df = agg_categories.pivot(
            index="order_date", columns="sub_category", values="sales"
        ).fillna(0)

        # Feature engineering: Add time-based features
        total_sales_df = total_sales_df.sort_index()
        total_sales_df.index = pd.to_datetime(total_sales_df.index)
        total_sales_df["year"] = total_sales_df.index.year
        total_sales_df["month"] = total_sales_df.index.month
        total_sales_df["day"] = total_sales_df.index.day
        total_sales_df["day_of_week"] = total_sales_df.index.dayofweek

        os.makedirs("reports/figures/trends", exist_ok=True)

        # Visualize each sub-category
        for column in total_sales_df.columns.drop(
            ["year", "month", "day", "day_of_week"]
        ):
            plt.figure()
            plt.plot(total_sales_df.index, total_sales_df[column])
            plt.title(f"Sales Trend - {column}")
            plt.xlabel("Date")
            plt.ylabel("Sales")
            plt.savefig(os.path.join("reports/figures/trends", f"{column}_trend.png"))
            plt.close()

        # Outlier removal and create prediction DataFrames
        logger.info("Removing outliers and creating prediction DataFrames")
        prediction_df_list = []
        for column in total_sales_df.columns.drop(
            ["year", "month", "day", "day_of_week"]
        ):
            df_clean = total_sales_df[[column]].copy()
            z = np.abs(stats.zscore(df_clean[column]))
            outlier_index = np.where(z > 2)[0]
            logger.info(f"Dropping {len(outlier_index)} outliers for {column}")
            df_clean.drop(index=df_clean.index[outlier_index], inplace=True)
            prediction_df_list.append(df_clean)

        # Save intermediate datasets
        os.makedirs("data/interim", exist_ok=True)
        total_sales_df.to_csv("data/interim/total_sales.csv")
        df_agg = df.groupby("order_date")["sales"].sum().reset_index()
        df_agg.to_csv("data/interim/sales_ts.csv", index=False)
        logger.info("Intermediate data saved to interim directory")

        # Save processed data
        os.makedirs(output_dir, exist_ok=True)
        for i, df_pred in enumerate(prediction_df_list):
            df_pred.to_csv(os.path.join(output_dir, f"subcategory_{i}.csv"))
        logger.info(f"Processed data saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise


if __name__ == "__main__":
    # Define paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    zip_path = os.path.join(project_root, "data", "raw", "train.csv.zip")
    csv_path = os.path.join(project_root, "data", "raw", "train.csv")
    process_data(zip_path, csv_path)

pd.read_csv("../../data/interim/sales_ts.csv").info()
pd.read_csv("../../data/interim/total_sales.csv").info()
