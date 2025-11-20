import os
import zipfile
from abc import ABC, abstractmethod
import pandas as pd
import logging
from typing import Optional

# Configure logging for debugging and tracking
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/data_ingestion.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Abstract base class for data ingestion
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Ingest data from a file path and return a pandas DataFrame."""
        pass


# Concrete class for ingesting CSV files
class CSVDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Ingest data from a CSV file and return a pandas DataFrame."""
        logger.info(f"Ingesting CSV file: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.endswith(".csv"):
            logger.error(f"File is not a CSV: {file_path}")
            raise ValueError(f"File is not a CSV file: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Successfully ingested CSV file: {file_path}, shape: {df.shape}")
        return df


# Concrete class for ingesting JSON files
class JSONDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Ingest data from a JSON file and return a pandas DataFrame."""
        logger.info(f"Ingesting JSON file: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.endswith(".json"):
            logger.error(f"File is not a JSON: {file_path}")
            raise ValueError(f"File is not a JSON file: {file_path}")
        df = pd.read_json(file_path, orient="records")
        logger.info(f"Successfully ingested JSON file: {file_path}, shape: {df.shape}")
        return df


# Concrete class for ingesting Parquet files
class ParquetDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Ingest data from a Parquet file and return a pandas DataFrame."""
        logger.info(f"Ingesting Parquet file: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.endswith(".parquet"):
            logger.error(f"File is not a Parquet: {file_path}")
            raise ValueError(f"File is not a Parquet file: {file_path}")
        df = pd.read_parquet(file_path)
        logger.info(
            f"Successfully ingested Parquet file: {file_path}, shape: {df.shape}"
        )
        return df


# Concrete class for ingesting ZIP files (containing a single CSV)
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Ingest data from a ZIP file containing a CSV and return a pandas DataFrame."""
        logger.info(f"Ingesting ZIP file: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.endswith(".zip"):
            logger.error(f"File is not a ZIP: {file_path}")
            raise ValueError(f"File is not a ZIP file: {file_path}")

        # Define extraction path (relative to the script's location)
        extraction_path = os.path.join(os.path.dirname(file_path), "../../data/raw")
        extraction_path = os.path.abspath(extraction_path)
        os.makedirs(extraction_path, exist_ok=True)

        # Extract the ZIP file
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extraction_path)
        logger.info(f"Extracted ZIP file to: {extraction_path}")

        # Find the extracted CSV file
        extracted_files = os.listdir(extraction_path)
        csv_files = [f for f in extracted_files if f.endswith(".csv")]
        if not csv_files:
            logger.error("No CSV file found in the extracted ZIP data")
            raise FileNotFoundError("No CSV file found in the extracted data")
        if len(csv_files) > 1:
            logger.error("Multiple CSV files found in the extracted ZIP data")
            raise ValueError("Multiple CSV files found in the extracted data")

        # Ingest the CSV file
        csv_file_path = os.path.join(extraction_path, csv_files[0])
        df = CSVDataIngestor().ingest(csv_file_path)
        logger.info(f"Successfully ingested CSV from ZIP: {csv_file_path}")
        return df


# Factory class to select the appropriate ingestor
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_path: str) -> DataIngestor:
        """Return the appropriate DataIngestor based on the file extension."""
        file_extension = os.path.splitext(file_path)[1].lower()
        logger.info(f"Selecting ingestor for file extension: {file_extension}")

        ingestors = {
            ".csv": CSVDataIngestor(),
            ".json": JSONDataIngestor(),
            ".parquet": ParquetDataIngestor(),
            ".zip": ZipDataIngestor(),
        }

        ingestor = ingestors.get(file_extension)
        if not ingestor:
            logger.error(f"Unsupported file extension: {file_extension}")
            raise ValueError(f"Unsupported file extension: {file_extension}")
        return ingestor


# Example usage
if __name__ == "__main__":
    # Example file paths for different formats
    file_paths = [
        "data/raw/train.csv.zip",  # Keep only existing file for now
        # "data/raw/sales_data.csv",
        # "data/raw/sales_data.json",
        # "data/raw/sales_data.parquet",
    ]

    for file_path in file_paths:
        try:
            # Get the appropriate ingestor
            ingestor = DataIngestorFactory.get_data_ingestor(file_path)
            # Ingest the data
            df = ingestor.ingest(file_path)
            logger.info(f"Data ingested successfully: {file_path}")
            print(df.head())
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {str(e)}")
