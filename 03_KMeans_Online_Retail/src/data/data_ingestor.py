import os
import zipfile
from abc import ABC, abstractmethod
import pandas as pd


# Define an abstract class for data ingestion
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Abstract method to ingest data from a file path.
        """
        pass


# Define a concrete class for ingesting data from a zip file containing Excel files
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Extracts data from a zip file, reads only the first sheet (Year 2009-2010)
        from the first Excel file found, and saves the DataFrame as a CSV in the interim folder.
        """
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check if the file is a zip file
        if not file_path.endswith(".zip"):
            raise ValueError(f"File is not a zip file: {file_path}")

        # Define the extraction path for the raw data (absolute path)
        extraction_path = os.path.join(os.path.dirname(file_path), "../../data/raw")
        extraction_path = os.path.abspath(extraction_path)
        os.makedirs(extraction_path, exist_ok=True)

        # Extract the zip file to the raw data directory
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extraction_path)

        # Search for Excel files (.xlsx or .xls) in the extraction directory
        extracted_files = os.listdir(extraction_path)
        excel_files = [f for f in extracted_files if f.endswith((".xlsx", ".xls"))]

        if len(excel_files) == 0:
            raise FileNotFoundError("No Excel file found in the extracted data.")

        # Use the first Excel file found (assumed to be the Year 2009-2010 sheet)
        excel_file_path = os.path.join(extraction_path, excel_files[0])

        # Read only the first sheet from the Excel file
        df = pd.read_excel(excel_file_path, sheet_name=0)

        # Define the interim folder path and ensure it exists
        interim_path = os.path.join(os.path.dirname(file_path), "../../data/interim")
        interim_path = os.path.abspath(interim_path)
        # Create the directory if it doesn't exist
        os.makedirs(interim_path, exist_ok=True)

        # Save the DataFrame as a CSV file in the interim folder
        output_file = os.path.join(interim_path, "online_retail_2009_2010.csv")
        df.to_csv(output_file, index=False)

        return df


class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """
        Returns the appropriate data ingestor based on the file extension.
        """
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")


# Example usage
if __name__ == "__main__":
    # Specify the zip file path (assuming it's placed in data/raw)
    file_path = os.path.join("data", "raw", "online+retail+ii.zip")
    file_extension = os.path.splitext(file_path)[1]

    # Get the appropriate data ingestor and ingest the data
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    df = data_ingestor.ingest(file_path)

    # Print the first few rows of the DataFrame
    print(df.head())
