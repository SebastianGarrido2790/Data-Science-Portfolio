import os
import pandas as pd
from dask import dataframe as dd

# Inventory data sources (adjust paths relative to script location)
base_dir = os.path.dirname(os.path.abspath(__file__))
data_sources = {
    "tlc_jan": os.path.join(base_dir, "../../data/raw/yellow_tripdata_2019-01.parquet"),
    "tlc_feb": os.path.join(base_dir, "../../data/raw/yellow_tripdata_2019-02.parquet"),
    "tlc_mar": os.path.join(base_dir, "../../data/raw/yellow_tripdata_2019-03.parquet"),
    "weather": os.path.join(base_dir, "../../data/external/nyc_weather.csv"),
    "zones": os.path.join(base_dir, "../../data/external/taxi_zone_lookup.csv"),
}

# Load and validate existence
for name, path in data_sources.items():
    assert os.path.exists(path), f"{name} data not found at {path}"
    print(f"{name} found at {path}")

# Load data with Dask for scalability
tlc_data = dd.read_parquet(
    [os.path.join(base_dir, "../../data/raw/yellow_tripdata_2019-*.parquet")]
)
weather_data = pd.read_csv(data_sources["weather"])
zone_data = pd.read_csv(data_sources["zones"])

# Basic metadata
print("TLC Data Shape:", tlc_data.shape)
print("Weather Data Shape:", weather_data.shape)
print("Zone Data Shape:", zone_data.shape)
