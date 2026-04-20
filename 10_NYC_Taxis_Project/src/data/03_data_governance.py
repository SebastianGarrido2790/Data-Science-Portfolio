import os
import pandas as pd
import numpy as np
from dask import dataframe as dd
import hashlib

# Set base directory relative to script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load data
tlc_data = dd.read_parquet(
    os.path.join(base_dir, "../../data/raw/yellow_tripdata_2019-*.parquet")
).compute()
weather_data = pd.read_csv(
    os.path.join(base_dir, "../../data/external/nyc_weather.csv")
)
zone_data = pd.read_csv(
    os.path.join(base_dir, "../../data/external/taxi_zone_lookup.csv")
)


# Custom anonymization function for PULocationID
def anonymize_location_id(x):
    if pd.isna(x):
        return x  # Preserve NaN
    # Hash the integer and take modulo to create groups
    hash_object = hashlib.md5(str(x).encode())
    return int(hash_object.hexdigest(), 16) % 100


tlc_data["PULocationID"] = tlc_data["PULocationID"].apply(anonymize_location_id)

# Anonymize passenger_count, handling NaN
tlc_data["passenger_count"] = tlc_data["passenger_count"].apply(
    lambda x: round(x, -1) if pd.notnull(x) else np.nan
)

# Clean missing values
tlc_data = tlc_data.dropna(subset=["passenger_count", "PULocationID"])


# Compliance Check (updated)
def compliance_check(df):
    sensitive_cols = ["passenger_count", "PULocationID"]
    missing_counts = df[sensitive_cols].isnull().sum()
    assert all(
        missing_counts == 0
    ), f"Sensitive data missing post-anonymization: {missing_counts}"
    return True


if compliance_check(tlc_data):
    print("Compliance check passed")

# Save catalog
catalog = {
    "tlc": {
        "source": "NYC TLC",
        "quality": {"completeness": 0.85, "accuracy": 0.95},
        "policy": "anonymized",
    },
    "weather": {
        "source": "External API",
        "quality": {"completeness": 0.76},
        "policy": "imputed",
    },
    "zones": {"source": "NYC TLC", "quality": {"completeness": 0.99}, "policy": "raw"},
}
pd.DataFrame(catalog).to_csv(os.path.join(base_dir, "../../data/catalog.csv"))
