import os
import pandas as pd
import numpy as np
from dask import dataframe as dd
from scipy import stats
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

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


# Profiling
def profile_data(df, name):
    print(f"{name} Summary:")
    print(df.describe())
    print(f"Missing Values: {df.isnull().sum()}")
    return df


profile_data(tlc_data, "TLC")
profile_data(weather_data, "Weather")
profile_data(zone_data, "Zones")

# Cleaning
tlc_data = tlc_data[(tlc_data["trip_distance"] > 0) & (tlc_data["fare_amount"] > 0)]

# Convert and impute precipitation
weather_data["amount of precipitation"] = pd.to_numeric(
    weather_data["amount of precipitation"], errors="coerce"
)
weather_data.loc[:, "amount of precipitation"] = weather_data[
    "amount of precipitation"
].fillna(weather_data["amount of precipitation"].median())

# Temporal Validation
tlc_data["tpep_pickup_datetime"] = pd.to_datetime(tlc_data["tpep_pickup_datetime"])
tlc_data = tlc_data[tlc_data["tpep_pickup_datetime"].dt.year == 2019]  # Filter to 2019
decomposition = seasonal_decompose(
    tlc_data.groupby(pd.Grouper(key="tpep_pickup_datetime", freq="D")).size(),
    model="additive",
)
decomposition.plot()
plt.title("Seasonal Decomposition of Ride Counts")
plt.savefig(os.path.join(base_dir, "../../reports/figures/seasonal_decomp.png"))

# Anomaly Detection
X = tlc_data[["trip_distance", "fare_amount"]].values
iso_forest = IsolationForest(contamination=0.01, random_state=42)
outliers = iso_forest.fit_predict(X)
tlc_data["outlier"] = outliers
print("Outliers Flagged:", (outliers == -1).sum())
