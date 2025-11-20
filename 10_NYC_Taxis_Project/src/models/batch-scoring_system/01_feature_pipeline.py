import os
import pandas as pd
from dask import dataframe as dd
from sklearn.preprocessing import StandardScaler
import mlflow

# Set base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Sub-step 1: Load raw data with Dask
tlc_data = dd.read_parquet(
    os.path.join(base_dir, "../../data/raw/yellow_tripdata_2019-*.parquet")
)
weather_data = pd.read_csv(
    os.path.join(base_dir, "../../data/external/nyc_weather.csv")
)
zone_data = pd.read_csv(
    os.path.join(base_dir, "../../data/external/taxi_zone_lookup.csv")
)


# Sub-step 2: Clean and validate data
def clean_data(tlc_df, weather_df, zone_df):
    # TLC: Filter 2019, remove invalid trips, drop missing passenger_count
    tlc_df = tlc_df[
        (tlc_df["tpep_pickup_datetime"].dt.year == 2019)
        & (tlc_df["trip_distance"] > 0)
        & (tlc_df["fare_amount"] > 0)
        & (~tlc_df["passenger_count"].isna())
    ]

    # Weather: Convert timestamps, impute missing values
    weather_df["date and time"] = pd.to_datetime(weather_df["date and time"])
    weather_df["amount of precipitation"] = pd.to_numeric(
        weather_df["amount of precipitation"], errors="coerce"
    )
    weather_df["amount of precipitation"] = weather_df[
        "amount of precipitation"
    ].fillna(weather_df["amount of precipitation"].median())
    weather_df["cloud cover"] = weather_df["cloud cover"].fillna(
        weather_df["cloud cover"].mode()[0]
    )

    # Zone: Fill missing values
    zone_df = zone_df.fillna("Unknown")

    return tlc_df, weather_df, zone_df


tlc_cleaned, weather_cleaned, zone_cleaned = clean_data(
    tlc_data, weather_data, zone_data
)

# Sub-step 3: Engineer features
# Normalization and Scaling on original TLC data
numeric_features = ["trip_distance", "fare_amount"]
scaler = StandardScaler()
tlc_scaled = tlc_cleaned.map_partitions(
    lambda df: pd.DataFrame(
        scaler.fit_transform(df[numeric_features]),
        columns=numeric_features,
        index=df.index,
    ).join(df.drop(columns=numeric_features))
).reset_index(drop=True)

# Sub-step 4: Aggregate into time-series, define (features, target)
# Aggregate hourly counts
hourly_counts = (
    tlc_scaled.groupby(
        [
            tlc_scaled["tpep_pickup_datetime"].dt.floor("h").rename("hour"),
            "PULocationID",
        ]
    )
    .size()
    .to_frame(name="ride_count")
    .reset_index()
    .compute()
)

# Merge with zone and weather data
hourly_data = hourly_counts.merge(
    zone_cleaned, left_on="PULocationID", right_on="LocationID", how="left"
)
hourly_data = hourly_data.merge(
    weather_cleaned,
    left_on=hourly_data["hour"].dt.floor("h"),
    right_on="date and time",
    how="left",
)

# Feature Engineering
hourly_data["hour_of_day"] = hourly_data["hour"].dt.hour
hourly_data["day_of_week"] = hourly_data["hour"].dt.dayofweek
hourly_data["month"] = hourly_data["hour"].dt.month
weather_features = ["temperature", "humidity", "wind speed", "amount of precipitation"]
for feature in weather_features:
    hourly_data[feature] = hourly_data[feature].fillna(hourly_data[feature].median())
zone_features = ["Borough", "service_zone"]
for feature in zone_features:
    hourly_data[feature] = hourly_data[feature].fillna("Unknown")
hourly_data = hourly_data.sort_values("hour")
hourly_data["lag_1"] = hourly_data.groupby("PULocationID")["ride_count"].shift(1)
hourly_data["lag_2"] = hourly_data.groupby("PULocationID")["ride_count"].shift(2)
hourly_data["lag_3"] = hourly_data.groupby("PULocationID")["ride_count"].shift(3)
hourly_data[["lag_1", "lag_2", "lag_3"]] = hourly_data[
    ["lag_1", "lag_2", "lag_3"]
].fillna(0)
numeric_features_agg = ["ride_count", "lag_1", "lag_2", "lag_3"]
scaler_agg = StandardScaler()
hourly_data[numeric_features_agg] = scaler_agg.fit_transform(
    hourly_data[numeric_features_agg]
)
hourly_data["target"] = hourly_data.groupby("PULocationID")["ride_count"].shift(-1)
hourly_data = hourly_data.dropna(subset=["target"])

# Sub-step 5: Split into train/test, save to Feature Store
# Time-based split
train_data = hourly_data[hourly_data["month"].isin([1, 2])]
test_data = hourly_data[hourly_data["month"] == 3]

# Save splits
train_path = os.path.join(base_dir, "../../data/processed/train_features.parquet")
test_path = os.path.join(base_dir, "../../data/processed/test_features.parquet")
train_data.to_parquet(train_path, index=False)
test_data.to_parquet(test_path, index=False)

# Feature Store with MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # Adjust URI as needed
with mlflow.start_run(run_name="feature_pipeline_run"):
    mlflow.log_param("dataset", "features_2019")
    mlflow.log_param("train_path", train_path)
    mlflow.log_param("test_path", test_path)
    mlflow.log_artifact(train_path)
    mlflow.log_artifact(test_path)
