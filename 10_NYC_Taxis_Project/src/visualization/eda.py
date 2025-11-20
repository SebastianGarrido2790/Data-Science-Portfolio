import os
import pandas as pd
import numpy as np
from dask import dataframe as dd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import streamlit as st

# Set base directory
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "../../reports/figures/eda")
os.makedirs(output_dir, exist_ok=True)

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

# Prepare time-series data
tlc_data["tpep_pickup_datetime"] = pd.to_datetime(tlc_data["tpep_pickup_datetime"])
tlc_hourly = (
    tlc_data.groupby([pd.Grouper(key="tpep_pickup_datetime", freq="h"), "PULocationID"])
    .size()
    .reset_index(name="ride_count")
)
weather_data["date and time"] = pd.to_datetime(weather_data["date and time"])
merged_data = tlc_hourly.merge(
    weather_data, left_on="tpep_pickup_datetime", right_on="date and time", how="left"
)

# 1. Data Exploration
# Univariate Analysis
plt.figure(figsize=(10, 6))
sns.histplot(data=tlc_hourly, x="ride_count", bins=50)
plt.title("Distribution of Hourly Ride Counts")
plt.savefig(os.path.join(output_dir, "ride_count_distribution.png"))
plt.close()

# Bivariate Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_data, x="amount of precipitation", y="ride_count")
plt.title("Precipitation vs Ride Count")
plt.savefig(os.path.join(output_dir, "precipitation_vs_ride.png"))
plt.close()

# Time-Series Analysis
decomposition = seasonal_decompose(
    tlc_hourly.groupby("tpep_pickup_datetime").sum()["ride_count"],
    model="additive",
    period=24,
)
decomposition.plot()
plt.title("Seasonal Decomposition of Ride Counts")
plt.savefig(os.path.join(output_dir, "seasonal_decomposition.png"))
plt.close()

# Anomaly Detection
X_anomaly = tlc_hourly[["ride_count"]].values
iso_forest = IsolationForest(contamination=0.01, random_state=42)
outliers = iso_forest.fit_predict(X_anomaly)
tlc_hourly["outlier"] = outliers
outlier_count = (outliers == -1).sum()
print(f"Outliers Flagged: {outlier_count}")

# 2. Advanced Exploration Methods
# Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
tlc_zone_agg = tlc_hourly.groupby("PULocationID").mean().reset_index()
kmeans.fit(tlc_zone_agg[["ride_count"]])
tlc_zone_agg["cluster"] = kmeans.labels_
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=tlc_zone_agg, x="PULocationID", y="ride_count", hue="cluster", palette="deep"
)
plt.title("Zone Clusters by Ride Count")
plt.savefig(os.path.join(output_dir, "zone_clusters.png"))
plt.close()

# PCA on Weather
weather_numeric = weather_data[["temperature", "humidity", "wind speed"]].dropna()
pca = PCA(n_components=2)
weather_pca = pca.fit_transform(weather_numeric)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=weather_pca[:, 0], y=weather_pca[:, 1])
plt.title("PCA of Weather Features")
plt.savefig(os.path.join(output_dir, "weather_pca.png"))
plt.close()

# 3. Hypothesis Testing
# Rain increases demand
rain_data = merged_data.dropna(subset=["amount of precipitation", "ride_count"])
rain_effect = stats.ttest_ind(
    rain_data[rain_data["amount of precipitation"] > 0]["ride_count"],
    rain_data[rain_data["amount of precipitation"] == 0]["ride_count"],
)
print(f"Rain Effect t-test p-value: {rain_effect.pvalue}")

# Weekend boost
tlc_hourly["weekday"] = tlc_hourly["tpep_pickup_datetime"].dt.dayofweek
weekend_effect = stats.ttest_ind(
    tlc_hourly[tlc_hourly["weekday"] >= 5]["ride_count"],
    tlc_hourly[tlc_hourly["weekday"] < 5]["ride_count"],
)
print(f"Weekend Effect t-test p-value: {weekend_effect.pvalue}")

# 4. Feature Prioritization
X = merged_data[["ride_count", "amount of precipitation"]].dropna()
y = merged_data["ride_count"].shift(-1).dropna()
X = X.iloc[:-1].reset_index(drop=True)
y = y.iloc[:-1].reset_index(drop=True)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
print(f"Feature Importances: {dict(zip(X.columns, importances))}")

# Streamlit Dashboard
st.title("EDA Dashboard")
st.header("Ride Count Distribution")
st.image(os.path.join(output_dir, "ride_count_distribution.png"))
st.header("Precipitation vs Ride Count")
st.image(os.path.join(output_dir, "precipitation_vs_ride.png"))
st.header("Zone Clusters")
st.image(os.path.join(output_dir, "zone_clusters.png"))
