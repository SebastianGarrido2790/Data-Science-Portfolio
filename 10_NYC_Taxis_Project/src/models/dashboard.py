import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import mean_absolute_error
from datetime import datetime

# Set base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load model, scaler, and data
xgb_model_path = os.path.join(base_dir, "../../models/xgboost_model.xgb")
scaler_path = os.path.join(base_dir, "../../models/scaler.joblib")
test_path = os.path.join(base_dir, "../../data/processed/test_features.parquet")
predictions_path = os.path.join(
    base_dir, "../../data/processed/predictions_xgboost.parquet"
)
zone_lookup_path = os.path.join(base_dir, "../../data/external/taxi_zone_lookup.csv")

model_xgb = xgb.XGBRegressor()
model_xgb.load_model(xgb_model_path)
scaler = joblib.load(scaler_path)
test_data = pd.read_parquet(test_path)
predictions = pd.read_parquet(predictions_path)["prediction"]
zone_lookup = pd.read_csv(zone_lookup_path)

# Verify and merge with zone lookup
if "Borough" not in test_data.columns or test_data["Borough"].isna().all():
    test_data = test_data.merge(
        zone_lookup, left_on="PULocationID", right_on="LocationID", how="left"
    )
test_data["Borough"] = test_data["Borough"].fillna("Unknown")
test_data["Zone"] = test_data["Zone"].fillna("Unknown")

# Prepare data
features = [
    "hour_of_day",
    "day_of_week",
    "temperature",
    "humidity",
    "wind speed",
    "amount of precipitation",
    "lag_1",
    "lag_2",
    "lag_3",
]
X_test = test_data[features].copy()
y_test = test_data["target"]
scaled_features = scaler.transform(X_test)
y_pred = model_xgb.predict(scaled_features)

# Calculate online MAE over time
test_data["prediction"] = y_pred
test_data["mae"] = np.abs(test_data["target"] - test_data["prediction"])
test_data["hour"] = pd.to_datetime(test_data["hour"])
mae_over_time = test_data.groupby(test_data["hour"].dt.date)["mae"].mean().reset_index()

# Offline MAE from training (simulated from 02_training_pipeline.py RMSE)
offline_rmse = 0.279274  # From training RMSE logged in MLflow
offline_mae = offline_rmse * 0.8  # Approximate MAE as 80% of RMSE for demonstration

# Merge with zone lookup for heatmap (already done above if needed)
manhattan_data = (
    test_data[test_data["Borough"] == "Manhattan"]
    .groupby("Zone")["prediction"]
    .mean()
    .reset_index()
)

# Streamlit dashboard
st.title("XGBoost Model Monitoring Dashboard")
st.header("Model Performance Metrics")

# Plot MAE over time
st.subheader("Mean Absolute Error (MAE) Over Time")
fig1, ax1 = plt.subplots()
ax1.plot(mae_over_time["hour"], mae_over_time["mae"], label="Online MAE")
ax1.axhline(y=offline_mae, color="r", linestyle="--", label="Offline MAE")
ax1.set_xlabel("Date")
ax1.set_ylabel("MAE")
ax1.legend()
plt.setp(ax1.get_xticklabels(), rotation=30)
st.pyplot(fig1)

# Plot Predictions vs Actuals
st.subheader("Predictions vs Actuals")
fig2, ax2 = plt.subplots()
ax2.scatter(y_test, y_pred, alpha=0.5)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
ax2.set_xlabel("Actual Ride Count")
ax2.set_ylabel("Predicted Ride Count")
st.pyplot(fig2)

# Heatmap of Demand Across Manhattan Zones
st.subheader("Demand Heatmap by Zone")
pivot_data = manhattan_data.pivot_table(
    values="prediction", index="Zone", aggfunc="mean"
).fillna(0)
fig3, ax3 = plt.subplots(figsize=(10, 8))
sns.heatmap(
    pivot_data,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    ax=ax3,
    cbar_kws={"label": "Average Predicted Demand"},
)
ax3.set_title("Average Predicted Demand by Manhattan Zone")
st.pyplot(fig3)

# Display basic statistics
st.subheader("Summary Statistics")
stats = {
    "Offline MAE": f"{offline_mae:.4f}",
    "Average Online MAE": f"{mae_over_time['mae'].mean():.4f}",
    "Data Points": len(test_data),
}
st.json(stats)

# Run the app
if __name__ == "__main__":
    st.write(f"Dashboard generated on {datetime.now().strftime('%Y-%m-%d %H:%M %Z')}")
