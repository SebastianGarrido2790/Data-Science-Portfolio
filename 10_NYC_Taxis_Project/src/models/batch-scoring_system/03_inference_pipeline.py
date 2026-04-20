import os
import pandas as pd
import mlflow
import mlflow.tracking

# Set base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load test data
test_path = os.path.join(base_dir, "../../data/processed/test_features.parquet")
test_data = pd.read_parquet(test_path)

# Prepare features and handle Prophet-specific requirements
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

# Determine model name from MLflow run metadata
run_id = ""  # Specify the run ID
client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)
model_flavor = run.data.tags.get(
    "flavor", "unknown"
)  # Fallback to 'unknown' if not set
if model_flavor == "unknown":
    run_name = run.data.tags.get("mlflow.runName", "unknown")
    model_flavor = (
        "xgboost"
        if "xgboost" in run_name.lower()
        else "prophet" if "prophet" in run_name.lower() else "unknown"
    )

# Load model from MLflow
model_uri = f"runs:/{run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)

# Adjust input for Prophet model
if model_flavor == "prophet":
    # Prophet requires 'ds' and 'y' columns; use 'hour' as 'ds' and target for prediction context
    prophet_input = test_data[["hour"]].rename(columns={"hour": "ds"})
    prophet_input["y"] = test_data["target"]  # Optional: include target if needed
    predictions = model.predict(prophet_input)
    # Extract 'yhat' as the prediction value
    predictions = predictions["yhat"]
else:
    # For other models (e.g., XGBoost), use the feature set
    predictions = model.predict(X_test)

# Save predictions with model name
output_path = os.path.join(
    base_dir, f"../../data/processed/predictions_{model_flavor}.parquet"
)
pd.DataFrame({"prediction": predictions}, index=test_data.index).to_parquet(
    output_path, index=True
)
print(f"Predictions saved: {output_path}")
