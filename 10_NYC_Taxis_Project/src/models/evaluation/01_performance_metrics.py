import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Set base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load test data and predictions
test_path = os.path.join(base_dir, "../../../data/processed/test_features.parquet")
test_data = pd.read_parquet(test_path)
pred_xgb_path = os.path.join(
    base_dir, "../../../data/processed/predictions_xgboost.parquet"
)
pred_prophet_path = os.path.join(
    base_dir, "../../../data/processed/predictions_prophet.parquet"
)
pred_xgb = pd.read_parquet(pred_xgb_path)
pred_prophet = pd.read_parquet(pred_prophet_path)

# Align predictions with test data target
y_test = test_data["target"]


# Calculate performance metrics
def calculate_metrics(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"model": model_name, "RMSE": rmse, "MAE": mae}


metrics_xgb = calculate_metrics(y_test, pred_xgb["prediction"], "XGBoost")
metrics_prophet = calculate_metrics(y_test, pred_prophet["prediction"], "Prophet")


# Business KPI mapping
def map_to_business_kpi(metrics):
    # Assume RMSE reduction correlates with cost savings (e.g., $100 per 0.1 RMSE reduction)
    cost_savings = (0.5 - metrics["RMSE"]) * 1000 if metrics["RMSE"] < 0.5 else 0
    ride_completion_rate = (
        0.95 + (0.05 * (0.5 - metrics["RMSE"]) / 0.5) if metrics["RMSE"] < 0.5 else 0.95
    )
    return {
        "cost_savings_usd": cost_savings,
        "ride_completion_rate": ride_completion_rate,
    }


kpi_xgb = map_to_business_kpi(metrics_xgb)
kpi_prophet = map_to_business_kpi(metrics_prophet)

# Combine results
results = pd.DataFrame([metrics_xgb, metrics_prophet])
results["cost_savings_usd"] = [
    kpi_xgb["cost_savings_usd"],
    kpi_prophet["cost_savings_usd"],
]
results["ride_completion_rate"] = [
    kpi_xgb["ride_completion_rate"],
    kpi_prophet["ride_completion_rate"],
]

# Save results
output_path = os.path.join(
    base_dir, "../../../data/processed/performance_metrics.parquet"
)
results.to_parquet(output_path, index=False)
print(f"Performance metrics saved: {output_path}")
