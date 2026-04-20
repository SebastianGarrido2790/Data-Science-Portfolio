import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shap import Explainer, summary_plot
import xgboost as xgb
import joblib

# Set base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load data
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

# Align predictions with test data
y_test = test_data["target"]
residuals_xgb = y_test - pred_xgb["prediction"]
residuals_prophet = y_test - pred_prophet["prediction"]


# Residual Analysis
def plot_residuals(residuals, model_name):
    plt.figure(figsize=(8, 4))
    sns.histplot(residuals, kde=True)
    plt.title(f"Residual Distribution - {model_name}")
    plt.xlabel("Residuals")
    plt.ylabel("Density")
    plt.savefig(
        os.path.join(
            base_dir, f"../../../reports/figures/residuals_{model_name.lower()}.png"
        )
    )
    plt.close()


plot_residuals(residuals_xgb, "XGBoost")
plot_residuals(residuals_prophet, "Prophet")

# Segment-Level Analysis by Borough
test_data = test_data.assign(
    residual_xgb=residuals_xgb, residual_prophet=residuals_prophet
)
segment_analysis = (
    test_data.groupby("Borough")
    .agg({"residual_xgb": ["mean", "std"], "residual_prophet": ["mean", "std"]})
    .reset_index()
)

# Save segment analysis
segment_output_path = os.path.join(
    base_dir, "../../../data/processed/segment_analysis.parquet"
)
segment_analysis.to_parquet(segment_output_path, index=False)
print(f"Segment analysis saved: {segment_output_path}")

# Explainability with SHAP (XGBoost)
xgb_model_path = os.path.join(base_dir, "../../models/xgboost_model.xgb")
model_xgb = xgb.XGBRegressor()
model_xgb.load_model(xgb_model_path)

X_test_xgb = test_data[
    [
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
]
explainer_xgb = Explainer(model_xgb, X_test_xgb)
shap_values_xgb = explainer_xgb(X_test_xgb)
summary_plot(shap_values_xgb, X_test_xgb, show=False)
plt.title("SHAP Summary Plot - XGBoost")
plt.savefig(os.path.join(base_dir, "../../../reports/figures/shap_summary_xgboost.png"))
plt.close()

# Save SHAP values for XGBoost
shap_df_xgb = pd.DataFrame(shap_values_xgb.values, columns=X_test_xgb.columns)
shap_output_path_xgb = os.path.join(
    base_dir, "../../../data/processed/shap_values_xgboost.parquet"
)
shap_df_xgb.to_parquet(shap_output_path_xgb, index=False)
print(f"SHAP values saved: {shap_output_path_xgb}")

# Explainability with SHAP (Prophet)
prophet_model_path = os.path.join(base_dir, "../../models/prophet_model.pkl")
model_prophet = joblib.load(prophet_model_path)

# Prepare Prophet input for SHAP as DataFrame with numeric conversion
X_test_prophet = test_data[["hour"]].rename(columns={"hour": "ds"})
X_test_prophet["ds"] = (
    X_test_prophet["ds"].astype(np.int64) // 10**9
)  # Convert to Unix timestamp in seconds


def prophet_predict(data):
    # Convert numeric timestamp back to datetime for Prophet prediction
    data_df = pd.DataFrame({"ds": pd.to_datetime(data["ds"], unit="s")})
    forecast = model_prophet.predict(data_df)
    return forecast["yhat"].values


explainer_prophet = Explainer(prophet_predict, X_test_prophet)
shap_values_prophet = explainer_prophet(X_test_prophet)
summary_plot(shap_values_prophet, X_test_prophet, show=False)
plt.title("SHAP Summary Plot - Prophet")
plt.savefig(os.path.join(base_dir, "../../../reports/figures/shap_summary_prophet.png"))
plt.close()

# Save SHAP values for Prophet
shap_df_prophet = pd.DataFrame(
    shap_values_prophet.values, columns=X_test_prophet.columns
)
shap_output_path_prophet = os.path.join(
    base_dir, "../../../data/processed/shap_values_prophet.parquet"
)
shap_df_prophet.to_parquet(shap_output_path_prophet, index=False)
print(f"SHAP values saved: {shap_output_path_prophet}")
