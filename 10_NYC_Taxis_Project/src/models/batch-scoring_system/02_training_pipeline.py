import os
import pandas as pd
import xgboost as xgb
from prophet import Prophet
import mlflow
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from optuna import create_study
import joblib

# Set base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load data
train_path = os.path.join(base_dir, "../../data/processed/train_features.parquet")
val_path = os.path.join(base_dir, "../../data/processed/val_data.parquet")
train_data = pd.read_parquet(train_path)
val_data = pd.read_parquet(val_path)

# Prepare features and target
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
X_train = train_data[features]
y_train = train_data["target"]
X_val = val_data[features]
y_val = val_data["target"]

# Model Training with XGBoost
mlflow.set_tracking_uri("http://localhost:5000")
with mlflow.start_run(run_name="xgboost_training"):
    # Hyperparameter optimization with Optuna
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        }
        model = xgb.XGBRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return mean_squared_error(y_val, y_pred)

    study = create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    best_params = study.best_params

    # Train final model
    final_model = xgb.XGBRegressor(**best_params, random_state=42)
    final_model.fit(X_train, y_train)

    # Evaluate
    y_pred = final_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mlflow.log_params(best_params)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(final_model, "model")

    # Save model as standalone .xgb file
    xgb_model_path = os.path.join(base_dir, "../../models/xgboost_model.xgb")
    final_model.save_model(xgb_model_path)
    print(f"XGBoost RMSE: {rmse}")
    print(f"XGBoost model saved: {xgb_model_path}")

# Prophet for time-series
prophet_model_path = os.path.join(base_dir, "../../models/prophet_model.pkl")
prophet_data = train_data[["hour", "target"]].rename(
    columns={"hour": "ds", "target": "y"}
)
model = Prophet(
    yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True
)
model.fit(prophet_data)
future = val_data[["hour"]].rename(columns={"hour": "ds"})
forecast = model.predict(future)
rmse = np.sqrt(mean_squared_error(val_data["target"], forecast["yhat"]))

# Save Prophet model as standalone pickle file
joblib.dump(model, prophet_model_path)
print(f"Prophet RMSE: {rmse}")
print(f"Prophet model saved: {prophet_model_path}")
