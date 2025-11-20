import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# Set MLflow tracking URI (adjust path to match your setup)
mlflow.set_tracking_uri(
    r"file:///C:/Users/.../12_Employee_Attrition/src/features/mlruns"
)


# Load models from MLflow
def load_model(model_name, version):
    client = mlflow.tracking.MlflowClient()
    model_uri = f"models:/{model_name}/{version}"
    if "XGBoost" in model_name:
        model = mlflow.xgboost.load_model(model_uri)
    else:
        model = mlflow.sklearn.load_model(model_uri)
    return model


# Preprocessing function (mirrors feature_engineering.py)
def preprocess_data(data):
    # Define numerical and categorical columns (same as feature_engineering.py)
    numerical_cols = [
        "Age",
        "DailyRate",
        "DistanceFromHome",
        "Education",
        "EmployeeNumber",
        "EnvironmentSatisfaction",
        "HourlyRate",
        "JobInvolvement",
        "JobLevel",
        "JobSatisfaction",
        "MonthlyIncome",
        "MonthlyRate",
        "NumCompaniesWorked",
        "PercentSalaryHike",
        "PerformanceRating",
        "RelationshipSatisfaction",
        "StockOptionLevel",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "WorkLifeBalance",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
        "TenureRatio",
        "SatisfactionScore",
        "IncomeToLevelRatio",
        "LongCommute",
    ]
    categorical_cols = [
        "BusinessTravel",
        "Department",
        "EducationField",
        "Gender",
        "JobRole",
        "MaritalStatus",
        "OverTime",
        "AgeGroup",
    ]

    # Drop irrelevant columns
    data = data.drop(
        columns=["EmployeeCount", "Over18", "StandardHours"], errors="ignore"
    )

    # Convert categorical columns to binary/numeric
    data["Attrition"] = (
        data["Attrition"].map({"Yes": 1, "No": 0})
        if "Attrition" in data.columns
        else None
    )
    data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
    data["OverTime"] = data["OverTime"].map({"Yes": 1, "No": 0})

    # Feature engineering (aligned with feature_engineering.py)
    # Tenure Ratio
    data["TenureRatio"] = data["YearsAtCompany"] / data["TotalWorkingYears"].replace(
        0, 1
    )

    # Satisfaction Score
    data["SatisfactionScore"] = data[
        ["EnvironmentSatisfaction", "JobSatisfaction", "RelationshipSatisfaction"]
    ].mean(axis=1)

    # Age Bins
    data["AgeGroup"] = pd.cut(
        data["Age"], bins=[0, 30, 40, 100], labels=["lt30", "30-40", "gt40"]
    ).astype(str)

    # Income-to-Level Ratio
    data["IncomeToLevelRatio"] = data["MonthlyIncome"] / data["JobLevel"]

    # Long Commute Flag
    data["LongCommute"] = (data["DistanceFromHome"] > 10).astype(int)

    # Split data into numerical and categorical parts
    num_data = data[numerical_cols]
    cat_data = data[categorical_cols]

    # Load the scaler used during training and apply to numerical features only
    scaler = joblib.load("../../models/scaler.pkl")
    num_data_scaled = scaler.transform(num_data)
    num_data = pd.DataFrame(num_data_scaled, columns=numerical_cols, index=data.index)

    # One-hot encode categorical variables
    cat_data = pd.get_dummies(cat_data, columns=categorical_cols, drop_first=True)

    # Clean feature names to match feature_engineering.py (remove special characters)
    cat_data.columns = [
        col.replace(" ", "_").replace("&", "_") for col in cat_data.columns
    ]

    # Combine numerical and categorical data
    data_processed = pd.concat([num_data, cat_data], axis=1)

    # Ensure all expected columns are present (match training data)
    expected_cols = pd.read_csv("../../data/processed/X_train.csv").columns.tolist()
    for col in expected_cols:
        if col not in data_processed.columns:
            data_processed[col] = 0
    data_processed = data_processed[expected_cols]

    return data_processed


# Prediction function
def make_predictions(model, data, threshold, model_name):
    # Predict probabilities
    proba = model.predict_proba(data)[:, 1]
    # Apply threshold
    predictions = (proba >= threshold).astype(int)
    return proba, predictions


# Main prediction script
def main():
    # Create output directory if it doesn't exist
    os.makedirs("../../data/predictions", exist_ok=True)

    # Drop the 'Attrition' column
    df = pd.read_csv("../../data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df = df.drop("Attrition", axis=1)
    # Save the modified dataset to a new CSV file
    new_csv_path = "../../data/interim/new_employees.csv"
    df.to_csv(new_csv_path, index=False)

    # Load new data
    new_data = pd.read_csv("../../data/interim/new_employees.csv")
    employee_ids = new_data["EmployeeNumber"].copy()  # Store IDs for output

    # Preprocess data
    print("Preprocessing new data...")
    processed_data = preprocess_data(new_data.copy())

    # Load models
    print("Loading models...")
    lr_model = load_model("LogisticRegression", "2")
    xgb_model = load_model("XGBoost", "2")

    # Make predictions
    print("Making predictions...")
    lr_proba, lr_pred = make_predictions(
        lr_model, processed_data, threshold=0.4, model_name="LogisticRegression"
    )
    xgb_proba, xgb_pred = make_predictions(
        xgb_model, processed_data, threshold=0.3, model_name="XGBoost"
    )

    # Combine results
    results = pd.DataFrame(
        {
            "EmployeeNumber": employee_ids,
            "LogisticRegression_Proba": lr_proba,
            "LogisticRegression_Pred": lr_pred,
            "XGBoost_Proba": xgb_proba,
            "XGBoost_Pred": xgb_pred,
        }
    )

    # Save predictions
    output_path = "../../data/predictions/predictions.csv"
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    # Summary
    print("\nPrediction Summary:")
    print(
        f"LogisticRegression (Threshold=0.4): {lr_pred.sum()} predicted attritions out of {len(lr_pred)} employees"
    )
    print(
        f"XGBoost (Threshold=0.3): {xgb_pred.sum()} predicted attritions out of {len(xgb_pred)} employees"
    )
    print("\nSample Predictions (first 10 rows):")
    print(results.head(10))


if __name__ == "__main__":
    main()
