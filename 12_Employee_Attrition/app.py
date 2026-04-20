from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import joblib
import os
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r") as f:
        return f.read()


mlflow.set_tracking_uri(
    r"file:///C:/Users/.../12_Employee_Attrition/src/features/mlruns"
)
try:
    lr_model = mlflow.sklearn.load_model("models:/LogisticRegression/2")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

try:
    scaler = joblib.load(r"C:\Users\...\12_Employee_Attrition\models\scaler.pkl")
    logger.info("Scaler loaded successfully")
except Exception as e:
    logger.error(f"Failed to load scaler: {e}")
    raise


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Employee Attrition Prediction API. Use /predict to make predictions."
    }


# Define input data structure
class EmployeeData(BaseModel):
    Age: int
    DailyRate: float
    DistanceFromHome: int
    Education: int
    EmployeeNumber: int
    EnvironmentSatisfaction: int
    HourlyRate: float
    JobInvolvement: int
    JobLevel: int
    JobSatisfaction: int
    MonthlyIncome: float
    MonthlyRate: float
    NumCompaniesWorked: int
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int
    BusinessTravel: str
    Department: str
    EducationField: str
    Gender: str
    JobRole: str
    MaritalStatus: str
    OverTime: str


# Preprocessing function
def preprocess_data(data):
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

    data_df = pd.DataFrame([data.dict()])
    data_df["Gender"] = data_df["Gender"].map({"Male": 1, "Female": 0})
    data_df["OverTime"] = data_df["OverTime"].map({"Yes": 1, "No": 0})

    data_df["TenureRatio"] = data_df["YearsAtCompany"] / data_df[
        "TotalWorkingYears"
    ].replace(0, 1)
    data_df["SatisfactionScore"] = data_df[
        ["EnvironmentSatisfaction", "JobSatisfaction", "RelationshipSatisfaction"]
    ].mean(axis=1)
    data_df["AgeGroup"] = pd.cut(
        data_df["Age"], bins=[0, 30, 40, 100], labels=["lt30", "30-40", "gt40"]
    ).astype(str)
    data_df["IncomeToLevelRatio"] = data_df["MonthlyIncome"] / data_df["JobLevel"]
    data_df["LongCommute"] = (data_df["DistanceFromHome"] > 10).astype(int)

    num_data = data_df[numerical_cols]
    cat_data = data_df[categorical_cols]

    num_data_scaled = scaler.transform(num_data)
    num_data = pd.DataFrame(
        num_data_scaled, columns=numerical_cols, index=data_df.index
    )

    cat_data = pd.get_dummies(cat_data, columns=categorical_cols, drop_first=True)
    cat_data.columns = [
        col.replace(" ", "_").replace("&", "_") for col in cat_data.columns
    ]

    data_processed = pd.concat([num_data, cat_data], axis=1)
    expected_cols = pd.read_csv("./data/processed/X_train.csv").columns.tolist()
    for col in expected_cols:
        if col not in data_processed.columns:
            data_processed[col] = 0
    data_processed = data_processed[expected_cols]

    return data_processed


# Prediction endpoint
@app.post("/predict")
async def predict(employee: EmployeeData):
    try:
        processed_data = preprocess_data(employee)
        proba = lr_model.predict_proba(processed_data)[:, 1][0]
        prediction = 1 if proba >= 0.45 else 0
        logger.info(
            f"Prediction for Employee {employee.EmployeeNumber}: {prediction} (Proba: {proba})"
        )
        return {
            "EmployeeNumber": employee.EmployeeNumber,
            "Attrition_Probability": proba,
            "Attrition_Prediction": prediction,
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
