## Step 7.1: Real-Time API Setup with FastAPI

**Objective**: Set up a real-time API using FastAPI for potential future HR needs, allowing on-demand predictions.

#### Design
- **Endpoint**: `/predict` accepts a JSON payload with employee data and returns predictions.
- **Scalability**: Use an ASGI server (Uvicorn) with Docker for deployment.
- **Security**: Add basic authentication (e.g., API key) to restrict access.
- **Frontend**: Serve a simple HTML page (`index.html`) for user-friendly interaction.

- **Script Explanation** (`app.py`):
  - Uses FastAPI to create a `/predict` endpoint accepting `EmployeeData` JSON.
  - Preprocesses single employee data, applies the optimized threshold (0.45), and returns probability and prediction.
  - Serves `index.html` at the root endpoint (`/`) for a web interface, mounted via `StaticFiles`.
  - Runs with Uvicorn for local testing.
  - Loads the MLflow model (`LogisticRegression/2`) and scaler (`models/scaler.pkl`) for predictions.
  - Uses `uv` for dependency management to ensure consistent environments.

#### Implementation Steps

1. **Dependencies with `uv`**:
   - Use `uv` to manage dependencies via `pyproject.toml`.
   - Example `pyproject.toml`:
     ```toml
     [project]
     name = "employee-attrition"
     version = "0.1.0"
     dependencies = [
         "pandas>=2.2.2",
         "numpy>=1.26.4",
         "scikit-learn>=1.5.0",
         "mlflow>=2.12.2",
         "xgboost>=2.0.3",
         "joblib>=1.4.2",
         "fastapi>=0.110.0",
         "uvicorn>=0.29.0",
     ]

     [build-system]
     requires = ["hatchling"]
     build-backend = "hatchling.build"

     [tool.hatch.build.targets.wheel]
     packages = ["app.py"]
     ```
   - Install dependencies:
     ```
     uv sync
     ```

2. **Run Locally**:
   - Start the FastAPI server:
     ```
     uv run uvicorn app:app --host 127.0.0.1 --port 8000 --log-level debug
     ```
   - Test the root endpoint:
     ```
     curl http://localhost:8000
     ```
     - Response: The `index.html` page is served, providing a form to input employee data.
   - Test the `/predict` endpoint:
     ```
     curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"Age": 35, "DailyRate": 1000, "DistanceFromHome": 5, "Education": 3, "EmployeeNumber": 1001, "EnvironmentSatisfaction": 3, "HourlyRate": 50, "JobInvolvement": 3, "JobLevel": 2, "JobSatisfaction": 4, "MonthlyIncome": 5000, "MonthlyRate": 12000, "NumCompaniesWorked": 2, "PercentSalaryHike": 10, "PerformanceRating": 3, "RelationshipSatisfaction": 3, "StockOptionLevel": 1, "TotalWorkingYears": 10, "TrainingTimesLastYear": 2, "WorkLifeBalance": 3, "YearsAtCompany": 5, "YearsInCurrentRole": 3, "YearsSinceLastPromotion": 2, "YearsWithCurrManager": 2, "BusinessTravel": "Travel_Rarely", "Department": "Research_&_Development", "EducationField": "Life_Sciences", "Gender": "Male", "JobRole": "Research_Scientist", "MaritalStatus": "Single", "OverTime": "No"}'
     ```
     - Actual response: `{"EmployeeNumber":1001,"Attrition_Probability":0.10233364995247053,"Attrition_Prediction":0}`.

3. **Dockerfile for API** (`Dockerfile.api`):
   - Use `uv` to manage dependencies inside the Docker container.
   ```dockerfile
   FROM python:3.9-bullseye

   WORKDIR /app

   RUN pip install uv

   COPY pyproject.toml uv.lock .
   RUN uv sync

   COPY app.py .
   COPY index.html .
   COPY models/scaler.pkl /app/models/scaler.pkl
   COPY data/processed/X_train.csv /app/data/processed/X_train.csv

   CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
   ```
   - **Build and Run**:
     ```
     docker build -f Dockerfile.api -t employee-attrition-api .
     docker run -d -p 8000:8000 employee-attrition-api
     ```
   - **Note**: Removed `apt-get install build-essential` since `uv` handles dependencies, and the `shap` build issue was resolved by adjusting `pyproject.toml`.

4. **Security**:
   - Plan to add an API key using FastAPI middleware (e.g., `fastapi.security.HTTPBearer`).
   - Example: Require a header `X-API-Key: your-secret-key`.

5. **Deployment**:
   - Host on a cloud provider (e.g., AWS ECS, Google Cloud Run) with auto-scaling for real-time demands.
   - Use HTTPS with a reverse proxy (e.g., Nginx) for production.

#### Benefits
- Enables HR to query predictions on-demand via a user-friendly web interface, supporting dynamic retention planning.
- Scalable with cloud deployment, preparing for future growth.
- Consistent dependency management with `uv`.

#### Challenges Faced
- **Path Issues**: Fixed incorrect relative path for `X_train.csv` (`../../data/processed/X_train.csv` → `./data/processed/X_train.csv`).
- **Build Issues**: Resolved `uv sync` failure by adding `[tool.hatch.build.targets.wheel]` to `pyproject.toml`.
- **Visibility**: Added `index.html` to serve a web interface, making the API accessible via a browser.

---

Let’s implement API security using an API key and deploy the Employee Attrition Prediction API to AWS ECS with CI/CD integration. I’ll update `app.py` to include API key authentication, modify the GitHub Actions workflow for deployment to AWS ECS, and provide instructions for setting up AWS resources. It’s 11:23 AM -04 on Friday, May 16, 2025, and we’re in `C:\Users\sebas\Documents\Data_Science\Portfolio\Data-Science-Portfolio\12_Employee_Attrition`.

---

## 1. Implement API Security with API Key

We’ll use FastAPI’s `HTTPBearer` middleware to require an API key via the `Authorization` header. The key will be validated against a predefined value (which should be stored securely in production, e.g., as an environment variable).

#### Update `app.py`

```python
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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

# API Key Security
security = HTTPBearer()
API_KEY = os.getenv("API_KEY", "your-secret-key")  # Default for local testing; use env var in production

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return credentials.credentials

# Root endpoint to serve HTML
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
    scaler = joblib.load(os.path.join(os.path.dirname(__file__), "models", "scaler.pkl"))
    logger.info("Scaler loaded successfully")
except Exception as e:
    logger.error(f"Failed to load scaler: {e}")
    raise

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
        "Age", "DailyRate", "DistanceFromHome", "Education", "EmployeeNumber",
        "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement", "JobLevel",
        "JobSatisfaction", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
        "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
        "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear",
        "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
        "YearsSinceLastPromotion", "YearsWithCurrManager", "TenureRatio",
        "SatisfactionScore", "IncomeToLevelRatio", "LongCommute",
    ]
    categorical_cols = [
        "BusinessTravel", "Department", "EducationField", "Gender",
        "JobRole", "MaritalStatus", "OverTime", "AgeGroup",
    ]

    data_df = pd.DataFrame([data.dict()])
    data_df["Gender"] = data_df["Gender"].map({"Male": 1, "Female": 0})
    data_df["OverTime"] = data_df["OverTime"].map({"Yes": 1, "No": 0})

    data_df["TenureRatio"] = data_df["YearsAtCompany"] / data_df["TotalWorkingYears"].replace(0, 1)
    data_df["SatisfactionScore"] = data_df[["EnvironmentSatisfaction", "JobSatisfaction", "RelationshipSatisfaction"]].mean(axis=1)
    data_df["AgeGroup"] = pd.cut(data_df["Age"], bins=[0, 30, 40, 100], labels=["lt30", "30-40", "gt40"]).astype(str)
    data_df["IncomeToLevelRatio"] = data_df["MonthlyIncome"] / data_df["JobLevel"]
    data_df["LongCommute"] = (data_df["DistanceFromHome"] > 10).astype(int)

    num_data = data_df[numerical_cols]
    cat_data = data_df[categorical_cols]

    num_data_scaled = scaler.transform(num_data)
    num_data = pd.DataFrame(num_data_scaled, columns=numerical_cols, index=data_df.index)

    cat_data = pd.get_dummies(cat_data, columns=categorical_cols, drop_first=True)
    cat_data.columns = [col.replace(" ", "_").replace("&", "_") for col in cat_data.columns]

    data_processed = pd.concat([num_data, cat_data], axis=1)
    expected_cols = pd.read_csv("./data/processed/X_train.csv").columns.tolist()
    for col in expected_cols:
        if col not in data_processed.columns:
            data_processed[col] = 0
    data_processed = data_processed[expected_cols]

    return data_processed

# Prediction endpoint with API key
@app.post("/predict")
async def predict(employee: EmployeeData, api_key: str = Depends(verify_api_key)):
    try:
        processed_data = preprocess_data(employee)
        proba = lr_model.predict_proba(processed_data)[:, 1][0]
        prediction = 1 if proba >= 0.45 else 0
        logger.info(f"Prediction for Employee {employee.EmployeeNumber}: {prediction} (Proba: {proba})")
        return {"EmployeeNumber": employee.EmployeeNumber, "Attrition_Probability": proba, "Attrition_Prediction": prediction}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

- **Changes**:
  - Added `HTTPBearer` security and a `verify_api_key` function to validate the API key.
  - The `/predict` endpoint now requires an API key via the `Authorization` header (e.g., `Authorization: Bearer your-secret-key`).
  - The API key is read from an environment variable (`API_KEY`), with a default value for local testing.

#### Test the API with Security Locally
1. **Set the API Key** (optional for local testing):
   - The default key is `your-secret-key`. You can set an environment variable (`.env`) for production:
     ```
     export API_KEY="your-secure-key"
     ```
   - On Windows (PowerShell):
     ```
     $env:API_KEY="your-secure-key"
     ```

2. **Run the Server**:
   ```
   uv run uvicorn app:app --host 127.0.0.1 --port 8000 --log-level debug
   ```

3. **Test the `/predict` Endpoint with API Key**:
   ```
   curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -H "Authorization: Bearer your-secret-key" -d '{"Age": 35, "DailyRate": 1000, "DistanceFromHome": 5, "Education": 3, "EmployeeNumber": 1001, "EnvironmentSatisfaction": 3, "HourlyRate": 50, "JobInvolvement": 3, "JobLevel": 2, "JobSatisfaction": 4, "MonthlyIncome": 5000, "MonthlyRate": 12000, "NumCompaniesWorked": 2, "PercentSalaryHike": 10, "PerformanceRating": 3, "RelationshipSatisfaction": 3, "StockOptionLevel": 1, "TotalWorkingYears": 10, "TrainingTimesLastYear": 2, "WorkLifeBalance": 3, "YearsAtCompany": 5, "YearsInCurrentRole": 3, "YearsSinceLastPromotion": 2, "YearsWithCurrManager": 2, "BusinessTravel": "Travel_Rarely", "Department": "Research_&_Development", "EducationField": "Life_Sciences", "Gender": "Male", "JobRole": "Research_Scientist", "MaritalStatus": "Single", "OverTime": "No"}'
   ```
   - **Expected Response**: `{"EmployeeNumber":1001,"Attrition_Probability":0.10233364995247053,"Attrition_Prediction":0}`.
   - Without the correct API key, you’ll get: `{"detail":"Invalid API Key"}`.

---

### 3. Next Steps
- **Monitoring**: Add Prometheus and Grafana for API uptime and latency monitoring on AWS ECS.
- **HTTPS**: Ensure the ALB uses HTTPS with a proper SSL certificate.
- **Refine Security**: Rotate API keys periodically and consider OAuth for more robust authentication.
