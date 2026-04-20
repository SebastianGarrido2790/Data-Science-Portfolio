import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# Load raw data
df = pd.read_csv("../../data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Step 1: Data Cleaning
# Drop redundant columns (identified in Step 2)
df = df.drop(columns=["Over18", "EmployeeCount", "StandardHours"])

# Step 2: Feature Engineering
# Tenure Ratio
df["TenureRatio"] = df["YearsAtCompany"] / df["TotalWorkingYears"].replace(
    0, 1
)  # Avoid division by 0

# Satisfaction Score
df["SatisfactionScore"] = df[
    ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction"]
].mean(axis=1)

# Age Bins (Cleaned names)
df["AgeGroup"] = pd.cut(
    df["Age"], bins=[0, 30, 40, 100], labels=["lt30", "30-40", "gt40"]
).astype(str)

# Income-to-Level Ratio
df["IncomeToLevelRatio"] = df["MonthlyIncome"] / df["JobLevel"]

# Long Commute Flag
df["LongCommute"] = (df["DistanceFromHome"] > 10).astype(int)

# Step 3: Define features and target
X = df.drop(columns=["Attrition"])
y = df["Attrition"].map({"Yes": 1, "No": 0})  # Encode target

# Step 4: Define preprocessing pipeline
# Numerical and categorical columns
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


# Custom function to clean feature names (XGBoost’s requirements)
def clean_feature_names(transformer, feature_names):
    """
    Clean features to meet XGBoost’s requirement that feature names be strings without [, ], <, or >.
    """
    return [
        name.replace("[", "").replace("]", "").replace("<", "").replace(">", "")
        for name in feature_names
    ]


# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols),
    ]
)

# Full pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

# Fit and transform with clean names
X_processed = pipeline.fit_transform(X)
feature_names = numerical_cols + clean_feature_names(
    preprocessor.named_transformers_["cat"],
    pipeline.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .get_feature_names_out(categorical_cols),
)
X_df = pd.DataFrame(X_processed, columns=feature_names)

# Save the scaler
scaler = pipeline.named_steps["preprocessor"].named_transformers_["num"]
os.makedirs(
    "../../models", exist_ok=True
)  # Create models directory if it doesn't exist
joblib.dump(scaler, "../../models/scaler.pkl")
print("Scaler saved to ../../models/scaler.pkl")

# Step 5: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, stratify=y, random_state=42
)

# Step 6: Save processed data
X_train.to_csv("../../data/processed/X_train.csv", index=False)
X_test.to_csv("../../data/processed/X_test.csv", index=False)
y_train.to_csv("../../data/processed/y_train.csv", index=False)
y_test.to_csv("../../data/processed/y_test.csv", index=False)

print("Processed data saved to ../../data/processed/")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Sample feature names:", X_train.columns.tolist()[:5], "...")
print("X_train columns:", X_train.columns)
