# Data Preparation & Feature Engineering

## Overview
Transformed the raw IBM HR Analytics dataset into a clean, feature-rich format for modeling, using automated pipelines.

## Data Cleaning
- **Removed Redundant Columns**: `Over18`, `EmployeeCount`, `StandardHours` (single values, no variation).
- **Missing Values/Duplicates**: None identified.
- **Outliers**: High `MonthlyIncome` values validated by `JobLevel`, no removal needed.

## Feature Engineering
- **TenureRatio**: `YearsAtCompany` / `TotalWorkingYears` (career stability).
- **SatisfactionScore**: Average of `JobSatisfaction`, `EnvironmentSatisfaction`, `RelationshipSatisfaction`.
- **AgeGroup**: Binned `Age` into `<30`, `30-40`, `>40`.
- **IncomeToLevelRatio**: `MonthlyIncome` / `JobLevel` (normalize income by seniority).
- **LongCommute**: Binary flag for `DistanceFromHome` > 10.

## Transformations
- **Numerical**: Scaled using `StandardScaler` (e.g., `MonthlyIncome`, `DistanceFromHome`).
- **Categorical**: One-hot encoded (e.g., `BusinessTravel`, `Department`).
- **Pipeline**: Automated preprocessing with `sklearn` `ColumnTransformer` and `Pipeline`.

## Train/Test Split
- **Method**: 80/20 split, stratified by `Attrition` (16% Yes, 84% No).
- **Output**: Saved as `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` in `data/processed/`.
- **Shapes**: X_train (1176, 59), X_test (294, 59).

## Next Steps
- Proceed to modeling (Step 5) using processed data in `data/processed/`.
- Evaluate feature importance to validate engineered features.