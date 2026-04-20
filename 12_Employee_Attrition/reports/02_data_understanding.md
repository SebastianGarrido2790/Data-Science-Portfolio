# Data Understanding and Governance

## Dataset Overview
- **Source**: IBM HR Analytics Employee Attrition & Performance dataset (Kaggle).
- **Storage**: `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`.
- **Shape**: 1,470 rows, 35 columns.
- **Target**: `Attrition` (Yes/No, binary classification).
- **Key Features**: `Age`, `JobSatisfaction`, `MonthlyIncome`, `DistanceFromHome`, `OverTime`.

## Data Quality Assessment
- **Completeness**: 100% (no missing values).
- **Duplicates**: None (0 rows, 0 duplicate `EmployeeNumber`).
- **Consistency**: High (consistent categorical values, e.g., `BusinessTravel`: ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']).
- **Issues**:
  - Redundant columns: `Over18`, `EmployeeCount`, `StandardHours` (single values).
  - Potential outliers: High `MonthlyIncome` values (~19,999), to be validated in EDA.
- **Business Logic**: No violations (e.g., `YearsAtCompany` ≤ `TotalWorkingYears`).

## Temporal Component
- No timestamps or time-series data.
- Tenure features (`YearsAtCompany`, `TotalWorkingYears`) are static, supporting classification, not forecasting.

## Governance
- **Lineage**: Sourced from Kaggle, stored in `data/raw/`.
- **Privacy**: Treat `EmployeeNumber`, `Gender` as sensitive.
- **Quality Scores**:
  - Completeness: 100%.
  - Consistency: High.
  - Accuracy: Assumed high (fictional data).

## Visualizations
- Boxplots for `MonthlyIncome`, `TotalWorkingYears`, `DistanceFromHome` saved in `reports/figures/`.

## Next Steps
- Proceed to EDA to test hypotheses (e.g., low `JobSatisfaction` drives attrition).
- Address redundant columns and outliers in Data Preparation.

See `references/data_issue_log.md` for detailed issue tracking.