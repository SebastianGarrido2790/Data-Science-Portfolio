# EDA & Insight Generation

## Overview
Explored the IBM HR Analytics dataset (1,470 rows, 35 columns) to uncover patterns, test hypotheses, and prioritize features for predicting `Attrition`.

## Key Findings
- **Attrition**: Imbalanced (16% Yes, 84% No), requiring focus on recall.
- **Univariate**:
  - `Age`: Normal, mean ~37, range 18-60.
  - `JobSatisfaction`: Uniform across 1-4, slight skew to 3-4.
- **Bivariate**:
  - Low `JobSatisfaction` (1-2) has higher attrition.
  - Longer `DistanceFromHome` (median ~10 vs. 7) linked to attrition.
  - `OverTime` employees have higher attrition.
- **Multivariate**:
  - High correlations: `TotalWorkingYears`, `YearsAtCompany`, `JobLevel` (0.7-0.8).
  - Low `MonthlyIncome` at lower `JobLevel` tied to attrition.
- **Temporal**: No time-series data; tenure features show higher attrition for `YearsAtCompany` < 5.

## Hypothesis Testing
1. **Low JobSatisfaction/WorkLifeBalance increases attrition**: Supported (p < 0.05, chi-square).
2. **Higher DistanceFromHome/BusinessTravel increases attrition**: Supported for `DistanceFromHome` (p < 0.05, t-test).
3. **Younger employees/fewer YearsAtCompany churn more**: Supported (visualized).
4. **Low MonthlyIncome relative to JobLevel drives turnover**: Supported for lower levels.

**Refined Hypothesis**: Overtime exacerbates attrition, especially with low satisfaction.

## Actionable Insights
- Target retention for low-satisfaction or overtime employees.
- Offer remote work for long-commute employees.
- Engage younger/newer employees with onboarding programs.

## Prioritized Features
- **High Impact**: `JobSatisfaction`, `OverTime`, `DistanceFromHome`, `YearsAtCompany`, `MonthlyIncome`, `WorkLifeBalance`.
- **Moderate**: `Age`, `BusinessTravel`, `JobLevel`, `MaritalStatus`.
- **Low**: `Over18`, `EmployeeCount`, `StandardHours` (remove).

## Visualizations
- Saved in `reports/figures/` (e.g., `attrition_distribution.png`, `correlation_heatmap.png`).

## Go/No-Go
**Go**: Dataset supports business goal; clear patterns identified.

## Refined Problem Statement
Develop a machine learning model to predict employee attrition with ≥80% recall, focusing on key drivers like low JobSatisfaction, OverTime, and long DistanceFromHome, to enable HR to implement targeted retention strategies that reduce turnover by 10% within 12 months.