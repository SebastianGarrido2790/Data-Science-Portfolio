# Data Issue Log

## Solvable Issues
1. **Redundant Columns**:
   - Columns: `Over18`, `EmployeeCount`, `StandardHours`.
   - Issue: Single values (`Y`, 1, 80) with no variation.
   - Resolution: Flag for removal in Data Preparation.
   - Impact: 3/35 columns (8.6%).

## Unsolvable Issues
1. **Potential Outliers in MonthlyIncome**:
   - Issue: High values (e.g., ~19,999) may reflect senior roles.
   - Resolution: Validate during EDA with `JobLevel` and `JobRole`.
   - Impact: Unknown until analysis; likely <5% of rows.

## Augmentation Ideas
- Create tenure ratio: `YearsAtCompany` / `TotalWorkingYears`.
- Aggregate satisfaction: Mean of `JobSatisfaction`, `EnvironmentSatisfaction`.
- Bin `Age` into groups (e.g., <30, 30-40, >40).

## Notes
- Dataset is clean with no missing values or duplicates.
- Outliers and business logic violations are minimal, pending EDA confirmation.
- Redundant columns will be addressed in Step 4.