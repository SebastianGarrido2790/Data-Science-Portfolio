import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("../../data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")

pd.options.display.max_columns = None
df

# Basic structure
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)

# Check duplicates
print("Duplicate Rows:", df.duplicated().sum())
print("Duplicate EmployeeNumber:", df["EmployeeNumber"].duplicated().sum())

# Check unique values for categorical columns
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    print(f"\nUnique values in {col}:", df[col].unique())

# Summary statistics for numerical columns
numerical_cols = df.select_dtypes(include="int64").columns
print("\nNumerical Summary:\n", df[numerical_cols].describe())
