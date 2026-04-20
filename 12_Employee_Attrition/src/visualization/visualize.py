import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Set plot style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Load dataset
df = pd.read_csv("../../data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Visualize outliers for key numerical columns
key_numerical = ["MonthlyIncome", "TotalWorkingYears", "DistanceFromHome"]
for col in key_numerical:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.savefig(
        f"../../reports/figures/data_distribution/numerical_data/boxplot_{col}.png"
    )
    plt.show()
    plt.close()

# Check business logic
logic_issues = df[df["YearsAtCompany"] > df["TotalWorkingYears"]]
print("YearsAtCompany > TotalWorkingYears:", len(logic_issues))

# Visualize distribution of categorical variables
categorical_columns = [
    col for col in df.select_dtypes(include=["object"]).columns if col != "Over18"
]
for col in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=df[col])
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.savefig(
        f"../../reports/figures/data_distribution/categorical_data/distribution_{col}.png"
    )
    plt.show()
    plt.close()

# ----
# EDA
# ----

# Univariate Analysis
# Attrition distribution
plt.figure()
sns.countplot(x="Attrition", data=df)
plt.title("Attrition Distribution")
plt.savefig("../../reports/figures/eda_visualization/attrition_distribution.png")
plt.show()
plt.close()

# Age distribution
plt.figure()
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.savefig("../../reports/figures/eda_visualization/age_distribution.png")
plt.show()
plt.close()

# JobSatisfaction distribution
plt.figure()
sns.countplot(x="JobSatisfaction", data=df)
plt.title("Job Satisfaction Distribution")
plt.savefig("../../reports/figures/eda_visualization/jobsatisfaction_distribution.png")
plt.show()
plt.close()

# Bivariate Analysis
# Attrition by JobSatisfaction
plt.figure()
sns.countplot(x="JobSatisfaction", hue="Attrition", data=df)
plt.title("Attrition by Job Satisfaction")
plt.savefig("../../reports/figures/eda_visualization/attrition_by_jobsatisfaction.png")
plt.show()
plt.close()

# Attrition by DistanceFromHome
plt.figure()
sns.boxplot(x="Attrition", y="DistanceFromHome", data=df)
plt.title("Attrition by Distance From Home")
plt.savefig("../../reports/figures/eda_visualization/attrition_by_distancefromhome.png")
plt.show()
plt.close()

# Attrition by OverTime
plt.figure()
sns.countplot(x="OverTime", hue="Attrition", data=df)
plt.title("Attrition by OverTime")
plt.savefig("../../reports/figures/eda_visualization/attrition_by_overtime.png")
plt.show()
plt.close()

# Multivariate Analysis
# Correlation heatmap (numerical features)
numerical_cols = df.select_dtypes(include="int64").columns
plt.figure(figsize=(12, 8))
sns.heatmap(df[numerical_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("../../reports/figures/eda_visualization/correlation_heatmap.png")
plt.show()
plt.close()

# MonthlyIncome vs. JobLevel by Attrition
plt.figure()
sns.scatterplot(x="MonthlyIncome", y="JobLevel", hue="Attrition", data=df)
plt.title("Monthly Income vs. Job Level by Attrition")
plt.savefig(
    "../../reports/figures/eda_visualization/income_vs_joblevel_by_attrition.png"
)
plt.show()
plt.close()
