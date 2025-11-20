from scipy.stats import chi2_contingency, ttest_ind
import pandas as pd

# Load dataset
df = pd.read_csv("../../data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Chi-square test for JobSatisfaction vs. Attrition
contingency_table = pd.crosstab(df["JobSatisfaction"], df["Attrition"])
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-square JobSatisfaction vs. Attrition: p-value = {p:.4f}")

# T-test for DistanceFromHome by Attrition
yes = df[df["Attrition"] == "Yes"]["DistanceFromHome"]
no = df[df["Attrition"] == "No"]["DistanceFromHome"]
t_stat, p = ttest_ind(yes, no)
print(f"T-test DistanceFromHome vs. Attrition: p-value = {p:.4f}")

# Chi-square for OverTime vs. Attrition
contingency_table = pd.crosstab(df["OverTime"], df["Attrition"])
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-square OverTime vs. Attrition: p-value = {p:.4f}")
