import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the file
df_remote = pd.read_csv("../csv_files/30_top_paying_skills.csv")

# Check structure and look for nulls
df_remote.info()
df_remote.isnull().sum()
df_remote.head()

# Sort by salary to get top-paying roles
df_remote_sorted = df_remote.sort_values(by="salary_year_avg", ascending=False)

# Plot top-paying job titles with company names
plt.figure(figsize=(14, 10))
sns.barplot(
    x="salary_year_avg",
    y="job_title",
    data=df_remote_sorted,
    hue="company_name",
    dodge=False,
    palette="Set3",
)

plt.xlabel("Average Yearly Salary (USD)")
plt.ylabel("Job Title")
plt.title("Top 30 Highest-Paying Remote Data Science Roles in the U.S. (2023)")
plt.legend(title="Company", bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.savefig("../figure/Top_30_Paying_Remote_Data_Science_Roles.png", dpi=300)
plt.tight_layout()
plt.show()
