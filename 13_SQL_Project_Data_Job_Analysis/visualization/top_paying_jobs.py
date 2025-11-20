import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")
plt.figure(figsize=(14, 8))

# Load dataset
df = pd.read_csv("../csv_files/1000_top_paying_skills.csv")

# Group by skill to get frequency and average salary
skill_stats = (
    df.groupby("skills")
    .agg(frequency=("skills", "count"), avg_salary=("salary_year_avg", "mean"))
    .sort_values(by="frequency", ascending=False)
)

# Select top 20 skills by frequency
top_skills = skill_stats.head(20).reset_index()

# Plot bar chart with two y-axes: frequency and average salary
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot frequency
sns.barplot(x="skills", y="frequency", data=top_skills, ax=ax1, color="skyblue")
ax1.set_ylabel("Frequency", fontsize=12)
ax1.set_xlabel("Skill", fontsize=12)
ax1.tick_params(axis="x", rotation=45)
ax1.set_title("Top 20 Data Science Skills by Frequency and Average Salary", fontsize=14)

# Plot average salary on secondary y-axis
ax2 = ax1.twinx()
sns.lineplot(
    x="skills", y="avg_salary", data=top_skills, ax=ax2, color="darkred", marker="o"
)
ax2.set_ylabel("Average Salary (USD)", fontsize=12)

# plt.savefig("../figure/Top_20_Data_Science_Skills.png", dpi=300)
plt.tight_layout()
plt.show()
