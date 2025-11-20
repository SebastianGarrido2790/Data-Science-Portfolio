import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data into a DataFrame
df = pd.read_csv("../csv_files/skills_combination.csv")

# Combine the two skills into a single string for plotting
df["pair"] = df["skill_a"] + " + " + df["skill_b"]

# Sort by pair_count for clearer plotting
df_sorted = df.sort_values(by="pair_count", ascending=True)

# Set the visual style
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Create the horizontal barplot
sns.barplot(x="pair_count", y="pair", data=df_sorted, palette="viridis")

# Titles and labels
plt.title("Top Skill Pair Co-occurrences in Job Postings")
plt.xlabel("Pair Count")
plt.ylabel("Skill Pair")
plt.tight_layout()

# plt.savefig("../figure/skills_combination.png", dpi=300)
plt.show()
