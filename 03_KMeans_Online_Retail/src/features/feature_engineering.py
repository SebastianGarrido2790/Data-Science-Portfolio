import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Set pandas display options
pd.options.display.float_format = lambda x: f"{x:20.2f}"
pd.set_option("display.max_columns", 999)

# -----------------------------
# Load and Prepare Dataset
# -----------------------------
df = pd.read_csv(
    "../../data/interim/online_retail_2009_2010_clean.csv", parse_dates=["InvoiceDate"]
)
df["TotalSales"] = df["TotalPrice"] * df["Quantity"]
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)


# -----------------------------
# Compute RFM Metrics Function
# -----------------------------
def compute_rfm(data: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    rfm = data.groupby(by="CustomerID", as_index=False).agg(
        {
            "InvoiceDate": lambda x: (
                reference_date - x.max()
            ).days,  # Get Last Purchase Date
            "Invoice": "nunique",  # Frequency: Number of unique purchases
            "TotalSales": "sum",  # Monetary: Total spending
        }
    )
    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
    return rfm


rfm = compute_rfm(df, reference_date)
rfm.to_csv("../../data/processed/rfm_metrics.csv", index=False)


# -----------------------------
# Visualize RFM Distributions
# -----------------------------
def plot_rfm_boxplots(rfm_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ["Recency", "Frequency", "Monetary"]
    colors = ["skyblue", "lightgreen", "salmon"]
    for i, metric in enumerate(metrics):
        sns.boxplot(data=rfm_df, y=metric, ax=axes[i], color=colors[i])
        axes[i].set_title(f"{metric} Distribution")
    plt.suptitle("Boxplots of RFM Metrics", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig("../../reports/figures/rfm/rfm_boxplots.png", dpi=300)
    plt.show()
    plt.close()


plot_rfm_boxplots(rfm)

# ----------------------------------------------
# Scatter Plot: Recency vs Monetary
# ----------------------------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=rfm, x="Recency", y="Monetary", s=50, color="navy")
plt.title("Recency vs Monetary Value")
plt.xlabel("Recency (Days since last purchase)")
plt.ylabel("Monetary (Total Spend)")
plt.savefig("../../reports/figures/rfm/Recency_vs_Monetary_Value", dpi=300)
plt.show()
plt.close()

# ----------------------------------------------
# Scatter Plot: Frequency vs Monetary
# ----------------------------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=rfm, x="Frequency", y="Monetary", s=50, color="darkgreen")
plt.title("Frequency vs Monetary Value")
plt.xlabel("Frequency (Number of Transactions)")
plt.ylabel("Monetary (Total Spend)")
plt.savefig("../../reports/figures/rfm/Frequency_vs_Monetary_Value", dpi=300)
plt.show()
plt.close()

# ----------------------------------------------
# Bar Plot: Top 10 Customers by Monetary Value
# ----------------------------------------------
top10 = rfm.sort_values("Monetary", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=top10.index.astype(str),
    y=top10["Monetary"],
    palette="Blues_d",
    hue=top10["Monetary"],
)
plt.title("Top 10 Customers by Monetary Value")
plt.xlabel("CustomerID")
plt.ylabel("Monetary (Total Spend)")
plt.xticks(rotation=45)
plt.savefig("../../reports/figures/rfm/Top_10_Customers_by_Monetary_Value", dpi=300)
plt.show()
plt.close()


# -----------------------------
# Remove Outliers (IQR Method)
# -----------------------------
def remove_outliers(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df_clean[
            (df_clean[col] >= Q1 - 1.5 * IQR) & (df_clean[col] <= Q3 + 1.5 * IQR)
        ]
    return df_clean


rfm_clean = remove_outliers(rfm, ["Frequency", "Monetary"])
rfm_clean.describe()
rfm_clean.to_csv(
    "../../data/processed/online_retail_2009_2010_without_outliers.csv", index=False
)
plot_rfm_boxplots(rfm_clean)
plt.savefig("../../reports/figures/rfm/rfm_boxplots_without_outliers.png", dpi=300)
plt.close()


# Plot data to verify the scale
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")
scatter = ax.scatter(
    rfm_clean["Monetary"], rfm_clean["Frequency"], rfm_clean["Recency"]
)
ax.set_xlabel("Monetary Value")
ax.set_ylabel("Frequency")
ax.set_zlabel("Recency")
ax.set_title("3D Scatter Plot of Customer Data")
plt.savefig("../../reports/figures/rfm/3d_scatter_plot.png", dpi=300)
plt.show()
plt.close()

# -----------------------------
# Scale Features
# -----------------------------
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_clean[["Recency", "Frequency", "Monetary"]])
rfm_scaled_df = pd.DataFrame(
    rfm_scaled, columns=["Recency", "Frequency", "Monetary"], index=rfm_clean.index
)
rfm_scaled_df.to_csv(
    "../../data/processed/online_retail_2009_2010_scaled.csv", index=True
)

# -----------------------------
# KMeans Clustering
# -----------------------------
kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
rfm_clean["Cluster"] = kmeans.fit_predict(rfm_scaled_df)

# -----------------------------
# Cluster Profiling
# -----------------------------
cluster_summary = (
    rfm_clean.groupby("Cluster")
    .agg(
        {
            "Recency": ["mean", "median"],
            "Frequency": ["mean", "median"],
            "Monetary": ["mean", "median", "count"],
        }
    )
    .round(1)
)

print("\nCluster Profiling Summary:")
print(cluster_summary)

# -----------------------------
# PCA for Cluster Visualization
# -----------------------------
pca = PCA(n_components=2)
pca_components = pca.fit_transform(rfm_scaled_df)
rfm_clean["PC1"] = pca_components[:, 0]
rfm_clean["PC2"] = pca_components[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=rfm_clean, x="PC1", y="PC2", hue="Cluster", palette="tab10", s=60)
plt.title("Customer Segments via PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.savefig("../../reports/figures/rfm/cluster_pca.png", dpi=300)
plt.show()
plt.close()
