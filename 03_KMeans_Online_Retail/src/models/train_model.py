import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# -----------------------------
# Display Settings
# -----------------------------
pd.options.display.float_format = lambda x: f"{x:20.2f}"
pd.set_option("display.max_columns", 999)

# -----------------------------
# Load Scaled RFM Data
# -----------------------------
df_scaled = pd.read_csv("../../data/processed/online_retail_2009_2010_scaled.csv")
df_scaled.drop(columns=["Unnamed: 0"], inplace=True)

# Load non-scaled data (without outliers) for interpretation
df_clean = pd.read_csv(
    "../../data/processed/online_retail_2009_2010_without_outliers.csv",
    parse_dates=["LastInvoiceDate"],
)

# -----------------------------
# Visualize 3D Scatter (Raw Scaled Data)
# -----------------------------
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")
scatter = ax.scatter(
    df_scaled["Monetary"], df_scaled["Frequency"], df_scaled["Recency"]
)
ax.set_xlabel("Monetary Value")
ax.set_ylabel("Frequency")
ax.set_zlabel("Recency")
ax.set_title("3D Scatter Plot of Scaled Customer Data")
plt.savefig("../../reports/figures/rfm/3d_scatter_scaled.png", dpi=300)
plt.show()
plt.close()


# -----------------------------
# Elbow & Silhouette Plot
# -----------------------------
def plot_elbow_silhouette(data, max_k=10):
    """
    Plots the Elbow Method and Silhouette Score to determine the optimal number of clusters for KMeans.

    Parameters:
    data (pd.DataFrame): Standardized RFM dataset.
    max_k (int): Maximum number of clusters to test.
    """
    wcss = []
    silhouette_scores = []
    k_values = range(2, max_k + 1)  # Silhouette requires at least 2 clusters

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=1000)
        cluster_labels = kmeans.fit_predict(data)

        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, cluster_labels))

    # Plot Elbow Method (WCSS)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(k_values, wcss, marker="o", linestyle="--", color="b", label="WCSS")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("WCSS", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.set_title("Elbow Method and Silhouette Score")

    # Plot Silhouette Score on the same graph
    ax2 = ax1.twinx()
    ax2.plot(
        k_values,
        silhouette_scores,
        marker="s",
        linestyle="-",
        color="r",
        label="Silhouette Score",
    )
    ax2.set_ylabel("Silhouette Score", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    ax1.legend(loc="upper right")
    ax2.legend(loc="lower right")
    plt.savefig("../../reports/figures/elbow_silhouette(best_k).png", dpi=300)
    plt.show()
    plt.close()


plot_elbow_silhouette(df_scaled)

# -----------------------------
# Train Final KMeans Model
# -----------------------------
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=1000)
cluster_labels = kmeans.fit_predict(df_scaled)
df_clean["Cluster"] = cluster_labels

# -----------------------------
# Save Final Clustered Data
# -----------------------------
df_clean.to_csv(
    "../../data/processed/online_retail_2009_2010_with_clusters.csv", index=False
)

# -----------------------------
# Cluster Profiling Summary
# -----------------------------
cluster_summary = (
    df_clean.groupby("Cluster")
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
# PCA for 2D Cluster Visualization
# -----------------------------
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_scaled)
df_clean["PC1"] = pca_components[:, 0]
df_clean["PC2"] = pca_components[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x="PC1", y="PC2", hue="Cluster", palette="tab10", s=60)
plt.title("Customer Segments via PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.savefig("../../reports/figures/rfm/cluster_pca_final.png", dpi=300)
plt.show()
plt.close()

# -----------------------------
# Final 3D Scatter Plot by Cluster
# -----------------------------
cluster_colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
colors = df_clean["Cluster"].map(cluster_colors)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")
scatter = ax.scatter(
    df_clean["Monetary"],
    df_clean["Frequency"],
    df_clean["Recency"],
    c=colors,
    marker="o",
)

ax.set_xlabel("Monetary Value")
ax.set_ylabel("Frequency")
ax.set_zlabel("Recency")
ax.set_title("3D Scatter Plot of Customer Data by Cluster")
plt.savefig("../../reports/figures/rfm/3d_scatter_final_clusters.png", dpi=300)
plt.show()
plt.close()
