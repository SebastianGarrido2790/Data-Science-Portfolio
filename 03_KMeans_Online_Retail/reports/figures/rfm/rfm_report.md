## 🧾 Customer Segmentation Report

Here is a summary of our RFM analysis, outlier handling, clustering, and PCA visualization for the **Online Retail II** dataset. This can be exported as a PDF or markdown/HTML report using a Jupyter Notebook, LaTeX, or a report generation tool (e.g., `nbconvert`, `reportlab`, `WeasyPrint`, or `pandas-profiling` if extended).


**Dataset**: Online Retail II (2009–2010)
**Source**: UCI Machine Learning Repository
**Total Customers Analyzed**: 4,285
**Customers after Outlier Removal**: 3,633
**Clustering Method**: K-Means (k=4)

---

### 1. 📊 RFM Metrics Summary

RFM (Recency, Frequency, Monetary) metrics were computed for each customer using the cleaned transaction dataset. The reference date was set to one day after the last recorded purchase to ensure Recency > 0.

| Metric    | Mean      | Median   | Min  | Max       |
| --------- | --------- | -------- | ---- | --------- |
| Recency   | 98.59     | 59.00    | 1    | 374       |
| Frequency | 2.84      | 2.00     | 1    | 11        |
| Monetary  | 11,484.16 | 5,960.97 | 1.55 | 63,744.81 |

**Figures**:

* `rfm_boxplots.png`
* `Recency_vs_Monetary_Value.png`
* `Frequency_vs_Monetary_Value.png`
* `Top_10_Customers_by_Monetary_Value.png`

---

### 2. 🔍 Outlier Handling

Outliers in `Frequency` and `Monetary` were removed using the IQR method. This reduced noise and stabilized clustering results.

**Figure**:

* `rfm_boxplots_without_outliers.png`

---

### 3. 📈 K-Means Clustering Results

K-Means clustering identified 4 distinct customer segments based on scaled RFM features.

#### 📌 Cluster Profiling Summary

| Cluster | Count | Recency (Mean) | Frequency (Mean) | Monetary (Mean) | Profile                    |
| ------- | ----- | -------------- | ---------------- | --------------- | -------------------------- |
| 0       | 542   | 31.6           | 6.7              | 17,528.6        | **Loyal High-Spenders**    |
| 1       | 872   | 251.4          | 1.4              | 4,794.0         | **Inactive Low-Spenders**  |
| 2       | 396   | 64.0           | 4.4              | 41,722.8        | **Engaged VIPs**           |
| 3       | 1,823 | 52.9           | 2.0              | 6,318.6         | **Recent Moderate Buyers** |

**Figure**:

* `cluster_pca.png`

---

### 4. 🌐 3D Visualization

A 3D scatter plot shows the raw unscaled relationship between `Monetary`, `Frequency`, and `Recency` before clustering.

**Figure**:

* `3d_scatter_plot.png`

---

### 5. 📂 Outputs

| File                                           | Description                      |
| ---------------------------------------------- | -------------------------------- |
| `rfm_metrics.csv`                              | RFM metrics for all customers    |
| `online_retail_2009_2010_without_outliers.csv` | Cleaned data without outliers    |
| `online_retail_2009_2010_scaled.csv`           | Standardized data for clustering |

---

### ✅ Recommendations

* **Cluster 0**: Target with loyalty programs or VIP benefits.
* **Cluster 1**: Re-engagement campaigns, possibly churned.
* **Cluster 2**: High-value customers—consider premium upsells.
* **Cluster 3**: Educate with product recommendations to increase frequency.
