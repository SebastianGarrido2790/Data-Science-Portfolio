# Online Retail Customer Segmentation

This project applies data science techniques—specifically RFM (Recency, Frequency, Monetary) analysis, KMeans clustering, and Customer Lifetime Value (CLV) estimation—to segment customers of an online retail business and evaluate their potential future value. The insights are intended to drive personalized, data-driven marketing strategies.

---

## Overview

This project focuses on:
- Importing and preprocessing customer transaction data.
- Creating RFM metrics (Recency, Frequency, and Monetary) for each customer.
- Removing outliers using the interquartile range (IQR) method.
- Applying KMeans clustering to segment customers based on their purchasing behavior.
- Visualizing the clusters using various plots (e.g., violin plots, pairplots, PCA projections).

The final goal is to provide actionable insights into customer segments that can inform customer-centric marketing strategies.

Key components:

* Data ingestion and preprocessing.
* Feature engineering using RFM metrics.
* Outlier removal for robust modeling.
* KMeans clustering for behavioral segmentation.
* CLV estimation for long-term value prediction using historical data.
* Segment-level lifetime value visualization and interpretation.

---

## Objectives

* Understand customer purchasing behavior.
* Segment customers for targeted marketing campaigns.
* Estimate customer value over a 6-month horizon.
* Prioritize high-value segments for retention and acquisition.

---

## Project Structure

```plaintext
├── LICENSE
├── README.md
├── data
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── models/
│   ├── cluster_analysis.ipynb
│   ├── customer_life_value.py
│   └── trained_kmeans_model.pkl
├── references/
├── reports/
│   └── figures/
├── pyproject.toml
└── src/
    ├── __init__.py
    ├── data/
    │   ├── make_dataset.py
    │   └── data_ingestor.py
    ├── features/
    │   └── build_features.py
    ├── models/
    │   ├── cluster_analysis.ipynb
    │   ├── customer_life_value.py
    │   └── train_model.py
    └── visualization/
        ├── EDA.ipynb
        └── plot_settings.py
```

---

## Dataset

The project uses the **Online Retail II** dataset containing transactions for a UK-based online retail business between December 2009 and December 2011. The dataset includes the following key variables:
- **Invoice**: Invoice number (unique per transaction).
- **StockCode**: Product code.
- **Description**: Product name.
- **Quantity**: Number of items per transaction.
- **InvoiceDate**: Date and time of the transaction.
- **Price**: Unit price in GBP.
- **CustomerID**: Unique customer identifier.
- **Country**: Customer country.

---

## Methodology

### RFM + KMeans Segmentation

* **RFM scoring** is used to quantify recency (days since last purchase), frequency (number of transactions), and monetary value (total revenue).
* **Outlier removal** is done using IQR on Frequency and Monetary columns.
* **KMeans** is trained on scaled RFM features to form distinct behavioral clusters.

### Customer Lifetime Value (CLV)

* CLV is estimated using the `uv` library, a probabilistic modeling tool.
* The model uses:

  * Historical purchase frequency.
  * Monetary value per transaction.
  * Time since last purchase and customer age.
* Results are grouped by segment to compute average CLV per segment over a 6-month period.

---

## Technical Architecture

* **Data Preprocessing:** `src/data/data_ingestor.py`, `build_features.py`
* **Modeling:** `train_model.py` (KMeans), `customer_life_value.py` (CLV)
* **Visualization:** PCA, violin plots, retention curves
* **Libraries:** `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `uv`

---

## Installation

```bash
git clone https://github.com/SebastianGarrido2790/online-retail-customer-segmentation.git
cd online-retail-customer-segmentation

# Create virtual environment
python -m venv env
env\Scripts\activate  # macOS/Linux: source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Data Cleaning and Feature Engineering

- **Script:** `src/features/build_features.py`
- **Description:** This script reads the raw data, converts data types (e.g., parsing `InvoiceDate` as datetime), renames columns (e.g., `Customer ID` to `CustomerID`), calculates `TotalSales`, and creates RFM metrics. Outliers in `Frequency` and `Monetary` are removed using an IQR-based method.
- **Output:** Cleaned dataset saved to `data/processed/online_retail_2009_2010_without_outliers.csv`.

### Cluster Analysis

- **Notebook:** `src/models/cluster_analysis.ipynb`
- **Description:** This notebook:
  - Loads the cleaned RFM data.
  - Applies KMeans clustering.
  - Uses the Elbow Method and silhouette scores to determine the optimal number of clusters.
  - Visualizes the clustering results (cluster distribution, RFM averages, PCA projections, etc.).
- **Output:** Cluster labels added to the dataset and various plots to interpret customer segments.

### Visualization

- **Directory:** `src/visualization/`
- **Description:** Contains scripts and notebooks for generating visualizations. Custom plot settings are defined in `plot_settings.py` for consistent style across figures.

## Outlier Removal

Outliers in the `Frequency` and `Monetary` features are removed using the Interquartile Range (IQR) method. The function `remove_outliers()` in `src/features/build_features.py` applies this method and saves the resulting dataset to the `data/processed` folder.

## Future Work

- **Feature Expansion:** Incorporate additional features such as customer demographics or web browsing behavior.
- **Model Improvements:** Experiment with alternative clustering algorithms (e.g., DBSCAN, hierarchical clustering) and compare their performance.
- **Visualization Enhancements:** Develop interactive dashboards (e.g., using Plotly or Dash) to explore customer segments in real time.

### Step 1: Prepare the Data

```bash
python src/data/data_ingestor.py
python src/features/build_features.py
```

### Step 2: Train Clustering Model

```bash
python src/models/train_model.py
```

### Step 3: Estimate Customer Lifetime Value

```bash
python src/models/customer_life_value.py
```

Generates:

* Segment-wise CLV summary table
* Visualizations: Segment CLV barplot, retention curves

Output saved to:

* `data/processed/clv_output.csv`
* `reports/figures/segment_clv_distribution.png`

---

## Key Results

| Segment   | Avg. 6-Month CLV | Size |
| --------- | ---------------- | ---- |
| Champions | £120.54          | 85   |
| High      | £82.17           | 142  |
| Medium    | £44.03           | 210  |
| Low       | £21.17           | 175  |

* **Champions** are the most valuable customers—top priority for retention.
* **Medium and Low** clusters offer opportunities for re-engagement.

---

## Maintenance

* Use `pyproject.toml` to manage dependencies.
* Follow modular scripts to update components independently.
* Visual assets and results are stored in `reports/`.

---

## Troubleshooting

* `KeyError: 'segment'`: Ensure the clustering script has run and segment labels exist before running CLV.
* Inconsistent CLV results: Verify that frequency and monetary values are clean and non-zero.
* Model convergence issues: Check for sparse data or missing customer histories.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
