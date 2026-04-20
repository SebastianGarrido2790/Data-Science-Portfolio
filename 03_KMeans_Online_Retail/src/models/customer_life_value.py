import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes.plotting import (
    plot_probability_alive_matrix,
)

# Load cleaned dataset
DATA_PATH = "../../data/interim/online_retail_2009_2010_clean.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["InvoiceDate"])

df.describe()


# --- Outlier Handling ---
def boundaries(series, q1=0.05, q2=0.95):
    return series.quantile(q1), series.quantile(q2)


def cap_outliers(df, columns):
    df = df.copy()
    for col in columns:
        lower, upper = boundaries(df[col])
        df[col] = df[col].clip(lower, upper)
    return df


df = cap_outliers(df, ["TotalPrice", "Quantity"])
df["TotalSales"] = df["TotalPrice"] * df["Quantity"]
df[["TotalPrice", "Quantity", "TotalSales"]].describe()

# Focus on United Kingdom transactions only
df = df[df["Country"] == "United Kingdom"]

# --- RFM Preparation ---
clv = summary_data_from_transaction_data(
    df,
    customer_id_col="CustomerID",
    datetime_col="InvoiceDate",
    monetary_value_col="TotalPrice",
    observation_period_end="2011-12-09",
)
clv = clv[clv["frequency"] > 1]  # Customers that shopped more that 1 time

# --- BG/NBD Model ---
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(clv["frequency"], clv["recency"], clv["T"])

# Plot Probability Alive Matrix
plt.figure(figsize=(8, 6))
plot_probability_alive_matrix(bgf)
plt.tight_layout()
plt.savefig("../../reports/figures/probability_alive_matrix.png", dpi=150)
plt.show()
plt.close()

# Expected number of purchases in 6 months
t = 180  # 30 days period
clv["expected_purchases_6_months"] = (
    bgf.conditional_expected_number_of_purchases_up_to_time(
        t, clv["frequency"], clv["recency"], clv["T"]
    )
)
clv.sort_values(by="expected_purchases_6_months", ascending=False).head(10)

# --- Gamma-Gamma model for predicting monetary value ---
clv[["frequency", "monetary_value"]].corr()

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(clv["frequency"], clv["monetary_value"])

# 6 months customer lifetime value
clv["6_months_clv"] = ggf.customer_lifetime_value(
    bgf,
    clv["frequency"],
    clv["recency"],
    clv["T"],
    clv["monetary_value"],
    time=6,
    freq="D",
    discount_rate=0.01,
)
clv.sort_values(by="6_months_clv", ascending=False).head(10)

# Segment customers based on CLV
clv["segment"] = pd.qcut(
    clv["6_months_clv"], q=4, labels=["low", "medium", "high", "champions"]
)

# Segment Contribution Plot
segment_summary = (
    clv.groupby("segment")
    .agg(
        count=("6_months_clv", "count"),
        avg_clv=("6_months_clv", "mean"),
        total_clv=("6_months_clv", "sum"),
    )
    .reset_index()
)

# Sort for consistent plotting
segment_summary["segment"] = pd.Categorical(
    segment_summary["segment"],
    categories=["low", "medium", "high", "champions"],
    ordered=True,
)
segment_summary = segment_summary.sort_values("segment")

# Plot: CLV distribution by segment
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Barplot of total CLV contribution
sns.barplot(
    data=segment_summary, x="segment", y="total_clv", ax=axes[0], palette="viridis"
)
axes[0].set_title("Total CLV Contribution by Segment")
axes[0].set_ylabel("Total CLV (€)")

# Pie chart of customer count per segment
axes[1].pie(
    segment_summary["count"],
    labels=segment_summary["segment"],
    autopct="%1.1f%%",
    colors=sns.color_palette("viridis"),
)
axes[1].set_title("Customer Distribution by Segment")

plt.tight_layout()
plt.savefig("../../reports/figures/segment_contribution.png", dpi=300)
plt.show()
plt.close()

# Save output
clv.to_csv("../../data/processed/customer_lifetime_value.csv", index=True)
