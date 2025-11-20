import pandas as pd
import os

# Display settings for better readability
pd.options.display.float_format = lambda x: f"{x:20.2f}"
pd.set_option("display.max_columns", 999)

# Load raw interim dataset
input_path = "../../data/interim/online_retail_2009_2010.csv"
output_path = "../../data/interim/online_retail_2009_2010_clean.csv"

df = pd.read_csv(input_path)
initial_rows = len(df)

# Convert Invoice to string and remove cancellations (start with 'C')
df["Invoice"] = df["Invoice"].astype(str)
df = df[~df["Invoice"].str.startswith("C")]

# Remove rows with missing Customer ID
df.dropna(subset=["Customer ID"], inplace=True)

# Check if the Customer ID column contains values with non-zero decimal places
(df[df["Customer ID"] % 1 != 0]).sum()

# Convert Customer ID into int
df["Customer ID"] = df["Customer ID"].astype(int)

# Remove rows with missing Description
df.dropna(subset=["Description"], inplace=True)

# Remove rows with non-positive price
df = df[df["Price"] > 0]

# Remove rows with negative quantity
df = df[df["Quantity"] > 0]

# Filter valid invoice numbers (6 digits)
df = df[df["Invoice"].str.match("^\d{6}$")]

# Filter valid stock codes (5 digits, optional suffix)
df["StockCode"] = df["StockCode"].astype(str)
df = df[
    df["StockCode"].str.match(r"^\d{5}$")
    | df["StockCode"].str.match(r"^\d{5}[a-zA-Z]+$")
    | df["StockCode"].str.match(r"^PADS$")
]

# Rename columns
df.rename(
    columns={"Invoice": "InvoiceNo", "Customer ID": "CustomerID", "Price": "UnitPrice"},
    inplace=True,
)

# Convert InvoiceDate to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Calculate total transaction value
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# Summary logging
final_rows = len(df)
dropped_pct = (initial_rows - final_rows) / initial_rows * 100
print(f"Cleaned dataset saved to {output_path}")
print(f"Original rows: {initial_rows:,}")
print(f"Cleaned rows:  {final_rows:,}")
print(f"Percentage dropped: {dropped_pct:.2f}%")

# Save cleaned dataset
df.to_csv(output_path, index=False)
