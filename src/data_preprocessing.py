import pandas as pd
from datetime import datetime
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw" / "Online Retail.xlsx"
DATA_PROCESSED = BASE_DIR / "data" / "processed" / "customer_features.csv"

# Create processed directory if it does not exist
(DATA_PROCESSED.parent).mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_excel(DATA_RAW)

print("Initial shape:", df.shape)

# -----------------------------
# Basic cleaning
# -----------------------------
# Remove rows without CustomerID
df = df.dropna(subset=["CustomerID"])

# Remove negative or zero quantity (returns / invalid)
df = df[df["Quantity"] > 0]

# Convert InvoiceDate to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Create total price per row
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

print("After cleaning:", df.shape)

# -----------------------------
# RFM Feature Engineering
# -----------------------------
# Reference date = last invoice date in dataset
reference_date = df["InvoiceDate"].max()

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,  # Recency
    "InvoiceNo": "nunique",                                     # Frequency
    "TotalPrice": "sum"                                         # Monetary
}).reset_index()

rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

print("\nSample RFM data:")
print(rfm.head())

# -----------------------------
# Save processed data
# -----------------------------
rfm.to_csv(DATA_PROCESSED, index=False)
print(f"\nCustomer features saved to: {DATA_PROCESSED}")
