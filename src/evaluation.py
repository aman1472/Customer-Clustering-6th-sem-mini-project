import pandas as pd
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load clustered data
DATA_PATH = BASE_DIR / "data" / "processed" / "customer_clusters.csv"

df = pd.read_csv(DATA_PATH)

# Cluster Profiling
profile = df.groupby("Cluster").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean",
    "CustomerID": "count"
}).rename(columns={"CustomerID": "Customer_Count"})

print("\n====== CUSTOMER SEGMENT PROFILE ======\n")
print(profile)

# Save profile
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "cluster_profile.csv"
profile.to_csv(OUTPUT_PATH)

print("\nProfile saved to:", OUTPUT_PATH)
