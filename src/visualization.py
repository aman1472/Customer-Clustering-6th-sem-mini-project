import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load data
DATA_PATH = BASE_DIR / "data" / "processed" / "customer_clusters.csv"
df = pd.read_csv(DATA_PATH)

# Create output folder
PLOT_DIR = BASE_DIR / "reports" / "figures"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 1. Cluster Distribution
# -----------------------------
plt.figure()
df["Cluster"].value_counts().sort_index().plot(kind="bar")
plt.title("Customer Distribution by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.savefig(PLOT_DIR / "cluster_distribution.png")
plt.close()


# -----------------------------
# 2. Monetary Value by Cluster
# -----------------------------
plt.figure()
df.boxplot(column="Monetary", by="Cluster")
plt.title("Monetary Value by Cluster")
plt.suptitle("")
plt.xlabel("Cluster")
plt.ylabel("Monetary Value")
plt.savefig(PLOT_DIR / "monetary_by_cluster.png")
plt.close()


# -----------------------------
# 3. RFM Scatter Plot
# -----------------------------
plt.figure()
plt.scatter(df["Recency"], df["Monetary"], c=df["Cluster"])
plt.title("Recency vs Monetary (Colored by Cluster)")
plt.xlabel("Recency")
plt.ylabel("Monetary")
plt.savefig(PLOT_DIR / "recency_monetary.png")
plt.close()


print("All plots saved to:", PLOT_DIR)
