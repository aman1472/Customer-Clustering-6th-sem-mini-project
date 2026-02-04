import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PROCESSED = BASE_DIR / "data" / "processed" / "customer_features.csv"
DATA_OUTPUT = BASE_DIR / "data" / "processed" / "customer_clusters.csv"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PROCESSED)

print("Loaded data shape:", df.shape)

# -----------------------------
# Features for clustering
# -----------------------------
X = df[["Recency", "Frequency", "Monetary"]]

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Elbow Method
# -----------------------------
wcss = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(2, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.show()

# -----------------------------
# Train Final Model
# -----------------------------
optimal_k = 4  # You can change after seeing elbow

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

# -----------------------------
# Evaluation
# -----------------------------
score = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", round(score, 3))

# -----------------------------
# Save Results
# -----------------------------
df.to_csv(DATA_OUTPUT, index=False)

print("Clustered data saved to:", DATA_OUTPUT)



DATA_PATH = BASE_DIR / "data" / "processed" / "customer_features.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing file: {DATA_PATH}")



df = pd.read_csv(DATA_PATH)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[["Recency", "Frequency", "Monetary"]])

kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_data)


OUTPUT_PATH = BASE_DIR / "data" / "processed" / "customer_clusters.csv"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)


# -----------------------------
# Cluster Profiling
# -----------------------------

profile = df.groupby("Cluster").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean",
    "CustomerID": "count"
}).rename(columns={"CustomerID": "Count"})

print("\nCustomer Cluster Profile:\n")
print(profile)




