import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# Page config
st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    layout="wide"
)

# -----------------------------
# Upload Dataset
# -----------------------------

st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.warning("Please upload a dataset to continue.")
    st.stop()


# -----------------------------
# Load Uploaded Data
# -----------------------------

@st.cache_data
def load_uploaded_data(file):

    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    return df


raw_df = load_uploaded_data(uploaded_file)


# -----------------------------
# Preprocessing + Clustering
# -----------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


import numpy as np

# Select numeric columns
numeric_df = raw_df.select_dtypes(include="number").copy()

# Remove ID columns
for col in numeric_df.columns:
    if "id" in col.lower():
        numeric_df.drop(col, axis=1, inplace=True)


# Replace infinite values
numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove columns with zero variance
numeric_df = numeric_df.loc[:, numeric_df.std() != 0]

# Fill missing values
numeric_df = numeric_df.fillna(numeric_df.mean())

# Final validation
if numeric_df.isnull().sum().sum() > 0:
    st.error("Dataset contains invalid values after cleaning.")
    st.stop()

if numeric_df.shape[0] < 2:
    st.error("Not enough valid rows for clustering.")
    st.stop()

# Check if data is still sufficient
if numeric_df.empty:
    st.error("After cleaning, no valid numeric data is left.")
    st.stop()


if numeric_df.shape[1] < 2:
    st.error("Dataset must contain at least 2 numeric columns.")
    st.stop()



# ✅ INSERT DEBUG CODE HERE (RIGHT HERE)
st.write("Numeric DF Info:")
st.write(numeric_df.describe())

st.write("Null Count:")
st.write(numeric_df.isnull().sum())

st.write("Any Inf:", np.isinf(numeric_df.values).any())


# Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)


# Select K
st.sidebar.header("Clustering Settings")

max_k = min(10, len(numeric_df))

k = st.sidebar.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=max_k,
    value=3
)


# Train model
model = KMeans(n_clusters=k, random_state=42)
labels = model.fit_predict(scaled_data)


# Add cluster column
df = raw_df.loc[numeric_df.index].copy()
df["Cluster"] = labels



# -----------------------------
# Cluster Profile
# -----------------------------

profile = df.groupby("Cluster").mean(numeric_only=True).reset_index()





# -----------------------------
# Header
# -----------------------------
st.title("📊 Customer Intelligence Dashboard")
st.markdown("AI-Powered Customer Segmentation System")


# -----------------------------
# KPIs
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    df["CustomerID"].nunique() if "CustomerID" in df.columns else "N/A"
)

col2.metric("Total Transactions", len(df))


# Dynamic Revenue Calculation
if "Monetary" in df.columns:
    revenue = df["Monetary"].sum()
elif {"Quantity", "UnitPrice"}.issubset(df.columns):
    revenue = (df["Quantity"] * df["UnitPrice"]).sum()
else:
    revenue = None

col3.metric(
    "Total Revenue",
    f"₹ {revenue:,.0f}" if revenue is not None else "N/A"
)


col4.metric("Clusters", df["Cluster"].nunique())



# -----------------------------
# Cluster Overview
# -----------------------------
st.subheader("Customer Distribution")

fig1 = px.histogram(
    df,
    x="Cluster",
    color="Cluster",
    title="Customers per Cluster"
)

st.plotly_chart(fig1, use_container_width=True)


# -----------------------------
# RFM Analysis (Optional)
# -----------------------------

st.subheader("RFM Analysis")

rfm_cols = {"Recency", "Frequency", "Monetary"}

if rfm_cols.issubset(df.columns):

    col1, col2 = st.columns(2)

    with col1:
        fig2 = px.box(
            df,
            x="Cluster",
            y="Monetary",
            title="Monetary Value by Cluster"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.scatter(
            df,
            x="Recency",
            y="Frequency",
            color="Cluster",
            title="Recency vs Frequency"
        )
        st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("RFM analysis not available for this dataset.")



# -----------------------------
# Cluster Profiles
# -----------------------------
st.subheader("Cluster Profiles")

st.dataframe(profile)


# -----------------------------
# Customer Explorer
# -----------------------------
st.subheader("Customer Explorer")

selected_cluster = st.selectbox(
    "Select Cluster",
    sorted(df["Cluster"].unique())
)

filtered = df[df["Cluster"] == selected_cluster]

st.dataframe(filtered.head(100))


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Built by Aman Choudhary section B| 6th Sem Mini Project")
