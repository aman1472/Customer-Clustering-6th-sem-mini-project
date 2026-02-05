import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# Page config
st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    layout="wide"
)


# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "customer_clusters.csv"
PROFILE_PATH = BASE_DIR / "data" / "processed" / "cluster_profile.csv"


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    profile = pd.read_csv(PROFILE_PATH)
    return df, profile


df, profile = load_data()


# -----------------------------
# Header
# -----------------------------
st.title("📊 Customer Intelligence Dashboard")
st.markdown("AI-Powered Customer Segmentation System")


# -----------------------------
# KPIs
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", df["CustomerID"].nunique())
col2.metric("Total Transactions", len(df))
col3.metric("Total Revenue", f"₹ {df['Monetary'].sum():,.0f}")
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
# RFM Analysis
# -----------------------------
st.subheader("RFM Analysis")

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
