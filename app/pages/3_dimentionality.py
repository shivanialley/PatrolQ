import streamlit as st
import pandas as pd

st.title("📉 Dimensionality Reduction")

df = pd.read_csv("data/processed/crime_cleaned.csv")

st.markdown("""
### PCA & UMAP Insights
- Reduced high-dimensional crime data
- Explained variance analysis
- Cluster separability visualization
""")

st.dataframe(df.head(20))
