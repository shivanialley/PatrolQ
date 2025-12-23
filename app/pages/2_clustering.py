import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

st.title("🧩 Geographic Crime Hotspots")

df = pd.read_csv("data/processed/crime_cleaned.csv")

X = df[["Latitude", "Longitude"]]

model = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = model.fit_predict(X)

st.map(df)

st.caption("KMeans clustering on latitude & longitude")
