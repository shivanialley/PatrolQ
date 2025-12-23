import streamlit as st
import pandas as pd

st.title("📊 Crime Analysis")

df = pd.read_csv("data/processed/crime_cleaned.csv")

st.subheader("Top 10 Crime Types")
st.bar_chart(df["Primary Type"].value_counts().head(10))

st.subheader("Arrest vs Domestic Incidents")
st.write(df[["Arrest", "Domestic"]].value_counts())

st.subheader("Hourly Crime Distribution")
st.line_chart(df.groupby("Hour").size())
