import pandas as pd

def run_eda(df):
    summary = {
        "total_crimes": len(df),
        "arrest_rate": df["Arrest"].mean(),
        "domestic_rate": df["Domestic"].mean()
    }

    crime_type_dist = df["Primary Type"].value_counts().reset_index()
    crime_type_dist.columns = ["Crime_Type", "Count"]

    time_dist = df.groupby("Hour").size().reset_index(name="Crimes")

    powerbi_df = crime_type_dist.merge(time_dist, how="cross")
    powerbi_df.to_csv("data/processed/powerbi_crime_summary.csv", index=False)

    return summary
