import numpy as np

def select_features(df):
    df["Season"] = df["Month"] % 12 // 3 + 1
    df["Crime_Severity"] = df["Primary Type"].map({
        "HOMICIDE": 5,
        "ROBBERY": 4,
        "BATTERY": 3
    }).fillna(1)

    return df[[
        "Latitude", "Longitude",
        "Hour", "Month",
        "Is_Weekend",
        "Arrest", "Domestic",
        "Crime_Severity"
    ]]
