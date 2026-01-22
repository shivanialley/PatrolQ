import pandas as pd

def clean_data(df):
    """
    Robust cleaning for large, messy Chicago crime data
    """

    # 1️⃣ Drop rows without geo coordinates
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()

    # 2️⃣ Force Date column to string (critical fix)
    df["Date"] = df["Date"].astype(str)

    # 3️⃣ Parse datetime safely (mixed formats)
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
        infer_datetime_format=True
    )

    # 4️⃣ Drop rows where datetime parsing failed
    df = df.dropna(subset=["Date"]).copy()

    # 5️⃣ FORCE datetime dtype (THIS FIXES .dt ERROR)
    df["Date"] = df["Date"].astype("datetime64[ns]")

    # 6️⃣ Feature engineering (now 100% safe)
    df["Hour"] = df["Date"].dt.hour
    df["Day"] = df["Date"].dt.day_name()
    df["Month"] = df["Date"].dt.month
    df["Is_Weekend"] = df["Day"].isin(["Saturday", "Sunday"])

    return df