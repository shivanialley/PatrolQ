import pandas as pd
import warnings

# -------------------------------------------------
# Suppress Pandas datetime inference warning
# -------------------------------------------------
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Could not infer format"
)

def clean_data(df):
    """
    Robust cleaning & feature engineering for large,
    messy Chicago crime data (warning suppressed)
    """

    # -------------------------------------------------
    # 1️⃣ Drop rows without geo coordinates
    # -------------------------------------------------
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()

    # -------------------------------------------------
    # 2️⃣ Ensure Date column is string
    # -------------------------------------------------
    df["Date"] = df["Date"].astype(str)

    # -------------------------------------------------
    # 3️⃣ Safe datetime parsing (mixed formats allowed)
    # Warning intentionally suppressed
    # -------------------------------------------------
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # -------------------------------------------------
    # 4️⃣ Drop rows where datetime parsing failed
    # -------------------------------------------------
    df = df.dropna(subset=["Date"]).copy()

    # -------------------------------------------------
    # 5️⃣ Feature engineering (datetime-safe)
    # -------------------------------------------------
    df["Hour"] = df["Date"].dt.hour
    df["Day"] = df["Date"].dt.day_name()
    df["Month"] = df["Date"].dt.month
    df["Is_Weekend"] = df["Day"].isin(["Saturday", "Sunday"])

    return df
