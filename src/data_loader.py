import pandas as pd

def load_data(path):
    """Load Chicago crime data from CSV"""
    print(f"Loading data from C:/Users/Dell/Documents/Project_PatrolQ/data/raw...")
    df = pd.read_csv("C:/Users/Dell/Documents/Project_PatrolQ/data/raw/Crimes_-_2001_to_Present_20251215.csv")
    return df