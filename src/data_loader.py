import pandas as pd

def load_data(path):
    df = pd.read_csv(r"C:\Users\Dell\Documents\guvi\Project\PatrolQ\data\Crimes_-_2001_to_Present_20251215.csv")
    return df
