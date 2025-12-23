import os
import mlflow

from src.data_loader import load_data
from src.preprocessing import clean_data
from src.features import select_features
from src.eda import run_eda
from src.clustering import run_clustering
from src.dimentionality import apply_pca
from src.logger import get_logger

# -------------------------------------------------
# Setup
# -------------------------------------------------
os.makedirs("data/processed", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logger = get_logger()
mlflow.set_experiment("PatrolIQ-Crime-Analytics")

if __name__ == "__main__":

    logger.info("Loading dataset...")
    df = load_data("data/raw/chicago_crime.csv")

    logger.info("Cleaning data...")
    df = clean_data(df)

    # 🔹 Sample recent 500K records
    df = df.sort_values("Date").tail(500_000)

    df.to_csv("data/processed/crime_cleaned.csv", index=False)

    logger.info("Running EDA...")
    summary = run_eda(df)

    logger.info("Selecting features...")
    X = select_features(df)

    with mlflow.start_run():

        mlflow.log_metrics(summary)

        logger.info("Running clustering...")
        clusters = run_clustering(X)
        for name, res in clusters.items():
            if "silhouette" in res:
                mlflow.log_metric(f"{name}_silhouette", res["silhouette"])

        logger.info("Running PCA...")
        X_pca, pca = apply_pca(X)

        mlflow.log_artifact("artifacts/pca_variance.json")

    logger.info("Pipeline completed successfully")