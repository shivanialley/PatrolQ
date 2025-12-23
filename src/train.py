import mlflow
from eda import run_eda
from clustering import run_clustering
from dimensionality import apply_pca
from logger import get_logger

logger = get_logger()

mlflow.set_experiment("PatrolIQ-Crime-Analytics")

with mlflow.start_run():
    summary = run_eda(df)
    mlflow.log_metrics(summary)

    clusters = run_clustering(X)
    for name, res in clusters.items():
        if "silhouette" in res:
            mlflow.log_metric(f"{name}_silhouette", res["silhouette"])

    X_pca, pca = apply_pca(X)
    mlflow.log_artifact("artifacts/pca_variance.json")