import streamlit as st
import mlflow
import pandas as pd

st.title("🧪 MLflow Experiment Tracking")

# -------------------------------------------------
# Connect to MLflow (local tracking)
# -------------------------------------------------
mlflow.set_tracking_uri("http://localhost:5000")

experiment_name = "PatrolIQ-Crime-Analytics"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    st.error(f"❌ Experiment '{experiment_name}' not found")
    st.stop()

# -------------------------------------------------
# Load Runs
# -------------------------------------------------
runs_df = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.kmeans_silhouette DESC"]
)

if runs_df.empty:
    st.warning("⚠️ No MLflow runs found yet.")
    st.stop()

# -------------------------------------------------
# Display Summary
# -------------------------------------------------
st.subheader("📊 Experiment Summary")

st.metric(
    label="Total Runs",
    value=len(runs_df)
)

best_run = runs_df.iloc[0]

st.metric(
    label="Best Silhouette Score",
    value=round(best_run["metrics.kmeans_silhouette"], 4)
)

# -------------------------------------------------
# Metrics Table
# -------------------------------------------------
st.subheader("📈 All Experiment Runs")

metrics_cols = [col for col in runs_df.columns if col.startswith("metrics.")]
display_cols = ["run_id"] + metrics_cols

st.dataframe(
    runs_df[display_cols].rename(columns=lambda x: x.replace("metrics.", ""))
)

# -------------------------------------------------
# Parameters
# -------------------------------------------------
st.subheader("⚙️ Parameters Used")

param_cols = [col for col in runs_df.columns if col.startswith("params.")]
if param_cols:
    st.dataframe(
        runs_df[["run_id"] + param_cols].rename(
            columns=lambda x: x.replace("params.", "")
        )
    )
else:
    st.info("No parameters logged.")

# -------------------------------------------------
# Artifacts Info
# -------------------------------------------------
st.subheader("📦 Artifacts")

selected_run = st.selectbox(
    "Select Run ID to inspect artifacts",
    runs_df["run_id"].tolist()
)

if selected_run:
    st.info(
        f"Artifacts available for run `{selected_run}` can be viewed in MLflow UI."
    )
    st.markdown("👉 **Open MLflow UI:** http://localhost:5000")
