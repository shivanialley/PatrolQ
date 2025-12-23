import streamlit as st

st.set_page_config(
    page_title="PatrolIQ – Crime Analytics",
    layout="wide"
)

st.title("🚓 PatrolIQ – Crime Analytics Platform")

st.markdown("""
### End-to-End Crime Intelligence System

**Capabilities**
- 📊 Exploratory Data Analysis  
- 🧩 Geographic & Temporal Clustering  
- 📉 Dimensionality Reduction (PCA / UMAP)  
- 🧪 MLflow Experiment Tracking  
- 📈 Power BI Ready Outputs  

👉 Use the **sidebar** to navigate between pages.
""")

st.info("This application is powered by ML, MLflow, Docker, and AWS-ready architecture.")
