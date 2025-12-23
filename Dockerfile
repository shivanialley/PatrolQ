# -----------------------------
# Base Image
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Copy project files
# -----------------------------
COPY . .

# -----------------------------
# Install dependencies
# -----------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Expose ports
# -----------------------------
# 8501 -> Streamlit
# 5000 -> MLflow
EXPOSE 8501 5000

# -----------------------------
# Run single pipeline entry
# -----------------------------
CMD ["python", "run_pipeline.py"]
