# Gunakan base image miniconda
FROM continuumio/miniconda3

# Set working directory di container
WORKDIR /app

# Copy environment conda.yaml ke container
COPY conda.yaml .

# Buat environment conda sesuai conda.yaml
RUN conda env create -f conda.yaml

# Aktifkan environment conda secara default
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

# Copy seluruh kode project ke container
COPY . .

# Install mlflow CLI jika belum ada
RUN conda run -n mlflow-env pip install mlflow

# Expose port untuk MLflow serve (jika ingin serve model)
EXPOSE 8001

# Perintah default menjalankan MLflow serve model pada folder model
CMD ["conda", "run", "--no-capture-output", "-n", "mlflow-env", "mlflow", "models", "serve", "-m", "model", "-h", "0.0.0.0", "-p", "8001"]
