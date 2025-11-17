from pathlib import Path

# Root del progetto: la cartella che contiene "mlruns" e "data"
ROOT_DIR = Path(__file__).resolve().parents[1]

# MLflow
MLFLOW_TRACKING_URI = f"file:{ROOT_DIR / 'mlruns'}"

# Nome del modello registrato in MLflow Model Registry
# Cambialo se hai usato un nome diverso
MLFLOW_MODEL_NAME = "bank_marketing_rf_pipeline"
# URI moderno basato su alias, non usare stage
MLFLOW_MODEL_URI = f"models:/{MLFLOW_MODEL_NAME}@production"

# Dataset di riferimento (facoltativo ma utile per categorie e range)
BANK_DATA_PATH = ROOT_DIR / "data" / "bank_marketing_ml_ready.csv"

# Target e classe positiva per il problema di classificazione
TARGET_COLUMN = "y"
POSITIVE_CLASS = "yes"
