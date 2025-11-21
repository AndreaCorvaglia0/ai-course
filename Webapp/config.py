from pathlib import Path

# Root del progetto
ROOT_DIR = Path(__file__).resolve().parents[1]

# MLflow configurazione
MLFLOW_TRACKING_URI = f"file:{ROOT_DIR / 'mlruns'}"
MLFLOW_MODEL_NAME = "bank_marketing_rf_pipeline"
MLFLOW_MODEL_URI = f"models:/{MLFLOW_MODEL_NAME}@production"

# Dataset di riferimento per categorie e range
BANK_DATA_PATH = ROOT_DIR / "data" / "bank_marketing_ml_ready.csv"

# Classificazione
TARGET_COLUMN = "y"
POSITIVE_CLASS = "yes"
