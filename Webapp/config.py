"""
Configurazione dell'applicazione Bank Marketing.

Questo file contiene i parametri principali per caricare il modello
registrato in MLflow. I partecipanti possono modificare questi valori
per sperimentare con modelli diversi.
"""
import os
from pathlib import Path

# Opt-in necessario nelle versioni recenti per usare il backend file legacy.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

# ==========================================
# CONFIGURAZIONE MLFLOW
# ==========================================

# URI del tracking server MLflow (locale): cartella mlruns nella root del progetto.
# Il percorso parte dalla posizione di questo file, quindi funziona
# qualunque sia la cartella da cui si lancia l'app.
MLRUNS_DIR = Path(__file__).resolve().parent.parent / "mlruns"
MLFLOW_TRACKING_URI = MLRUNS_DIR.as_uri()

# Nome del modello registrato nel Model Registry
# NOTA: Modificare questo valore se hai registrato il modello con un nome diverso
MLFLOW_MODEL_NAME = "bank_marketing_model"

# Alias del modello da caricare (production, staging, champion)
MLFLOW_MODEL_ALIAS = "production"

# URI completo per caricare il modello
MLFLOW_MODEL_URI = f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"


# ==========================================
# CONFIGURAZIONE DATASET
# ==========================================

# Nome della colonna target nel dataset originale
TARGET_COLUMN = "y"

# Classe positiva per la classificazione (sottoscrizione del deposito)
POSITIVE_CLASS = "yes"
