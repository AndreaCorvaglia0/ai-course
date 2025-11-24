"""
Configurazione dell'applicazione Bank Marketing.

Questo file contiene i parametri principali per caricare il modello
registrato in MLflow. I partecipanti possono modificare questi valori
per sperimentare con modelli diversi.
"""

# ==========================================
# CONFIGURAZIONE MLFLOW
# ==========================================

# URI del tracking server MLflow (locale)
MLFLOW_TRACKING_URI = "file:../mlruns"

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
