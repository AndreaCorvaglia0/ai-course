"""
Configurazione webapp Wine Quality Prediction
"""
from pathlib import Path

# Percorsi
BASE_DIR = Path(__file__).resolve().parent
MLFLOW_TRACKING_URI = f"file:{BASE_DIR}/mlruns"

# Modello MLflow Registry
MLFLOW_MODEL_NAME = "wine_clf"
MLFLOW_MODEL_ALIAS = "production"

# Feature ranges (basati sul dataset UCI Wine Quality)
FEATURE_RANGES = {
    "fixed_acidity": (4.0, 16.0),
    "volatile_acidity": (0.1, 1.6),
    "citric_acid": (0.0, 1.0),
    "residual_sugar": (0.5, 65.0),
    "chlorides": (0.01, 0.6),
    "free_sulfur_dioxide": (1.0, 72.0),
    "total_sulfur_dioxide": (6.0, 440.0),
    "density": (0.990, 1.040),
    "pH": (2.7, 4.0),
    "sulphates": (0.3, 2.0),
    "alcohol": (8.0, 15.0)
}

# Default values (valori medi tipici per un buon vino)
FEATURE_DEFAULTS = {
    "fixed_acidity": 7.5,
    "volatile_acidity": 0.4,
    "citric_acid": 0.3,
    "residual_sugar": 2.5,
    "chlorides": 0.08,
    "free_sulfur_dioxide": 30.0,
    "total_sulfur_dioxide": 100.0,
    "density": 0.996,
    "pH": 3.3,
    "sulphates": 0.6,
    "alcohol": 10.5
}

# Descrizioni features per tooltips
FEATURE_DESCRIPTIONS = {
    "fixed_acidity": "Acidità fissa (acido tartarico): contribuisce al gusto fresco e alla conservazione del vino. Valori tipici: 4-16 g/L",
    "volatile_acidity": "Acidità volatile (acido acetico): valori alti (>0.8) possono dare sapore di aceto. Valori ideali: 0.2-0.6 g/L",
    "citric_acid": "Acido citrico: aggiunge freschezza e note agrumate al vino. Piccole quantità (0-1 g/L) migliorano il bilanciamento",
    "residual_sugar": "Zuccheri residui dopo fermentazione: <4 g/L = vino secco, 4-12 g/L = semi-secco, >45 g/L = dolce",
    "chlorides": "Cloruri (contenuto di sale): in piccole quantità esalta il sapore, troppo sale rende il vino poco gradevole",
    "free_sulfur_dioxide": "SO₂ libero: forma attiva che previene ossidazione e crescita microbica. Essenziale per conservazione (1-72 mg/L)",
    "total_sulfur_dioxide": "SO₂ totale (libero + legato): antimicrobico e antiossidante. Troppo può essere percepito nel gusto (6-440 mg/L)",
    "density": "Densità del vino: dipende da alcol (la riduce) e zuccheri (la aumenta). Range tipico: 0.990-1.040 g/cm³",
    "pH": "Livello di acidità: vini tipicamente tra 2.9-4.0. pH più basso = più acido, influenza gusto e stabilità",
    "sulphates": "Solfati (potassio solfato): additivo che migliora conservazione e qualità del vino (0.3-2.0 g/L)",
    "alcohol": "Gradazione alcolica: influenza corpo, struttura e conservazione. Vini tipici: 8-15% vol"
}

# Unità di misura
FEATURE_UNITS = {
    "fixed_acidity": "g/L",
    "volatile_acidity": "g/L",
    "citric_acid": "g/L",
    "residual_sugar": "g/L",
    "chlorides": "g/L",
    "free_sulfur_dioxide": "mg/L",
    "total_sulfur_dioxide": "mg/L",
    "density": "g/cm³",
    "pH": "",
    "sulphates": "g/L",
    "alcohol": "% vol"
}

# Soglie per visualizzazione
QUALITY_THRESHOLDS = {
    "excellent": 0.50,  
    "good": 0.30,       
    "medium": 0.15
    }

# Colori tema cantina
COLORS = {
    "excellent": "#8B0000",     # Bordeaux scuro
    "good": "#CD5C5C",          # Rosso corallo
    "medium": "#DAA520",        # Oro antico
    "low": "#696969"            # Grigio scuro
}
