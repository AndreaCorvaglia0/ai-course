"""
Configurazione webapp Wine Quality Prediction
"""
import os
from pathlib import Path

# Opt-in necessario nelle versioni recenti per usare il backend file legacy.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

# Percorsi
BASE_DIR = Path(__file__).resolve().parent
MLFLOW_TRACKING_URI = f"file:{BASE_DIR}/mlruns"

# Modello MLflow Registry
MLFLOW_MODEL_NAME = "wine_clf"
MLFLOW_MODEL_ALIAS = "production"

# Informazioni sul corso (header e footer)
COURSE_NAME = "AI for Data Analysis"
COURSE_REPO_URL = "https://github.com/AndreaCorvaglia0/ai-course"

# Quota di vini di alta qualità (voto >= 7) nel dataset UCI Wine Quality
BASE_HIGH_QUALITY_RATE = 0.197

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

# Default values (mediane del dataset: un lotto "tipico")
FEATURE_DEFAULTS = {
    "fixed_acidity": 7.0,
    "volatile_acidity": 0.29,
    "citric_acid": 0.31,
    "residual_sugar": 3.0,
    "chlorides": 0.047,
    "free_sulfur_dioxide": 29.0,
    "total_sulfur_dioxide": 118.0,
    "density": 0.9949,
    "pH": 3.21,
    "sulphates": 0.51,
    "alcohol": 10.3
}

# Nomi in italiano mostrati nell'interfaccia
FEATURE_LABELS = {
    "fixed_acidity": "Acidità fissa",
    "volatile_acidity": "Acidità volatile",
    "citric_acid": "Acido citrico",
    "residual_sugar": "Zuccheri residui",
    "chlorides": "Cloruri",
    "free_sulfur_dioxide": "SO₂ libera",
    "total_sulfur_dioxide": "SO₂ totale",
    "density": "Densità",
    "pH": "pH",
    "sulphates": "Solfati",
    "alcohol": "Alcol"
}

# Passo e formato degli slider (decimali leggibili per ogni grandezza)
FEATURE_STEPS = {
    "fixed_acidity": (0.1, "%.1f"),
    "volatile_acidity": (0.01, "%.2f"),
    "citric_acid": (0.01, "%.2f"),
    "residual_sugar": (0.1, "%.1f"),
    "chlorides": (0.001, "%.3f"),
    "free_sulfur_dioxide": (1.0, "%.0f"),
    "total_sulfur_dioxide": (1.0, "%.0f"),
    "density": (0.0001, "%.4f"),
    "pH": (0.01, "%.2f"),
    "sulphates": (0.01, "%.2f"),
    "alcohol": (0.1, "%.1f")
}

# Cosa misura ogni feature (nota mostrata sotto ogni slider)
FEATURE_DESCRIPTIONS = {
    "fixed_acidity": "Acidi non volatili, soprattutto tartarico: danno freschezza e struttura e aiutano la conservazione.",
    "volatile_acidity": "Soprattutto acido acetico. Valori alti danno odore di aceto: è il principale segnale di difetto.",
    "citric_acid": "Presente in piccole quantità, aggiunge freschezza e note agrumate.",
    "residual_sugar": "Zucchero rimasto dopo la fermentazione: sotto 4 g/L il vino è secco, oltre 45 g/L è dolce.",
    "chlorides": "Quantità di sale nel vino: valori alti danno un gusto salato poco gradito.",
    "free_sulfur_dioxide": "Forma attiva della SO₂, protegge da ossidazione e microbi. Oltre 50 mg/L si sente al naso e in bocca.",
    "total_sulfur_dioxide": "SO₂ libera più quella legata. I bianchi ne contengono in genere molta più dei rossi.",
    "density": "Vicina a quella dell'acqua: l'alcol la abbassa, gli zuccheri la alzano.",
    "pH": "Misura l'acidità: più è basso, più il vino è acido. Quasi tutti i vini stanno tra 3 e 4.",
    "sulphates": "Additivo (solfato di potassio) che aumenta la SO₂ e quindi protegge il vino.",
    "alcohol": "Gradazione alcolica. Nei dati è la grandezza più legata alla qualità: i vini migliori ne hanno di più."
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

# Valori tipici nel dataset: (5° percentile, 95° percentile, mediana dei vini di alta qualità)
FEATURE_TYPICAL = {
    "fixed_acidity": (5.7, 9.8, 6.9),
    "volatile_acidity": (0.16, 0.67, 0.27),
    "citric_acid": (0.05, 0.56, 0.32),
    "residual_sugar": (1.2, 15.0, 2.9),
    "chlorides": (0.028, 0.102, 0.039),
    "free_sulfur_dioxide": (6.0, 61.0, 31.0),
    "total_sulfur_dioxide": (19.0, 206.0, 114.0),
    "density": (0.9899, 0.9994, 0.9923),
    "pH": (2.97, 3.50, 3.22),
    "sulphates": (0.35, 0.79, 0.51),
    "alcohol": (9.0, 12.7, 11.5)
}

# Gruppi di input: (titolo, icona, sottotitolo, feature)
FEATURE_GROUPS = [
    ("Acidità", ":material/science:", "Freschezza e difetti", ["fixed_acidity", "volatile_acidity", "citric_acid", "pH"]),
    ("Corpo e dolcezza", ":material/water_drop:", "Alcol, zuccheri e densità", ["alcohol", "residual_sugar", "density"]),
    ("Conservanti e sali", ":material/shield:", "Protezione e sapidità", ["free_sulfur_dioxide", "total_sulfur_dioxide", "sulphates", "chlorides"]),
]

# Lotti di esempio (profili mediani reali del dataset)
EXAMPLE_LOTS = {
    "Rosso strutturato": {
        "summary": "Profilo mediano dei rossi di alta qualità: alcol alto, solfati alti",
        "values": {"fixed_acidity": 8.7, "volatile_acidity": 0.37, "citric_acid": 0.40, "residual_sugar": 2.3,
                   "chlorides": 0.073, "free_sulfur_dioxide": 11.0, "total_sulfur_dioxide": 27.0, "density": 0.9960,
                   "pH": 3.27, "sulphates": 0.74, "alcohol": 11.6},
    },
    "Bianco equilibrato": {
        "summary": "Profilo mediano dei vini di alta qualità: buono, ma non ancora eccellente",
        "values": {"fixed_acidity": 6.9, "volatile_acidity": 0.27, "citric_acid": 0.32, "residual_sugar": 2.9,
                   "chlorides": 0.039, "free_sulfur_dioxide": 31.0, "total_sulfur_dioxide": 114.0, "density": 0.9923,
                   "pH": 3.22, "sulphates": 0.51, "alcohol": 11.5},
    },
    "Bianco da tavola": {
        "summary": "Profilo mediano dei bianchi di bassa qualità: poco alcol, poca SO₂ libera",
        "values": {"fixed_acidity": 6.9, "volatile_acidity": 0.32, "citric_acid": 0.30, "residual_sugar": 2.7,
                   "chlorides": 0.046, "free_sulfur_dioxide": 18.0, "total_sulfur_dioxide": 119.0, "density": 0.9940,
                   "pH": 3.16, "sulphates": 0.47, "alcohol": 10.1},
    },
    "Rosso difettoso": {
        "summary": "Profilo mediano dei rossi di bassa qualità: acidità volatile alta",
        "values": {"fixed_acidity": 7.5, "volatile_acidity": 0.68, "citric_acid": 0.08, "residual_sugar": 2.1,
                   "chlorides": 0.080, "free_sulfur_dioxide": 9.0, "total_sulfur_dioxide": 26.0, "density": 0.9970,
                   "pH": 3.38, "sulphates": 0.56, "alcohol": 10.0},
    },
}

# Soglie per visualizzazione
QUALITY_THRESHOLDS = {
    "excellent": 0.50,
    "good": 0.30,
    "medium": 0.15
    }

# Livelli di qualità: chiave soglia -> (livello, colore badge, icona, azione, dettaglio)
# Colori badge: red, orange, yellow, blue, green, violet, gray (definiti in .streamlit/config.toml)
QUALITY_LEVELS = {
    "excellent": ("Eccellente", "red", ":material/workspace_premium:", "Affinamento in barrique",
                  "Lotto ideale per l'invecchiamento in botti di rovere e il mercato premium."),
    "good": ("Buono", "orange", ":material/thumb_up:", "Affinamento controllato",
             "Lotto promettente: consigliato un affinamento breve, da ricontrollare con l'enologo."),
    "medium": ("Medio", "yellow", ":material/local_shipping:", "Imbottigliamento diretto",
               "Lotto da commercializzare subito, senza investire in affinamento."),
    "low": ("Base", "gray", ":material/blender:", "Assemblaggio",
            "Lotto da usare per blend o prodotti entry-level."),
}
