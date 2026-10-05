# 🍷 Wine Quality Assessment Webapp

Sistema di valutazione della qualità del vino per supportare le decisioni di affinamento in cantina.

## 📋 Descrizione

Webapp Streamlit moderna ed elegante che utilizza modelli di Machine Learning per valutare la qualità del vino basandosi su analisi chimico-fisiche. Il sistema aiuta l'enologo a decidere quali lotti destinare all'affinamento in barrique.

## 🚀 Avvio

```bash
cd capstone_project
uv run streamlit run app.py
```

## 🎯 Funzionalità

### Caricamento Dinamico del Modello
- **Strategia a fallback intelligente**:
  1. Prova a caricare dal **MLflow Registry** (`wine_clf@production`)
  2. Se fallisce, cerca una cartella `model/` nella working directory
- Estrae dinamicamente le feature richieste dal modello
- Nome del modello configurabile in `config.py`
- Supporta formati: `.pkl`, `.joblib`

### Interface Intelligente
- **11 sliders interattivi** per parametri chimico-fisici
- **Tooltip informativi** (hover su ⓘ) per ogni parametro
- Visualizza solo le feature richieste dal modello specifico
- **🎲 Generazione Random**: crea campioni di vino casuali per testare il modello
- **Predizione Automatica**: aggiorna la valutazione istantaneamente al cambio dei parametri

### Valutazione Dinamica
Il sistema fornisce 4 livelli di raccomandazione:

| Livello | Probabilità | Raccomandazione |
|---------|-------------|-----------------|
| 🍷 **Eccellente** | ≥ 50% | Affinamento in Barrique - Invecchiamento in cantina di pregio |
| 🍇 **Buono** | 30-49% | Affinamento Controllato - Affinamento breve |
| 📦 **Medio** | 15-29% | Imbottigliamento Diretto - Commercializzazione immediata |
| ⚗️ **Base** | < 15% | Assemblaggio - Utilizzo per blend |

### Design Moderno
- **Tema cantina**: sfondo gradient bordeaux/marrone
- **Colori dinamici**: cambiano in base alla qualità predetta
- **Font eleganti**: Playfair Display + Lato
- **Layout responsivo**: 2 colonne (input | risultato)

## ⚙️ Configurazione

Modifica `config.py` per personalizzare:

```python
# Nome del modello nel registry
MLFLOW_MODEL_NAME = "wine_clf"
MLFLOW_MODEL_ALIAS = "production"

# Soglie di qualità
QUALITY_THRESHOLDS = {
    "excellent": 0.50,
    "good": 0.30,
    "medium": 0.15
}
```

## 📊 Features Analizzate

Il modello analizza 11 parametri chimico-fisici:

1. **Fixed Acidity** (g/L) - Acidità fissa (acido tartarico)
2. **Volatile Acidity** (g/L) - Acidità volatile (acido acetico)
3. **Citric Acid** (g/L) - Acido citrico
4. **Residual Sugar** (g/L) - Zuccheri residui
5. **Chlorides** (g/L) - Cloruri (sale)
6. **Free Sulfur Dioxide** (mg/L) - SO₂ libero
7. **Total Sulfur Dioxide** (mg/L) - SO₂ totale
8. **Density** (g/cm³) - Densità
9. **pH** - Livello di acidità
10. **Sulphates** (g/L) - Solfati
11. **Alcohol** (% vol) - Gradazione alcolica

## 🔧 Dipendenze

Vedi `requirements.txt`:
- streamlit
- mlflow
- scikit-learn
- pandas
- numpy

## 📁 Struttura

```
capstone_project/
├── app.py              # Webapp Streamlit
├── config.py           # Configurazione
├── model_utils.py      # Utilità modello MLflow
├── development.ipynb   # Training del modello
├── mlruns/             # MLflow tracking
├── model/              # (Opzionale) Modello locale come fallback
│   └── model.pkl       # Pipeline sklearn serializzata
└── requirements.txt    # Dipendenze
```

## 🔧 Deployment

### Opzione 1: Con MLflow Registry (Raccomandato)
```bash
# Il modello viene caricato automaticamente dal registry
uv run streamlit run app.py
```

### Opzione 2: Con Modello Locale
Se MLflow non è disponibile, crea una cartella `model/`:
```bash
mkdir model
# Copia il tuo modello (pipeline.pkl o model.pkl)
cp /path/to/your/model.pkl model/
uv run streamlit run app.py
```

La webapp rileverà automaticamente la fonte migliore disponibile.

## 🎓 Dataset

UCI Machine Learning Repository - Wine Quality Dataset (ID: 186)
- Vini portoghesi "Vinho Verde"
- 6,497 campioni
- Classificazione binaria: alta qualità (≥7) vs standard
