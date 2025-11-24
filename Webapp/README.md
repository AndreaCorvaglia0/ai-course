# Webapp – Bank Marketing AI Predictor

Webapp Streamlit che carica un modello registrato in MLflow e permette di stimare la probabilità che un cliente sottoscriva un deposito a termine.

## Struttura

- `app_bank.py`: applicazione Streamlit per predizioni Bank Marketing
- `app_house.py`: applicazione Streamlit per predizioni House Sales
- `config.py`: configurazione modello MLflow (IMPORTANTE: modificabile dai partecipanti)
- `model_utils.py`: funzioni di utilità per caricare modelli e fare predizioni
- `assets/styles.css`: stile CSS personalizzato della webapp
- `requirements.txt`: dipendenze Python necessarie

## Configurazione

Prima di lanciare l'app, verifica il file `config.py`:

```python
MLFLOW_TRACKING_URI = "file:../mlruns"          # URI del tracking server
MLFLOW_MODEL_NAME = "bank_marketing_model"      # Nome del modello registrato
MLFLOW_MODEL_ALIAS = "production"                # Alias del modello da caricare
```

**Nota per i partecipanti**: Puoi modificare `MLFLOW_MODEL_NAME` e `MLFLOW_MODEL_ALIAS` per testare modelli diversi registrati in MLflow.

## Come eseguire

### Dalla root del progetto (cartella 3.0):

```bash
streamlit run Webapp/app_bank.py
```

L'applicazione si aprirà automaticamente nel browser all'indirizzo http://localhost:8501

### Alternative:

Se preferisci, puoi anche spostarti nella cartella Webapp:

```bash
cd Webapp
streamlit run app_bank.py
```

## Funzionalità

- **Caricamento dinamico del modello**: legge automaticamente il modello dal Model Registry MLflow
- **Interfaccia adattiva**: genera controlli UI basati sulle feature del modello
- **Preprocessing automatico**: la pipeline gestisce tutte le trasformazioni necessarie
- **Predizione in tempo reale**: calcola la probabilità di sottoscrizione immediatamente
- **Visualizzazione avanzata**: grafici, probabilità, raccomandazioni

## Troubleshooting

**Errore: "Model not found"**
- Verifica che il modello sia registrato in MLflow con il nome corretto
- Controlla che l'alias sia stato assegnato (es. `production`)
- Verifica che il tracking URI sia corretto (`file:../mlruns`)

**Errore: "Feature mismatch"**
- Il modello si aspetta feature diverse da quelle fornite
- Assicurati di aver addestrato il modello con le feature corrette
- Verifica che il dataset di riferimento sia presente in `Dati/`

## Requisiti

- Python 3.11+
- Streamlit
- MLflow
- scikit-learn
- pandas
- numpy

Installa le dipendenze con:

```bash
pip install -r requirements.txt
```
