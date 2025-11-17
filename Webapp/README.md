# Webapp – Bank Marketing

Webapp Streamlit che carica un modello registrato in MLflow e permette
di stimare la probabilità che un cliente sottoscriva il deposito.

## Struttura

- `app.py`: applicazione Streamlit
- `config.py`: configurazione modello, tracking URI, dataset di riferimento
- `model_utils.py`: funzioni di utilità per MLflow, schema delle feature e predizione
- `assets/styles.css`: stile della webapp
- `requirements.txt`: dipendenze minime

## Come eseguire

1. Attiva l'ambiente Python usato per il progetto MLOps.
2. Posizionati nella cartella `Webapp`:

   ```bash
   cd Webapp
