"""
Utilità per caricamento modello MLflow e predizioni
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import json
import joblib
from pathlib import Path
from typing import Dict, Tuple, List
import config


def setup_mlflow() -> None:
    """Configura MLflow tracking URI"""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)


def load_model_from_mlflow():
    """
    Carica il modello con strategia a fallback:
    1. Prova dal MLflow Registry (nome e alias dal config)
    2. Se fallisce, cerca cartella 'model' nella working directory
    
    Legge dinamicamente le feature dal modello caricato.
    
    Returns:
        tuple: (model, run_info_dict, feature_names)
    """
    # TENTATIVO 1: MLflow Registry
    try:
        setup_mlflow()
        
        # Costruisci URI del modello dal registry
        model_uri = f"models:/{config.MLFLOW_MODEL_NAME}@{config.MLFLOW_MODEL_ALIAS}"
        
        # Carica il modello dal registry
        model = mlflow.sklearn.load_model(model_uri)
        
        # Estrai feature names DIRETTAMENTE dal modello caricato
        feature_names = extract_feature_names_from_model(model)
        
        # Ottieni informazioni sulla versione del modello
        client = mlflow.MlflowClient()
        try:
            # Cerca la versione con l'alias specificato
            model_versions = client.get_model_version_by_alias(
                config.MLFLOW_MODEL_NAME, 
                config.MLFLOW_MODEL_ALIAS
            )
            run_id = model_versions.run_id
            version = model_versions.version
            
            # Cerca il run per ottenere le metriche
            run = client.get_run(run_id)
            
            run_info = {
                "run_id": run_id,
                "run_name": run.data.tags.get("mlflow.runName", "N/A"),
                "model_name": config.MLFLOW_MODEL_NAME,
                "model_version": version,
                "model_alias": config.MLFLOW_MODEL_ALIAS,
                "source": "MLflow Registry",
                "accuracy": run.data.metrics.get("accuracy", 0.0),
                "precision": run.data.metrics.get("precision", 0.0),
                "recall": run.data.metrics.get("recall", 0.0),
                "f1_score": run.data.metrics.get("f1_score", 0.0),
                "roc_auc": run.data.metrics.get("roc_auc", 0.0)
            }
        except Exception:
            # Fallback info se non riesce a ottenere dal registry
            run_info = {
                "run_id": "N/A",
                "run_name": "N/A",
                "model_name": config.MLFLOW_MODEL_NAME,
                "model_version": "N/A",
                "model_alias": config.MLFLOW_MODEL_ALIAS,
                "source": "MLflow Registry",
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "roc_auc": 0.0
            }
        
        return model, run_info, feature_names
        
    except Exception as e:
        # TENTATIVO 2: Cartella locale 'model'
        print(f"⚠️ MLflow Registry non disponibile: {e}")
        print("🔍 Cercando modello nella cartella locale 'model'...")
        
        try:
            # Cerca cartella model nella working directory
            model_dir = Path(__file__).parent / "model"
            
            if not model_dir.exists():
                raise FileNotFoundError(f"Cartella 'model' non trovata in {model_dir.parent}")
            
            # Cerca file del modello (prova diversi formati)
            model_file = None
            for pattern in ["model.pkl", "model.joblib", "*.pkl", "*.joblib"]:
                matches = list(model_dir.glob(pattern))
                if matches:
                    model_file = matches[0]
                    break
            
            if not model_file:
                raise FileNotFoundError(f"Nessun file modello trovato in {model_dir}")
            
            print(f"✓ Trovato modello: {model_file.name}")
            
            # Carica il modello
            model = joblib.load(model_file)
            
            # Estrai feature names
            feature_names = extract_feature_names_from_model(model)
            
            # Info minimali per il modello locale
            run_info = {
                "run_id": "local",
                "run_name": model_file.stem,
                "model_name": "model",
                "model_version": "local",
                "model_alias": "local",
                "source": f"Local File ({model_file.name})",
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "roc_auc": 0.0
            }
            
            return model, run_info, feature_names
            
        except Exception as e2:
            raise RuntimeError(
                f"Impossibile caricare il modello né da MLflow Registry né da cartella locale.\n"
                f"Errore MLflow: {e}\n"
                f"Errore locale: {e2}"
            )


def extract_feature_names_from_model(model) -> List[str]:
    """
    Estrae DINAMICAMENTE i nomi delle feature dal modello caricato.
    Questo determina quale sottoinsieme di feature mostrare nell'interfaccia.
    
    Args:
        model: Pipeline sklearn caricata da MLflow
        
    Returns:
        list: Lista di nomi delle features richieste dal modello
    """
    # Metodo 1: Usa feature_names_in_ (sklearn >= 1.0)
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    
    # Metodo 2: Se è una Pipeline, prova il primo step
    if hasattr(model, 'steps'):
        first_step = model.steps[0][1]
        if hasattr(first_step, 'feature_names_in_'):
            return list(first_step.feature_names_in_)
    
    # Metodo 3: Prova con n_features_in_ e genera nomi generici
    if hasattr(model, 'n_features_in_'):
        n_features = model.n_features_in_
        # Usa le prime n_features dal config in ordine
        all_features = list(config.FEATURE_RANGES.keys())
        return all_features[:n_features]
    
    # Fallback: usa tutte le feature dal config
    return list(config.FEATURE_RANGES.keys())


def predict_wine_quality(model, features: Dict[str, float]) -> Tuple[int, float]:
    """
    Effettua predizione sulla qualità del vino.
    
    Args:
        model: Modello MLflow caricato
        features: Dizionario con valori delle features
        
    Returns:
        tuple: (classe_predetta, probabilità_alta_qualità)
    """
    # Crea DataFrame con le features
    df = pd.DataFrame([features])
    
    # Predizione
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0, 1]  # Probabilità classe positiva (alta qualità)
    
    return int(prediction), float(probability)


def get_quality_recommendation(probability: float) -> Tuple[str, str, str]:
    """
    Restituisce la raccomandazione basata sulla probabilità.
    
    Args:
        probability: Probabilità di alta qualità (0-1)
        
    Returns:
        tuple: (livello, raccomandazione, colore)
    """
    if probability >= config.QUALITY_THRESHOLDS["excellent"]:
        return (
            "Eccellente",
            "🍷 **Affinamento in Barrique** - Lotto ideale per invecchiamento in cantina di pregio",
            config.COLORS["excellent"]
        )
    elif probability >= config.QUALITY_THRESHOLDS["good"]:
        return (
            "Buono",
            "🍇 **Affinamento Controllato** - Lotto promettente, consigliato affinamento breve",
            config.COLORS["good"]
        )
    elif probability >= config.QUALITY_THRESHOLDS["medium"]:
        return (
            "Medio",
            "📦 **Imbottigliamento Diretto** - Lotto da commercializzare senza affinamento",
            config.COLORS["medium"]
        )
    else:
        return (
            "Base",
            "⚗️ **Assemblaggio** - Lotto da utilizzare per blend o prodotti entry-level",
            config.COLORS["low"]
        )
