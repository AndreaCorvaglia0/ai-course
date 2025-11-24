from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Any

import mlflow
import mlflow.pyfunc
import mlflow.sklearn

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

from config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_MODEL_NAME,
    MLFLOW_MODEL_URI,
    TARGET_COLUMN,
    POSITIVE_CLASS,
)


def load_models():
    """
    Carica il modello registrato in MLflow usando l'URI con alias.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    pyfunc_model = mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)
    sklearn_model = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)

    return pyfunc_model, sklearn_model


def load_reference_data() -> pd.DataFrame | None:
    """
    Carica un dataset di riferimento, se presente.
    Serve per ricavare categorie e range numerici.
    Nota: ritorna None se il percorso non è configurato.
    """
    # Prova a cercare un file di dati nella cartella padre
    root_dir = Path(__file__).resolve().parents[1]
    possible_paths = [
        root_dir / "data" / "bank_marketing_ml_ready.csv",
        root_dir / "Dati" / "bank_marketing_prepared_classification.csv",
    ]
    
    for path in possible_paths:
        if path.exists():
            df = pd.read_csv(path)
            # Togliamo il target se presente
            if TARGET_COLUMN in df.columns:
                df = df.drop(columns=[TARGET_COLUMN])
            return df
    
    return None


def load_input_schema() -> List[Any]:
    """
    Ricava lo schema delle feature:
    - se la signature MLflow è disponibile, usa quella;
    - altrimenti ripiega sulle colonne del dataset di riferimento.
    Ritorna una lista di oggetti con .name e .type.
    """
    pyfunc_model, _ = load_models()
    signature = getattr(pyfunc_model.metadata, "signature", None)

    # Caso 1: signature presente
    if signature is not None and getattr(signature, "inputs", None) is not None:
        inputs = getattr(signature.inputs, "inputs", None)
        if inputs:
            return inputs

    # Caso 2: niente signature, usiamo il dataset
    df_ref = load_reference_data()
    if df_ref is None:
        return []

    cols = []
    for col_name, dtype in df_ref.dtypes.items():
        if col_name == TARGET_COLUMN:
            continue
        if pd.api.types.is_numeric_dtype(dtype):
            col_type = "double"
        else:
            col_type = "string"
        cols.append(SimpleNamespace(name=col_name, type=col_type))

    return cols


def _get_categorical_from_pipeline(pipeline: Any) -> Dict[str, List[str]]:
    """
    Prova a leggere le categorie dalle trasformazioni presenti nella pipeline.
    Gestisce:
    - feature_engine OneHotEncoder (via attributo encoder_dict_);
    - sklearn OneHotEncoder dentro ColumnTransformer.
    """
    cat_map: Dict[str, List[str]] = {}

    # Pipeline classica sklearn o feature_engine
    if hasattr(pipeline, "named_steps"):
        for name, step in pipeline.named_steps.items():
            # feature_engine OneHotEncoder: ha encoder_dict_
            if hasattr(step, "encoder_dict_"):
                for col, cats in step.encoder_dict_.items():
                    clean_cats = [c for c in cats if pd.notna(c)]
                    cat_map[col] = [str(c) for c in clean_cats]

            # ColumnTransformer inside pipeline
            if isinstance(step, ColumnTransformer):
                _update_cat_map_from_column_transformer(step, cat_map)

    # Caso limite: pipeline è direttamente un ColumnTransformer
    if isinstance(pipeline, ColumnTransformer):
        _update_cat_map_from_column_transformer(pipeline, cat_map)

    return cat_map


def _update_cat_map_from_column_transformer(
    ct: ColumnTransformer, cat_map: Dict[str, List[str]]
) -> None:
    """
    Estrae le categorie da un ColumnTransformer sklearn con OneHotEncoder.
    """
    for _, transformer, cols in ct.transformers_:
        if isinstance(transformer, SklearnOneHotEncoder):
            if hasattr(transformer, "categories_"):
                for col, cats in zip(cols, transformer.categories_):
                    clean_cats = [c for c in cats if pd.notna(c)]
                    cat_map[col] = [str(c) for c in clean_cats]


def infer_feature_config() -> Dict[str, Any]:
    """
    Costruisce una descrizione delle feature estraendo informazioni direttamente dalla pipeline MLflow.
    Compatibile con pipeline scikit-learn (ColumnTransformer) come quelle create in T4.

    Ritorna un dict con:
    - feature_meta: lista di dict {name, dtype, kind}
    - numeric_features: lista di nomi numerici
    - categorical_features: lista di nomi categorici (originali, non one-hot)
    - categories_by_feature: dict {feature -> [categoria1, ...]}
    - stats_numeric: dict {feature -> {min, max, median}}
    """
    _, pipeline = load_models()
    
    # Estrai lo step di preprocessing dalla pipeline
    preprocess_step = None
    if hasattr(pipeline, 'named_steps') and 'preprocess' in pipeline.named_steps:
        preprocess_step = pipeline.named_steps['preprocess']
    
    # Inizializza strutture dati
    feature_meta: List[Dict[str, Any]] = []
    numeric_features: List[str] = []
    categorical_features: List[str] = []
    categories_by_feature: Dict[str, List[str]] = {}
    stats_numeric: Dict[str, Dict[str, float]] = {}
    
    # Cerca dati di riferimento per statistiche
    df_ref = load_reference_data()
    
    # Caso 1: Pipeline scikit-learn con ColumnTransformer (come in T4)
    if isinstance(preprocess_step, ColumnTransformer):
        for name, transformer, cols in preprocess_step.transformers_:
            if name == 'remainder':
                continue
                
            # Binary encoder (OrdinalEncoder con 2 categorie)
            if name == 'bin' or name == 'binary':
                for col in cols:
                    feature_meta.append({"name": col, "dtype": "binary", "kind": "categorical"})
                    categorical_features.append(col)
                    # Per variabili binarie yes/no
                    categories_by_feature[col] = ["no", "yes"]
            
            # Ordinal encoder (OrdinalEncoder con ordine semantico)
            elif name == 'ord' or name == 'ordinal':
                if hasattr(transformer, 'categories_'):
                    for col, cats in zip(cols, transformer.categories_):
                        feature_meta.append({"name": col, "dtype": "ordinal", "kind": "categorical"})
                        categorical_features.append(col)
                        categories_by_feature[col] = [str(c) for c in cats]
                else:
                    # Fallback se non ancora fitted
                    for col in cols:
                        feature_meta.append({"name": col, "dtype": "ordinal", "kind": "categorical"})
                        categorical_features.append(col)
                        if df_ref is not None and col in df_ref.columns:
                            cats = sorted(df_ref[col].dropna().unique().tolist())
                            categories_by_feature[col] = cats
            
            # Nominal encoder (OneHotEncoder)
            elif name == 'nom' or name == 'nominal':
                if hasattr(transformer, 'categories_'):
                    for col, cats in zip(cols, transformer.categories_):
                        feature_meta.append({"name": col, "dtype": "string", "kind": "categorical"})
                        categorical_features.append(col)
                        categories_by_feature[col] = [str(c) for c in cats]
                else:
                    # Fallback se non ancora fitted
                    for col in cols:
                        feature_meta.append({"name": col, "dtype": "string", "kind": "categorical"})
                        categorical_features.append(col)
                        if df_ref is not None and col in df_ref.columns:
                            cats = sorted(df_ref[col].dropna().unique().tolist())
                            categories_by_feature[col] = cats
            
            # Numeric scaler (StandardScaler, RobustScaler, etc.)
            elif name == 'num' or name == 'numeric':
                for col in cols:
                    feature_meta.append({"name": col, "dtype": "double", "kind": "numeric"})
                    numeric_features.append(col)
                    
                    # Estrai statistiche dal dataset di riferimento
                    if df_ref is not None and col in df_ref.columns:
                        series = df_ref[col].dropna()
                        if not series.empty:
                            stats_numeric[col] = {
                                "min": float(series.min()),
                                "max": float(series.max()),
                                "median": float(series.median()),
                                "mean": float(series.mean()),
                            }
    
    # Fallback: se non riusciamo a estrarre dalla pipeline, usa il dataset
    if not feature_meta and df_ref is not None:
        for col in df_ref.columns:
            dtype = df_ref[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                feature_meta.append({"name": col, "dtype": "double", "kind": "numeric"})
                numeric_features.append(col)
                series = df_ref[col].dropna()
                if not series.empty:
                    stats_numeric[col] = {
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "median": float(series.median()),
                    }
            else:
                feature_meta.append({"name": col, "dtype": "string", "kind": "categorical"})
                categorical_features.append(col)
                cats = sorted(df_ref[col].dropna().unique().tolist())
                categories_by_feature[col] = cats
    
    return {
        "feature_meta": feature_meta,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "categories_by_feature": categories_by_feature,
        "stats_numeric": stats_numeric,
    }


def make_prediction(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Esegue la predizione con il modello caricato da MLflow.

    input_dict: {feature_name: value} - valori categorici originali (es. job="student")

    Ritorna:
    - per classificazione: {
        "positive_class": ...,
        "proba_positive": float,
        "proba_raw": [p_class_0, p_class_1, ...],
        "classes": [...],  # per debug
      }
    - per regressione: {
        "prediction": float
      }
    """
    _, pipeline = load_models()
    config = infer_feature_config()
    
    # Le feature originali (prima dell'encoding)
    original_features = [f["name"] for f in config["feature_meta"]]
    
    # Costruiamo DataFrame con le feature originali
    # La pipeline si occuperà di tutte le trasformazioni (ordinal, one-hot, scaling)
    X = pd.DataFrame([input_dict])
    X = X.reindex(columns=original_features)

    # Caso classificazione
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)[0]

        # Recupera il classificatore finale se è un Pipeline sklearn/feature_engine
        clf = pipeline
        if isinstance(pipeline, SklearnPipeline) and pipeline.steps:
            clf = pipeline.steps[-1][1]

        # Cerca la posizione della classe positiva
        if hasattr(clf, "classes_"):
            classes = list(clf.classes_)
            
            # Il target può essere "yes"/"no" oppure 1/0
            # Proviamo prima "yes", poi 1, poi "1"
            idx_pos = None
            for positive_label in [POSITIVE_CLASS, 1, "1", True]:
                if positive_label in classes:
                    idx_pos = classes.index(positive_label)
                    break
            
            # Se non trovato, assumiamo che la classe positiva sia l'ultima
            if idx_pos is None:
                idx_pos = len(classes) - 1
        else:
            # Fallback: assumiamo indice 1 per classe positiva
            idx_pos = min(1, len(proba) - 1)

        return {
            "positive_class": POSITIVE_CLASS,
            "proba_positive": float(proba[idx_pos]),
            "proba_raw": proba.tolist(),
            "classes": classes if hasattr(clf, "classes_") else "unknown",
            "idx_pos": idx_pos,
        }

    # Caso regressione
    y_pred = pipeline.predict(X)[0]
    return {"prediction": float(y_pred)}

