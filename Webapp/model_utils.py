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
    BANK_DATA_PATH,
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
    """
    if BANK_DATA_PATH.exists():
        df = pd.read_csv(BANK_DATA_PATH)
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
    Costruisce una descrizione delle feature che serve all'interfaccia Streamlit.

    Ritorna un dict con:
    - feature_meta: lista di dict {name, dtype, kind}
    - numeric_features: lista di nomi numerici
    - categorical_features: lista di nomi categorici
    - categories_by_feature: dict {feature -> [categoria1, ...]}
    - stats_numeric: dict {feature -> {min, max, median}}
    """
    _, pipeline = load_models()
    input_schema = load_input_schema()
    df_ref = load_reference_data()

    # Prova a leggere categorie dal pipeline
    cat_from_encoder = _get_categorical_from_pipeline(pipeline)

    feature_meta: List[Dict[str, Any]] = []
    numeric_features: List[str] = []
    categorical_features: List[str] = []

    for col in input_schema:
        name = col.name
        dtype = str(col.type).lower() if hasattr(col, "type") else "unknown"

        if name in cat_from_encoder:
            kind = "categorical"
        elif dtype in ("string", "binary"):
            kind = "categorical"
        else:
            kind = "numeric"

        feature_meta.append(
            {
                "name": name,
                "dtype": dtype,
                "kind": kind,
            }
        )

        if kind == "categorical":
            categorical_features.append(name)
        else:
            numeric_features.append(name)

    # Costruzione mappa categorie
    categories_by_feature: Dict[str, List[str]] = {}

    # 1) dal pipeline, se disponibile
    if cat_from_encoder:
        categories_by_feature.update(cat_from_encoder)

    # 2) fallback dal dataset
    if df_ref is not None:
        for feat in categorical_features:
            if feat not in categories_by_feature and feat in df_ref.columns:
                cats = (
                    df_ref[feat]
                    .dropna()
                    .astype(str)
                    .drop_duplicates()
                    .sort_values()
                    .tolist()
                )
                categories_by_feature[feat] = cats

    # Statistiche numeriche per suggerire range e default
    stats_numeric: Dict[str, Dict[str, float]] = {}
    if df_ref is not None:
        for feat in numeric_features:
            if feat in df_ref.columns:
                series = df_ref[feat].dropna()
                if not series.empty:
                    stats_numeric[feat] = {
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "median": float(series.median()),
                    }

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

    input_dict: {feature_name: value}

    Ritorna:
    - per classificazione: {
        "positive_class": ...,
        "proba_positive": float,
        "proba_raw": [p_class_0, p_class_1, ...],
      }
    - per regressione: {
        "prediction": float
      }
    """
    _, pipeline = load_models()
    config = infer_feature_config()
    ordered_cols = [f["name"] for f in config["feature_meta"]]

    # Costruiamo DataFrame con una sola riga, ordinando le colonne
    X = pd.DataFrame([input_dict])
    X = X.reindex(columns=ordered_cols)

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
            if POSITIVE_CLASS in classes:
                idx_pos = classes.index(POSITIVE_CLASS)
            else:
                idx_pos = int(np.argmax(proba))
        else:
            idx_pos = int(np.argmax(proba))

        return {
            "positive_class": POSITIVE_CLASS,
            "proba_positive": float(proba[idx_pos]),
            "proba_raw": proba.tolist(),
        }

    # Caso regressione
    y_pred = pipeline.predict(X)[0]
    return {"prediction": float(y_pred)}
