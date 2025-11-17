from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    MLFLOW_MODEL_NAME,
    MLFLOW_MODEL_URI,
)
from model_utils import (
    infer_feature_config,
    make_prediction,
    load_reference_data,
)


def load_css():
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Bank Marketing – Propensione al deposito",
        layout="centered",
    )

    load_css()

    # Header
    st.markdown(
        """
        <div class="page-header">
            <h1>Bank Marketing – Propensione al deposito</h1>
            <p>Demo webapp collegata al modello registrato in MLflow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar con info modello
    st.sidebar.title("Configurazione modello")
    st.sidebar.markdown(f"**Nome modello**  \n`{MLFLOW_MODEL_NAME}`")
    st.sidebar.markdown(f"**Model URI**  \n`{MLFLOW_MODEL_URI}`")

    # Carichiamo descrizione delle feature
    config = infer_feature_config()
    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]
    categories_by_feature = config["categories_by_feature"]
    stats_numeric = config["stats_numeric"]

    # Expander con schema
    with st.expander("Schema delle feature del modello"):
        df_schema = pd.DataFrame(config["feature_meta"])
        st.dataframe(df_schema, use_container_width=True)

    st.markdown("### Inserisci le caratteristiche del cliente")

    input_values = {}

    col_left, col_right = st.columns(2)

    # Feature categoriche
    with col_left:
        for feat in categorical_features:
            cats = categories_by_feature.get(feat, [])
            default_idx = 0 if cats else None
            input_values[feat] = st.selectbox(
                feat,
                options=cats,
                index=default_idx,
                key=f"cat_{feat}",
            )

    # Feature numeriche
    with col_right:
        for feat in numeric_features:
            stats = stats_numeric.get(feat, {})
            default_val = stats.get("median", 0.0)
            min_val = stats.get("min", None)
            max_val = stats.get("max", None)

            number_kwargs = {
                "label": feat,
                "value": float(default_val) if default_val is not None else 0.0,
                "key": f"num_{feat}",
            }
            if min_val is not None:
                number_kwargs["min_value"] = float(min_val)
            if max_val is not None:
                number_kwargs["max_value"] = float(max_val)

            input_values[feat] = st.number_input(**number_kwargs)

    st.markdown("---")

    if st.button("Calcola probabilità di sottoscrizione"):
        pred = make_prediction(input_values)

        st.markdown("### Risultato")

        if "proba_positive" in pred:
            proba = pred["proba_positive"]
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">
                        Probabilità che il cliente sottoscriva il deposito
                    </div>
                    <div class="prediction-value">
                        {proba:.1%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.write("Predizione:", pred.get("prediction"))

    # Dati di riferimento (opzionale)
    with st.expander("Dati di riferimento utilizzati per i menu"):
        df_ref = load_reference_data()
        if df_ref is not None:
            st.dataframe(df_ref.head(), use_container_width=True)
        else:
            st.write(
                "Nessun dataset locale trovato. "
                "Controlla il percorso in `config.BANK_DATA_PATH`."
            )


if __name__ == "__main__":
    main()
