from pathlib import Path

import pandas as pd
import streamlit as st

from config import MLFLOW_MODEL_NAME
from model_utils import infer_feature_config, make_prediction


def load_css():
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Bank Marketing – Propensione al deposito",
        page_icon="🏦",
        layout="centered",
    )

    load_css()

    # Header
    st.markdown(
        """
        <div class="page-header">
            <h1>🏦 Bank Marketing Prediction</h1>
            <p>Prevedi la probabilità di sottoscrizione di un deposito a termine</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar con info modello
    with st.sidebar:
        st.markdown("### ℹ️ Informazioni")
        st.info(f"**Modello**: {MLFLOW_MODEL_NAME}")
        
        with st.expander("📋 Come funziona"):
            st.markdown("""
            1. **Inserisci** i dati del cliente
            2. **Valida** le informazioni
            3. **Calcola** la probabilità
            4. **Visualizza** il risultato
            """)

    # Carichiamo descrizione delle feature
    config = infer_feature_config()
    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]
    categories_by_feature = config["categories_by_feature"]
    stats_numeric = config["stats_numeric"]

    # Step 1: Input dei dati
    st.markdown(
        """
        <div class="step-header">
            <div class="step-number">1</div>
            <div class="step-title">Inserisci i dati del cliente</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_values = {}

    # Feature categoriche
    if categorical_features:
        st.markdown("#### Informazioni categoriche")
        cols = st.columns(2)
        for idx, feat in enumerate(categorical_features):
            with cols[idx % 2]:
                cats = categories_by_feature.get(feat, [])
                default_idx = 0 if cats else None
                input_values[feat] = st.selectbox(
                    feat.replace("_", " ").title(),
                    options=cats,
                    index=default_idx,
                    key=f"cat_{feat}",
                )

    # Feature numeriche
    if numeric_features:
        st.markdown("#### Informazioni numeriche")
        cols = st.columns(2)
        for idx, feat in enumerate(numeric_features):
            with cols[idx % 2]:
                stats = stats_numeric.get(feat, {})
                default_val = stats.get("median", 0.0)
                min_val = stats.get("min", None)
                max_val = stats.get("max", None)

                number_kwargs = {
                    "label": feat.replace("_", " ").title(),
                    "value": float(default_val) if default_val is not None else 0.0,
                    "key": f"num_{feat}",
                }
                if min_val is not None:
                    number_kwargs["min_value"] = float(min_val)
                if max_val is not None:
                    number_kwargs["max_value"] = float(max_val)

                input_values[feat] = st.number_input(**number_kwargs)

    # Step 2: Validazione (visual separator)
    st.markdown(
        """
        <div class="step-header">
            <div class="step-number">2</div>
            <div class="step-title">Calcola la predizione</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🎯 Calcola probabilità",
            type="primary",
            use_container_width=True,
        )

    # Step 3: Risultato
    if predict_button:
        with st.spinner("Elaborazione in corso..."):
            pred = make_prediction(input_values)

        st.markdown(
            """
            <div class="step-header">
                <div class="step-number">3</div>
                <div class="step-title">Risultato della predizione</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "proba_positive" in pred:
            proba = pred["proba_positive"]
            
            # Determina il livello di probabilità
            if proba >= 0.7:
                level = "alta"
                icon = "🟢"
            elif proba >= 0.4:
                level = "media"
                icon = "🟡"
            else:
                level = "bassa"
                icon = "🔴"
            
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">
                        Probabilità di sottoscrizione
                    </div>
                    <div class="prediction-value">
                        {proba:.1%}
                    </div>
                    <div class="prediction-level">
                        {icon} Probabilità {level}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # Visualizzazione aggiuntiva con barra di progresso
            st.markdown("##### Indicatore visivo")
            st.progress(proba)
        else:
            st.write("Predizione:", pred.get("prediction"))


if __name__ == "__main__":
    main()
