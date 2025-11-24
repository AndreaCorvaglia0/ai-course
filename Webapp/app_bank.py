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
)


# Feature descriptions for tooltips - Original features (before encoding)
FEATURE_INFO = {
    # Anagrafica Cliente
    "age": "Età del cliente in anni",
    "job": "Tipo di occupazione/lavoro",
    "marital": "Stato civile (married, single, divorced)",
    "education": "Livello di istruzione (primary, secondary, tertiary)",
    
    # Situazione Finanziaria
    "default": "Ha crediti in default? (yes/no)",
    "balance": "Saldo medio annuo del conto corrente in euro",
    "housing": "Ha un mutuo per la casa? (yes/no)",
    "loan": "Ha un prestito personale? (yes/no)",
    
    # Informazioni Contatto
    "contact": "Tipo di comunicazione utilizzata (cellular, telephone, unknown)",
    "day_of_week": "Giorno del mese dell'ultimo contatto",
    "month": "Mese dell'ultimo contatto",
    
    # Campagna Marketing
    "campaign": "Numero di contatti effettuati durante questa campagna",
    "pdays": "Giorni dall'ultimo contatto della campagna precedente (-1 = mai contattato)",
    "previous": "Numero di contatti prima di questa campagna",
    "poutcome": "Risultato della campagna precedente (success, failure, unknown)",
}


# Slider configuration for numeric features
SLIDER_CONFIG = {
    "age": {"min": 18, "max": 95, "step": 1},
    "balance": {"min": -10000, "max": 100000, "step": 100},
    "campaign": {"min": 1, "max": 50, "step": 1},
    "pdays": {"min": -1, "max": 900, "step": 1},
    "previous": {"min": 0, "max": 50, "step": 1},
    "day_of_week": {"min": 1, "max": 31, "step": 1},
}



def load_css():
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def get_probability_level(proba: float) -> tuple[str, str]:
    """Return risk level label and color class based on probability."""
    if proba >= 0.7:
        return "High Propensity", "high"
    elif proba >= 0.4:
        return "Medium Propensity", "medium"
    else:
        return "Low Propensity", "low"


def get_feature_description(feature_name: str) -> str:
    """Get description for a feature from the info dictionary."""
    if feature_name in FEATURE_INFO:
        return FEATURE_INFO[feature_name]
    return f"Feature: {feature_name.replace('_', ' ').title()}"


def main():
    st.set_page_config(
        page_title="Bank Marketing AI Predictor",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    load_css()

    # Hero Header
    st.markdown(
        """
        <div class="hero-header">
            <div class="hero-icon">🎯</div>
            <h1>Bank Marketing AI Predictor</h1>
            <p class="hero-subtitle">Advanced ML-powered customer propensity analysis</p>
            <div class="hero-badges">
                <span class="badge">MLflow Pipeline</span>
                <span class="badge">Real-time Prediction</span>
                <span class="badge">Production Ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model info in compact header
    col_info1, col_info2, col_info3 = st.columns([1, 2, 1])
    with col_info2:
        st.markdown(
            f"""
            <div class="model-info-bar">
                <div class="info-item">
                    <span class="info-label">Model</span>
                    <span class="info-value">{MLFLOW_MODEL_NAME}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Status</span>
                    <span class="info-value"><span class="status-dot"></span>Active</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Load feature configuration
    config = infer_feature_config()
    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]
    categories_by_feature = config["categories_by_feature"]
    stats_numeric = config["stats_numeric"]

    # Main content container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Step 1: Input Section
    st.markdown(
        """
        <div class="section-header">
            <div class="step-number">1</div>
            <div class="step-content">
                <h2>Customer Profile Input</h2>
                <p>Enter the customer characteristics for prediction analysis</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_values = {}

    # Create responsive 3-column layout
    col1, col2, col3 = st.columns([1, 1, 1], gap="large")

    # Distribute features across columns
    all_features = categorical_features + numeric_features
    features_per_col = len(all_features) // 3 + 1

    def render_feature_input(feat):
        """Render appropriate input widget for a feature."""
        if feat in categorical_features:
            cats = categories_by_feature.get(feat, [])
            return st.selectbox(
                f"{feat.replace('_', ' ').title()}",
                options=cats,
                key=f"cat_{feat}",
                help=get_feature_description(feat),
            )
        else:
            stats = stats_numeric.get(feat, {})
            default_val = stats.get("median", 0.0)
            min_val = stats.get("min", None)
            max_val = stats.get("max", None)
            
            # Use slider for specific features
            if feat in SLIDER_CONFIG:
                slider_conf = SLIDER_CONFIG[feat]
                # Use stats if available, otherwise use config
                actual_min = int(min_val) if min_val is not None else slider_conf["min"]
                actual_max = int(max_val) if max_val is not None else slider_conf["max"]
                actual_default = int(default_val) if default_val is not None else (actual_min + actual_max) // 2
                
                return st.slider(
                    f"{feat.replace('_', ' ').title()}",
                    min_value=actual_min,
                    max_value=actual_max,
                    value=actual_default,
                    step=slider_conf["step"],
                    key=f"num_{feat}",
                    help=get_feature_description(feat),
                )
            else:
                # Use number input for other numeric features
                return st.number_input(
                    f"{feat.replace('_', ' ').title()}",
                    value=float(default_val) if default_val is not None else 0.0,
                    min_value=float(min_val) if min_val is not None else None,
                    max_value=float(max_val) if max_val is not None else None,
                    step=1.0,
                    key=f"num_{feat}",
                    help=get_feature_description(feat),
                )

    with col1:
        for feat in all_features[:features_per_col]:
            input_values[feat] = render_feature_input(feat)

    with col2:
        for feat in all_features[features_per_col:features_per_col*2]:
            input_values[feat] = render_feature_input(feat)

    with col3:
        for feat in all_features[features_per_col*2:]:
            input_values[feat] = render_feature_input(feat)

    st.markdown("<br>", unsafe_allow_html=True)

    # Step 2: Prediction Button
    st.markdown(
        """
        <div class="section-header">
            <div class="step-number">2</div>
            <div class="step-content">
                <h2>Run Prediction</h2>
                <p>Execute the ML model to analyze subscription propensity</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Center the button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button(
            "🚀 Analyze Customer Propensity",
            type="primary",
            use_container_width=True,
        )

    # Step 3: Results
    if predict_button:
        with st.spinner("🔮 Analyzing customer data with ML model..."):
            try:
                pred = make_prediction(input_values)
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.stop()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-header">
                <div class="step-number">3</div>
                <div class="step-content">
                    <h2>Prediction Results</h2>
                    <p>AI-generated insights and probability analysis</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "proba_positive" in pred:
            proba = pred["proba_positive"]
            level_label, level_class = get_probability_level(proba)
            
            # Debug info
            with st.expander("🔍 Debug Info (per sviluppatori)", expanded=False):
                st.write("**Raw Prediction Output:**")
                st.json(pred)
                st.write("**Input Values:**")
                st.json(input_values)
            
            # Dynamic color based on probability
            st.markdown(
                f"""
                <div class="prediction-card {level_class}">
                    <div class="prediction-header">
                        <div class="prediction-icon">{'🎯' if proba >= 0.7 else '📈' if proba >= 0.4 else '📉'}</div>
                        <div class="prediction-level">{level_label}</div>
                    </div>
                    <div class="prediction-main">
                        <div class="prediction-value">{proba:.1%}</div>
                        <div class="prediction-label">Subscription Probability</div>
                    </div>
                    <div class="probability-bar">
                        <div class="probability-fill" style="width: {proba*100}%"></div>
                    </div>
                    <div class="prediction-footer">
                        <div class="footer-item">
                            <span class="footer-label">Confidence</span>
                            <span class="footer-value">{'High' if abs(proba - 0.5) > 0.3 else 'Moderate'}</span>
                        </div>
                        <div class="footer-item">
                            <span class="footer-label">Recommendation</span>
                            <span class="footer-value">{'Priority Contact' if proba >= 0.7 else 'Standard Process' if proba >= 0.4 else 'Low Priority'}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Fallback for regression
            st.success(f"Prediction: {pred.get('prediction', 'N/A')}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown(
        """
        <div class="footer">
            <p>Powered by MLflow • Streamlit • scikit-learn</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
