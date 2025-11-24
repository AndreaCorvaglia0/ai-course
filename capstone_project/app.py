"""
🍷 Wine Quality Assessment - Sistema di Valutazione per Cantina
Webapp per la valutazione della qualità del vino e decisioni di affinamento
"""

import streamlit as st
import pandas as pd
import numpy as np
from model_utils import (
    load_model_from_mlflow,
    predict_wine_quality,
    get_quality_recommendation
)
import config

# ==================== CONFIGURAZIONE PAGINA ====================
st.set_page_config(
    page_title="Wine Quality Assessment",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Font e tema generale */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Lato:wght@300;400&display=swap');
    
    .main {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d1810 100%);
    }
    
    /* Header principale */
    .wine-header {
        text-align: center;
        padding: 2rem 0;
        font-family: 'Playfair Display', serif;
        color: #f4e8d8;
        border-bottom: 2px solid #8B0000;
        margin-bottom: 2rem;
    }
    
    .wine-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #CD5C5C;
    }
    
    .wine-header p {
        font-size: 1.1rem;
        font-family: 'Lato', sans-serif;
        color: #d4c5b0;
        font-weight: 300;
    }
    
    /* Card risultato */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        border: 2px solid;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.5);
    }
    
    /* Probability display */
    .prob-display {
        text-align: center;
        font-family: 'Playfair Display', serif;
        margin: 1.5rem 0;
    }
    
    .prob-value {
        font-size: 4.5rem;
        font-weight: 700;
        line-height: 1;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    .prob-label {
        font-size: 1.2rem;
        font-family: 'Lato', sans-serif;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Recommendation box */
    .recommendation {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        font-family: 'Lato', sans-serif;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    /* Model info sidebar */
    .model-info {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Lato', sans-serif;
        font-size: 0.9rem;
    }
    
    .metric-box {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #8B0000, #CD5C5C);
    }
    
    /* Feature label con info icon */
    .feature-label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.3rem;
    }
    
    .info-icon {
        cursor: help;
        font-size: 1rem;
        color: #CD5C5C;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CARICAMENTO MODELLO ====================
@st.cache_resource
def load_model():
    """Carica il modello MLflow (cached)"""
    try:
        model, run_info, feature_names = load_model_from_mlflow()
        return model, run_info, feature_names
    except Exception as e:
        st.error(f"❌ Errore nel caricamento del modello: {str(e)}")
        st.stop()

model, run_info, model_features = load_model()

# ==================== HEADER ====================
st.markdown("""
<div class="wine-header">
    <h1>Valutazione Qualità Vino</h1>
    <p>Sistema di supporto decisionale per l'affinamento in cantina</p>
</div>
""", unsafe_allow_html=True)

# ==================== BOTTONI IN ALTO ====================
col_btn1, col_btn2 = st.columns([3, 1], gap="small")
with col_btn1:
    analyze_button = st.button("🔬 Analizza Lotto", type="primary", key="analyze")
with col_btn2:
    random_button = st.button("🎲 Random", key="random")

# Gestione stato random
if random_button:
    st.session_state.random_trigger = not st.session_state.get('random_trigger', False)

st.markdown("<br>", unsafe_allow_html=True)

# ==================== LAYOUT ====================
col_input, col_result = st.columns([1, 1], gap="large")

# ==================== COLONNA INPUT (SINISTRA) ====================
with col_input:
    st.markdown("### Parametri Chimico-Fisici")
    
    # Crea dizionario per raccogliere i valori
    feature_values = {}
    
    # Crea sliders per ogni feature richiesta dal modello
    for feature in model_features:
        if feature in config.FEATURE_RANGES:
            min_val, max_val = config.FEATURE_RANGES[feature]
            default_val = config.FEATURE_DEFAULTS.get(feature, (min_val + max_val) / 2)
            
            # Se random è stato cliccato, genera valore random
            if st.session_state.get('random_trigger', False):
                # Genera valore random nell'intervallo
                default_val = np.random.uniform(min_val, max_val)
            
            unit = config.FEATURE_UNITS.get(feature, "")
            description = config.FEATURE_DESCRIPTIONS.get(feature, "")
            
            # Nome display pulito
            display_name = feature.replace("_", " ").title()
            
            # Label con icona info e tooltip
            label_text = f"{display_name} ({unit})" if unit else display_name
            
            # Slider con help integrato (hovering)
            value = st.slider(
                label_text,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                step=(max_val - min_val) / 100,
                key=f"slider_{feature}_{st.session_state.get('random_trigger', False)}",  # Key dinamica per re-render
                help=f"ℹ️ {description}",
                on_change=lambda: st.session_state.update({'auto_predict': True})  # Trigger auto-predict
            )
            
            feature_values[feature] = value
    
    # Reset random trigger dopo il render
    if st.session_state.get('random_trigger'):
        st.session_state.random_trigger = False

# ==================== COLONNA RISULTATO (DESTRA) ====================
with col_result:
    st.markdown("### Valutazione e Raccomandazione")
    
    # Predizione automatica dopo il primo run o cambio parametri
    should_predict = analyze_button or st.session_state.get('auto_predict', False) or st.session_state.get('has_predicted', False)
    
    if should_predict:
        # Segna che abbiamo fatto almeno una predizione
        st.session_state.has_predicted = True
        
        with st.spinner("Analisi in corso..."):
            # Predizione
            prediction, probability = predict_wine_quality(model, feature_values)
            level, recommendation, color = get_quality_recommendation(probability)
            
            # CAMBIO COLORE BACKGROUND DELL'INTERA PAGINA
            st.markdown(f"""
            <style>
                .main {{
                    background: linear-gradient(135deg, {color}15 0%, {color}05 100%) !important;
                }}
            </style>
            """, unsafe_allow_html=True)
            
            # RISULTATO PRINCIPALE - Grande e chiaro
            st.markdown(f"""
            <div style="
                background: {color};
                border-radius: 20px;
                padding: 3rem 2rem;
                text-align: center;
                margin: 2rem 0;
                box-shadow: 0 10px 40px {color}80;
            ">
                <div style="font-size: 5rem; font-weight: 700; color: white; line-height: 1;">
                    {probability:.0%}
                </div>
                <div style="font-size: 1.3rem; color: white; margin-top: 1rem; opacity: 0.95;">
                    Probabilità Alta Qualità
                </div>
                <div style="
                    background: rgba(255,255,255,0.2);
                    border-radius: 10px;
                    padding: 0.8rem;
                    margin-top: 2rem;
                    font-size: 1.8rem;
                    font-weight: 600;
                    color: white;
                ">
                    {level}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # RACCOMANDAZIONE - Box separato
            st.markdown(f"""
            <div style="
                background: white;
                border-left: 6px solid {color};
                border-radius: 10px;
                padding: 2rem;
                margin: 2rem 0;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            ">
                <div style="font-size: 1.2rem; line-height: 1.6; color: #333;">
                    {recommendation}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar
            st.progress(probability)
    
    else:
        # Placeholder elegante quando non c'è predizione
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            border: 2px dashed #696969;
            padding: 4rem 2rem;
            text-align: center;
            margin: 2rem 0;
        ">
            <div style="font-size: 4rem; opacity: 0.3;">
                🔬
            </div>
            <p style="color: #d4c5b0; font-size: 1.1rem; margin-top: 1.5rem; line-height: 1.6;">
                Imposta i parametri e clicca<br>
                <strong>"Analizza Lotto"</strong><br>
                per ottenere la valutazione
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==================== SIDEBAR - INFO MODELLO ====================
with st.sidebar:
    st.markdown("### Informazioni Modello")
    
    st.markdown(f"""
    <div class="model-info">
        <strong>Modello:</strong><br>
        {run_info['model_name']} <span style="color: #CD5C5C;">@{run_info['model_alias']}</span><br><br>
        <strong>Versione:</strong><br>
        v{run_info['model_version']}<br><br>
        <strong>Fonte:</strong><br>
        {run_info['source']}<br><br>
        <strong>Run ID:</strong><br>
        <code>{run_info['run_id'][:8] if run_info['run_id'] not in ['N/A', 'local'] else run_info['run_id']}...</code>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### Performance Metriche")
    
    metrics_data = [
        ("Accuracy", run_info['accuracy']),
        ("Precision", run_info['precision']),
        ("Recall", run_info['recall']),
        ("F1 Score", run_info['f1_score']),
        ("ROC AUC", run_info['roc_auc'])
    ]
    
    for metric_name, metric_value in metrics_data:
        st.markdown(f"""
        <div class="metric-box">
            <strong>{metric_name}</strong><br>
            <span style="font-size: 1.2rem; color: #CD5C5C;">{metric_value:.3f}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Info sul dataset
    st.markdown("### Features Modello")
    st.markdown(f"""
    <div style="font-size: 0.9rem; color: #d4c5b0; line-height: 1.6;">
        Il modello analizza <strong>{len(model_features)}</strong> parametri chimico-fisici:<br><br>
        {'<br>'.join([f'• {feat.replace("_", " ").title()}' for feat in model_features[:5]])}
        <br>... e altri {len(model_features) - 5}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Legenda soglie
    st.markdown("### Soglie Qualità")
    st.markdown(f"""
    <div style="font-size: 0.9rem; line-height: 1.8;">
        <div style="color: {config.COLORS['excellent']};">
            ⬤ Eccellente: ≥ {config.QUALITY_THRESHOLDS['excellent']:.0%}
        </div>
        <div style="color: {config.COLORS['good']};">
            ⬤ Buono: ≥ {config.QUALITY_THRESHOLDS['good']:.0%}
        </div>
        <div style="color: {config.COLORS['medium']};">
            ⬤ Medio: ≥ {config.QUALITY_THRESHOLDS['medium']:.0%}
        </div>
        <div style="color: {config.COLORS['low']};">
            ⬤ Base: < {config.QUALITY_THRESHOLDS['medium']:.0%}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.8rem; color: #888; text-align: center; margin-top: 2rem;">
        🍇 Wine Quality Assessment v1.0<br>
        Powered by MLflow & Scikit-learn
    </div>
    """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem; padding: 1rem;">
    Dataset: UCI Machine Learning Repository - Wine Quality<br>
    Sistema decisionale basato su analisi chimico-fisica per la selezione dei lotti da affinare
</div>
""", unsafe_allow_html=True)
