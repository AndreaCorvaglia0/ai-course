"""
🍷 Wine Quality Assessment - Sistema di Valutazione per Cantina
Webapp per la valutazione della qualità del vino e decisioni di affinamento

L'aspetto (colori, font, bordi) è definito in .streamlit/config.toml.
"""

import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import config
from model_utils import (
    get_feature_importance,
    get_quality_level,
    load_model_from_mlflow,
    predict_wine_quality,
)

# ==================== CONFIGURAZIONE PAGINA ====================
st.set_page_config(
    page_title="Wine Quality Assessment",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="auto",
)


# ==================== CARICAMENTO MODELLO ====================
@st.cache_resource
def load_model():
    """Carica il modello MLflow (cached)"""
    return load_model_from_mlflow()


try:
    model, run_info, model_features = load_model()
except Exception as e:
    st.title("Valutazione qualità del vino", icon=":material/wine_bar:")
    st.error("Nessun modello disponibile.", icon=":material/error:")
    st.markdown(
        "Esegui prima `development.ipynb`: allena il modello, registralo come "
        f"`{config.MLFLOW_MODEL_NAME}` con alias `@{config.MLFLOW_MODEL_ALIAS}` "
        "ed esporta la cartella `model/` con l'ultima cella del notebook."
    )
    with st.expander("Dettagli dell'errore", icon=":material/bug_report:"):
        st.code(str(e), language=None)
    st.stop()


# ==================== FUNZIONI DI SUPPORTO ====================
def feature_label(feature: str) -> str:
    """Nome italiano della feature con unità di misura."""
    label = config.FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())
    unit = config.FEATURE_UNITS.get(feature, "")
    return f"{label} ({unit})" if unit else label


def format_value(feature: str, value: float) -> str:
    """Valore formattato con gli stessi decimali dello slider."""
    _, fmt = config.FEATURE_STEPS.get(feature, (None, "%.2f"))
    return fmt % value


def snap(feature: str, value: float) -> float:
    """Arrotonda al passo dello slider e resta dentro il range."""
    low, high = config.FEATURE_RANGES[feature]
    step, _ = config.FEATURE_STEPS.get(feature, ((high - low) / 100, None))
    decimals = max(0, -math.floor(math.log10(step)))
    value = min(max(value, low), high)
    return round(round(value / step) * step, decimals)


def set_lot(values: dict) -> None:
    """Copia i valori di un lotto negli slider (callback dei bottoni)."""
    for feature, value in values.items():
        if feature in model_features and feature in config.FEATURE_RANGES:
            st.session_state[f"input_{feature}"] = snap(feature, float(value))


def random_lot() -> None:
    """Lotto casuale con valori dentro i range tipici del dataset (5°-95° percentile)."""
    rng = np.random.default_rng()
    values = {}
    for feature in model_features:
        if feature in config.FEATURE_TYPICAL:
            p05, p95, _ = config.FEATURE_TYPICAL[feature]
            values[feature] = rng.uniform(p05, p95)
    set_lot(values)


def feature_input(feature: str) -> float:
    """Slider della feature con la nota sul suo contenuto sotto."""
    key = f"input_{feature}"
    if feature in config.FEATURE_RANGES:
        low, high = config.FEATURE_RANGES[feature]
        step, fmt = config.FEATURE_STEPS.get(feature, ((high - low) / 100, "%.2f"))
        value = st.slider(
            feature_label(feature),
            min_value=float(low),
            max_value=float(high),
            step=float(step),
            format=fmt,
            key=key,
        )
    else:
        # Feature non prevista nel config: input numerico libero
        value = st.number_input(feature_label(feature), key=key)

    notes = f"`{feature}` · {config.FEATURE_DESCRIPTIONS.get(feature, '')}"
    if feature in config.FEATURE_TYPICAL:
        p05, p95, high_quality = config.FEATURE_TYPICAL[feature]
        unit = config.FEATURE_UNITS.get(feature, "")
        notes += (
            f"  \n:material/straighten: Tipico {format_value(feature, p05)}–{format_value(feature, p95)} {unit}"
            f" · vini di alta qualità: {format_value(feature, high_quality)} {unit}"
        )
    st.caption(notes)
    return value


def profile_chart(features: dict) -> alt.Chart:
    """Posizione del lotto rispetto ai valori tipici del dataset, feature per feature."""
    rows = []
    for feature, value in features.items():
        if feature not in config.FEATURE_TYPICAL:
            continue
        p05, p95, high_quality = config.FEATURE_TYPICAL[feature]
        unit = config.FEATURE_UNITS.get(feature, "")
        rows.append({
            "feature": config.FEATURE_LABELS.get(feature, feature),
            "lotto": float(np.clip((value - p05) / (p95 - p05), -0.3, 1.3)),
            "alta_qualita": (high_quality - p05) / (p95 - p05),
            "Il tuo lotto": f"{format_value(feature, value)} {unit}",
            "Tipico": f"{format_value(feature, p05)}–{format_value(feature, p95)} {unit}",
            "Vini di alta qualità": f"{format_value(feature, high_quality)} {unit}",
            "inizio": 0.0,
            "fine": 1.0,
        })
    data = pd.DataFrame(rows)
    order = list(data["feature"])
    tooltip = ["feature:N", "Il tuo lotto:N", "Tipico:N", "Vini di alta qualità:N"]
    x_axis = alt.Axis(
        values=[0, 1],
        labelExpr="datum.value == 0 ? '5° percentile' : '95° percentile'",
        title=None,
        grid=False,
    )
    y = alt.Y("feature:N", sort=order, title=None)

    band = alt.Chart(data).mark_bar(size=12, color="#E8D9C4", cornerRadius=6).encode(
        x=alt.X("inizio:Q", scale=alt.Scale(domain=[-0.3, 1.3]), axis=x_axis),
        x2="fine:Q",
        y=y,
        tooltip=tooltip,
    )
    high_quality_tick = alt.Chart(data).mark_tick(color="#C9A227", thickness=3, size=20).encode(
        x="alta_qualita:Q", y=y, tooltip=tooltip
    )
    lot_point = alt.Chart(data).mark_circle(color="#7B1E3A", size=160, opacity=1).encode(
        x="lotto:Q", y=y, tooltip=tooltip
    )
    return (band + high_quality_tick + lot_point).properties(height=34 * len(data))


def importance_chart(importance: pd.Series) -> alt.Chart:
    """Importanza delle feature per il modello."""
    data = pd.DataFrame({
        "feature": [config.FEATURE_LABELS.get(f, f) for f in importance.index],
        "importanza": importance.values,
    })
    return alt.Chart(data).mark_bar(color="#7B1E3A", cornerRadiusEnd=6).encode(
        x=alt.X("importanza:Q", axis=alt.Axis(format="%", title=None)),
        y=alt.Y("feature:N", sort="-x", title=None),
        tooltip=["feature:N", alt.Tooltip("importanza:Q", format=".1%")],
    ).properties(height=30 * len(data))


# ==================== STATO INIZIALE DEGLI SLIDER ====================
for feature in model_features:
    if feature in config.FEATURE_RANGES:
        low, high = config.FEATURE_RANGES[feature]
        default = config.FEATURE_DEFAULTS.get(feature, (low + high) / 2)
        st.session_state.setdefault(f"input_{feature}", snap(feature, float(default)))
    else:
        st.session_state.setdefault(f"input_{feature}", 0.0)


# ==================== SIDEBAR - INFO MODELLO ====================
with st.sidebar:
    st.header("Il modello", icon=":material/hub:")
    st.markdown(f"**{run_info['model_name']}** `@{run_info['model_alias']}`")
    st.caption(f"Versione {run_info['model_version']} · fonte: {run_info['source']}")
    if run_info["run_id"] not in ("N/A", "local"):
        st.caption(f"Run `{run_info['run_name']}` · `{run_info['run_id'][:8]}`")

    st.subheader("Performance sul test set")
    metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 score", "f1_score"),
        ("ROC AUC", "roc_auc"),
    ]
    if any(run_info[key] for _, key in metrics):
        metric_cols = st.columns(2)
        for i, (label, key) in enumerate(metrics):
            metric_cols[i % 2].metric(label, f"{run_info[key]:.3f}")
    else:
        st.caption(
            "Metriche non disponibili: il modello è stato caricato da file locale, "
            "oppure le metriche non sono state loggate con i nomi "
            "`accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`."
        )

    st.subheader("Il modello usa")
    st.caption(" · ".join(config.FEATURE_LABELS.get(f, f) for f in model_features))


# ==================== HEADER ====================
st.title("Valutazione qualità del vino", icon=":material/wine_bar:", text_alignment="center")
st.markdown(
    "Dalle analisi di laboratorio di un lotto di **Vinho Verde**, stima se diventerà un vino "
    "di **alta qualità** e se merita l'investimento in **barrique**.",
    text_alignment="center",
)
with st.container(horizontal=True, horizontal_alignment="center"):
    st.badge("MLflow Model Registry", icon=":material/hub:", color="red")
    st.badge("Predizione in tempo reale", icon=":material/bolt:", color="orange")
    st.badge("Progetto capstone", icon=":material/school:", color="gray")
st.caption(
    f"Webapp di esempio del corso **{config.COURSE_NAME}** · "
    f"[Vedi il corso su GitHub]({config.COURSE_REPO_URL})",
    text_alignment="center",
)

# Quando / su cosa / cosa ottieni
usage_cards = [
    (":material/schedule:", "Quando usarla",
     "Dopo le **analisi di laboratorio** del lotto, **prima** di decidere se destinarlo alle barrique."),
    (":material/wine_bar:", "Su quale lotto",
     "Un lotto di **Vinho Verde**, rosso o bianco, di cui hai le analisi chimico-fisiche."),
    (":material/insights:", "Cosa ottieni",
     "La **probabilità** che il lotto sia di alta qualità (voto ≥ 7) e l'**azione di cantina** consigliata."),
]
for col, (icon, title, text) in zip(st.columns(3), usage_cards):
    with col.container(border=True, height="stretch"):
        st.markdown(f"**{icon} {title}**")
        st.markdown(text)
st.caption(
    "Il colore del vino e l'assaggio non sono input: il modello usa solo le analisi chimico-fisiche.",
    text_alignment="center",
)

st.space("small")
col_input, col_result = st.columns([3, 2], gap="large")

# ==================== COLONNA INPUT (SINISTRA) ====================
with col_input:
    st.header("1 · Descrivi il lotto", icon=":material/science:")
    st.caption("Imposta i valori delle analisi di laboratorio, oppure parti da un lotto di esempio.")

    with st.container(horizontal=True):
        for name, lot in config.EXAMPLE_LOTS.items():
            st.button(name, icon=":material/wine_bar:", help=lot["summary"], on_click=set_lot, args=(lot["values"],))
        st.button(
            "Lotto casuale",
            icon=":material/casino:",
            help="Valori casuali dentro i range tipici del dataset",
            on_click=random_lot,
        )

    # Solo le feature usate dal modello; quelle non previste finiscono in "Altri parametri"
    groups = [
        (title, icon, subtitle, [f for f in feats if f in model_features])
        for title, icon, subtitle, feats in config.FEATURE_GROUPS
    ]
    grouped = {f for *_, feats in groups for f in feats}
    other = [f for f in model_features if f not in grouped]
    if other:
        groups.append(("Altri parametri", ":material/tune:", "Altre variabili usate dal modello", other))

    values = {}
    for title, icon, subtitle, feats in groups:
        if not feats:
            continue
        with st.container(border=True):
            st.subheader(title, icon=icon)
            st.caption(subtitle)
            input_cols = st.columns(2, gap="medium")
            for i, feature in enumerate(feats):
                with input_cols[i % 2]:
                    values[feature] = feature_input(feature)

    # Stesso ordine delle feature usato in training
    features = {feature: values[feature] for feature in model_features}

# ==================== COLONNA RISULTATO (DESTRA) ====================
with col_result:
    st.header("2 · Valutazione", icon=":material/insights:")
    st.caption("Si aggiorna in tempo reale a ogni modifica dei parametri.")

    prediction, probability = predict_wine_quality(model, features)
    level, color, level_icon, action, action_detail = config.QUALITY_LEVELS[get_quality_level(probability)]
    thresholds = config.QUALITY_THRESHOLDS
    base_rate = config.BASE_HIGH_QUALITY_RATE

    with st.container(border=True):
        st.badge(level, icon=level_icon, color=color)
        st.metric(
            "Probabilità di alta qualità",
            f"{probability:.0%}",
            delta=f"{(probability - base_rate) * 100:+.0f} punti",
            delta_description=f"rispetto alla media storica ({base_rate:.0%})",
        )
        st.progress(probability)
        st.caption(
            f"Base sotto il {thresholds['medium']:.0%} · Medio dal {thresholds['medium']:.0%} · "
            f"Buono dal {thresholds['good']:.0%} · Eccellente dal {thresholds['excellent']:.0%}"
        )
        st.markdown(f"**Azione consigliata: {action}**")
        st.markdown(action_detail)
        st.caption(
            "Classe prevista dal modello (soglia 50%): "
            f"**{'alta qualità' if prediction == 1 else 'standard'}**"
        )

    with st.expander("Come leggere lo score", icon=":material/help:"):
        st.markdown(
            f"Nello storico solo il **{base_rate:.0%}** dei vini ha ricevuto un voto di 7 o più. "
            "Per questo le soglie della raccomandazione sono più basse del 50%: anche un lotto "
            "al 35% è molto più promettente della media. Usa lo score per **confrontare e ordinare** "
            "i lotti, non come certezza: l'ultima parola resta all'assaggio dell'enologo. "
            "Le soglie si cambiano in `config.py` (`QUALITY_THRESHOLDS`)."
        )

    tab_profile, tab_model, tab_details = st.tabs(
        [":material/tune: Profilo", ":material/bar_chart: Il modello", ":material/code: Dettagli"]
    )
    with tab_profile:
        st.altair_chart(profile_chart(features), width="stretch")
        st.caption(
            ":red[**●**] il tuo lotto · :orange[**|**] mediana dei vini di alta qualità · "
            "barra: valori tipici del dataset (dal 5° al 95° percentile). "
            "Passa sopra ai punti per vedere i valori."
        )
    with tab_model:
        importance = get_feature_importance(model)
        if importance is None:
            st.caption("Questo modello non espone l'importanza delle feature.")
        else:
            st.altair_chart(importance_chart(importance), width="stretch")
            st.caption(
                "Quanto ogni parametro pesa nelle decisioni del modello, su tutti i lotti "
                "(`feature_importances_` o coefficienti). Non dice in che direzione: "
                "per quello guarda il profilo del lotto."
            )
    with tab_details:
        st.json({
            "input": {f: float(v) for f, v in features.items()},
            "probabilita_alta_qualita": round(probability, 4),
            "classe_prevista": prediction,
            "modello": f"models:/{run_info['model_name']}@{run_info['model_alias']}",
            "fonte": run_info["source"],
        })

# ==================== FOOTER ====================
st.space("medium")
st.caption(
    f"Webapp di esempio del corso {config.COURSE_NAME} · [GitHub]({config.COURSE_REPO_URL})  \n"
    "Dataset: UCI Machine Learning Repository, Wine Quality · Powered by MLflow, Streamlit e scikit-learn",
    text_alignment="center",
)
