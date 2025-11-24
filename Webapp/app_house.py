import pandas as pd
import streamlit as st
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml

# --------------------------------------------------
# Configurazione MLflow e dataset
# --------------------------------------------------
TRACKING_URI = "file:../mlruns"
MODEL_NAME = "house_sales_rf_prod"
MODEL_ALIAS = "production"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

OPENML_DATA_ID = 42092

st.set_page_config(
    page_title="House Sales – Predizione del prezzo di vendita",
    layout="centered",
)

# --------------------------------------------------
# Caricamento del modello da MLflow Model Registry
# --------------------------------------------------
@st.cache_resource
def load_model():
    """Carica il modello registrato in MLflow usando alias."""
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.sklearn.load_model(MODEL_URI)

# --------------------------------------------------
# Caricamento statistiche del dataset
# --------------------------------------------------
@st.cache_data
def load_feature_stats():
    """Scarica dataset House Sales e calcola statistiche numeriche."""
    raw = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True)
    X = raw.data.select_dtypes(include=["number"])
    stats = X.describe().T
    return stats

# --------------------------------------------------
# UI HEADER
# --------------------------------------------------
st.title("House Sales – Predizione del prezzo di vendita")
st.write(
    "Interfaccia semplice per interrogare il modello di regressione "
    "registrato in MLflow (alias: 'production')."
)

# --------------------------------------------------
# Carica il modello
# --------------------------------------------------
try:
    model = load_model()
except Exception as e:
    st.error(
        "❌ Impossibile caricare il modello da MLflow.\n\n"
        "Verifica che:\n"
        f"- il modello '{MODEL_NAME}' esista nel registry\n"
        f"- abbia l'alias '{MODEL_ALIAS}'\n"
        "- la cartella '../mlruns' sia quella giusta."
    )
    st.exception(e)
    st.stop()

# Recupera le feature dal modello
feature_names = list(getattr(model, "feature_names_in_", []))
if not feature_names:
    st.error(
        "❌ Il modello non contiene l'attributo 'feature_names_in_'.\n"
        "Controlla che sia stato addestrato su un DataFrame con nomi colonna."
    )
    st.stop()

# --------------------------------------------------
# Statistiche per definire slider sensati
# --------------------------------------------------
try:
    stats = load_feature_stats()
except Exception:
    stats = None
    st.warning(
        "⚠️ Non è stato possibile caricare il dataset da OpenML.\n"
        "Slider impostati con range generici."
    )

st.sidebar.header("Input caratteristiche abitazione")

# --------------------------------------------------
# GENERAZIONE SLIDER
# --------------------------------------------------
input_data = {}

for feature in feature_names:
    if stats is not None and feature in stats.index:
        col_stats = stats.loc[feature]
        min_val = float(col_stats["min"])
        max_val = float(col_stats["max"])
        default_val = float(col_stats["50%"])

        if min_val == max_val:
            min_val = 0.0
            max_val = default_val * 2 if default_val else 1.0
    else:
        min_val = 0.0
        max_val = 1_000_000.0
        default_val = (min_val + max_val) / 2

    is_int = (
        float(int(min_val)) == min_val
        and float(int(max_val)) == max_val
    )

    if is_int:
        value = st.sidebar.slider(
            feature,
            min_value=int(min_val),
            max_value=int(max_val),
            value=int(default_val),
        )
    else:
        step = (max_val - min_val) / 100 if max_val > min_val else 0.1
        if step == 0:
            step = 0.1
        value = st.sidebar.slider(
            feature,
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=step,
        )

    input_data[feature] = value

# DataFrame di input
input_df = pd.DataFrame([input_data])

# --------------------------------------------------
# PREVISIONE
# --------------------------------------------------
if st.button("Stima il prezzo di vendita"):
    with st.spinner("Calcolo in corso..."):
        try:
            pred = model.predict(input_df)
            price = float(pred[0])
        except Exception as e:
            st.error("❌ Errore durante la predizione.")
            st.exception(e)
        else:
            st.subheader("🏡 Prezzo stimato")
            st.metric("Valore previsto", f"{price:,.0f}")
            st.caption("Il valore è espresso nella stessa unità del target del dataset House Sales.")
