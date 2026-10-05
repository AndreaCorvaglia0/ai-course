from pathlib import Path

import streamlit as st

from config import MLFLOW_MODEL_NAME
from model_utils import (
    infer_feature_config,
    make_prediction,
)


# Input labels and descriptions - Original features (before encoding)
# Written from the point of view of someone deciding whether to call the customer
FEATURE_INFO = {
    # Customer profile
    "age": ("Età", "Età del cliente in anni."),
    "job": ("Professione", "Tipo di lavoro."),
    "marital": ("Stato civile", "Sposato, single o divorziato (divorced include anche i vedovi)."),
    "education": ("Istruzione", "Livello di istruzione più alto."),

    # Finances
    "balance": ("Saldo medio annuo (€)", "Saldo medio annuo dei conti del cliente. Può essere negativo."),
    "default": ("Crediti in default?", "Il cliente ha crediti in default?"),
    "housing": ("Mutuo casa?", "Il cliente ha un mutuo per la casa?"),
    "loan": ("Prestito personale?", "Il cliente ha un prestito personale?"),

    # Planned call
    "contact": ("Canale di contatto", "Come chiamerai il cliente: cellulare (cellular) o telefono fisso (telephone)."),
    "month": ("Mese della chiamata", "Mese in cui prevedi di chiamare."),
    "day_of_week": ("Giorno della chiamata", "Giorno del mese in cui prevedi di chiamare."),
    "campaign": ("Chiamate in questa campagna", "Chiamate a questo cliente nella campagna attuale, inclusa quella prevista (1 = prima chiamata)."),

    # Past campaigns
    "previous": ("Contatti nelle campagne passate", "Quante volte il cliente è stato contattato prima di questa campagna (0 = mai)."),
    "pdays": ("Giorni dall'ultimo contatto passato", "Giorni dall'ultimo contatto in una campagna precedente (-1 = mai contattato)."),
    "poutcome": ("Esito della campagna passata", "Risultato della campagna precedente per questo cliente. Scegli 'sconosciuto' se non è mai stato contattato."),
}


# Input groups, shown as a 2x2 grid: (title, subtitle, features)
FEATURE_GROUPS = [
    ("Chi è il cliente", "Dall'anagrafica della banca", ["age", "job", "marital", "education"]),
    ("Situazione finanziaria", "Saldo e prodotti che il cliente ha già", ["balance", "default", "housing", "loan"]),
    ("La chiamata che stai pianificando", "Come, quando e quante volte chiamerai", ["contact", "month", "day_of_week", "campaign"]),
    ("Campagne passate", "Cosa è successo nelle campagne precedenti", ["previous", "pdays", "poutcome"]),
]


# How option values are shown in the selectboxes (the model still receives the raw value)
OPTION_LABELS = {
    "nan": "sconosciuto",
    "cellular": "cellular (cellulare)",
    "telephone": "telephone (fisso)",
}


# Share of called customers who subscribed in the course dataset (UCI Bank Marketing)
BASE_SUBSCRIPTION_RATE = 0.117


# Score bands: (min score, css class, level label, suggested action, action detail)
SCORE_BANDS = [
    (0.7, "high", "Probabilità alta", "Chiama per primo", "Metti questo cliente in cima alla lista delle chiamate."),
    (0.4, "medium", "Probabilità media", "Chiama dopo i clienti ad alta probabilità", "Vale una chiamata dopo aver coperto i clienti più promettenti."),
    (0.0, "low", "Probabilità bassa", "Chiama per ultimo, o salta", "Chiamalo solo se resta budget dopo gli altri clienti."),
]


# Course information shown in the header and footer
COURSE_NAME = "AI for Data Analysis"
COURSE_REPO_URL = "https://github.com/AndreaCorvaglia0/ai-course"


# Slider configuration for numeric features
# "default" is used when no reference dataset is available (dataset medians)
SLIDER_CONFIG = {
    "age": {"min": 18, "max": 95, "step": 1, "default": 39},
    "balance": {"min": -10000, "max": 100000, "step": 100, "default": 400},
    "campaign": {"min": 1, "max": 50, "step": 1, "default": 2},
    "pdays": {"min": -1, "max": 900, "step": 1, "default": -1},
    "previous": {"min": 0, "max": 50, "step": 1, "default": 0},
    "day_of_week": {"min": 1, "max": 31, "step": 1, "default": 16},
}


# Example customers that can be loaded into the inputs with one click
EXAMPLE_CUSTOMERS = {
    "Cliente di esempio 1": {
        "summary": "Pensionato, 66 anni, nessun debito, ha detto sì nella campagna precedente",
        "values": {
            "age": 66,
            "job": "retired",
            "marital": "married",
            "education": "tertiary",
            "default": "no",
            "balance": 4500,
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "month": "mar",
            "campaign": 1,
            "pdays": 95,
            "previous": 3,
            "poutcome": "success",
        },
    },
    "Cliente di esempio 2": {
        "summary": "Operaio, 35 anni, mutuo e prestito personale, ha detto no nella campagna precedente",
        "values": {
            "age": 35,
            "job": "blue-collar",
            "marital": "married",
            "education": "primary",
            "default": "no",
            "balance": -200,
            "housing": "yes",
            "loan": "yes",
            "contact": "cellular",
            "month": "may",
            "campaign": 6,
            "pdays": 350,
            "previous": 1,
            "poutcome": "failure",
        },
    },
}


def load_css():
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def get_score_band(proba: float) -> tuple[str, str, str, str]:
    """Return css class, level label, suggested action and action detail for a score."""
    for min_score, css_class, level, action, detail in SCORE_BANDS:
        if proba >= min_score:
            return css_class, level, action, detail
    return SCORE_BANDS[-1][1:]


def get_feature_info(feature_name: str) -> tuple[str, str]:
    """Get label and description for a feature from the info dictionary."""
    if feature_name in FEATURE_INFO:
        return FEATURE_INFO[feature_name]
    return feature_name.replace('_', ' ').capitalize(), ""


def load_example_customer(values: dict) -> None:
    """Copy an example customer's values into the input widgets."""
    for feat, value in values.items():
        st.session_state[f"input_{feat}"] = value


def section_header(number: int, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-header">
            <div class="step-number">{number}</div>
            <div class="step-content">
                <h2>{title}</h2>
                <p>{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        f"""
        <div class="hero-header">
            <div class="hero-icon">🎯</div>
            <h1>Bank Marketing AI Predictor</h1>
            <p class="hero-subtitle">
                Prima di chiamare un cliente per proporre un <strong>deposito a termine</strong>,
                stima quanto è probabile che accetti e decidi chi chiamare per primo.
            </p>
            <p class="course-note">
                Questa è una webapp di esempio del corso <strong>{COURSE_NAME}</strong>.
                <a href="{COURSE_REPO_URL}" target="_blank" rel="noopener">Vedi il corso su GitHub →</a>
            </p>
            <div class="hero-badges">
                <span class="badge">MLflow Pipeline</span>
                <span class="badge">Predizione in tempo reale</span>
                <span class="badge">Esempio del corso</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # When / who / what: the use case in three cards
    st.markdown(
        """
        <div class="usage-grid">
            <div class="usage-card">
                <div class="usage-label">Quando usarla</div>
                <p>Mentre pianifichi una <strong>campagna di telemarketing</strong>, <strong>prima</strong> di fare la chiamata.</p>
            </div>
            <div class="usage-card">
                <div class="usage-label">Su quale cliente</div>
                <p>Un <strong>cliente già della banca</strong> che stai pensando di chiamare per proporre un deposito a termine.</p>
            </div>
            <div class="usage-card">
                <div class="usage-label">Cosa ottieni</div>
                <p>Uno <strong>score di probabilità</strong> che il cliente sottoscriva se chiamato, e una
                <strong>priorità di chiamata</strong> suggerita.</p>
            </div>
        </div>
        <p class="usage-footnote">
            La durata della chiamata non viene chiesta: si conosce solo a chiamata avvenuta, quindi il modello non la usa.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Model info in compact header
    _, col_info, _ = st.columns([1, 2, 1])
    with col_info:
        st.markdown(
            f"""
            <div class="model-info-bar">
                <div class="info-item">
                    <span class="info-label">Modello</span>
                    <span class="info-value">{MLFLOW_MODEL_NAME}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Stato</span>
                    <span class="info-value"><span class="status-dot"></span>Attivo</span>
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
    section_header(
        1,
        "Descrivi il cliente che vuoi chiamare",
        "Inserisci ciò che la banca sa già del cliente e come intendi chiamarlo, "
        "oppure parti da un esempio.",
    )

    # Example customers: fill all inputs with one click
    with st.container(horizontal=True, vertical_alignment="center"):
        for name, customer in EXAMPLE_CUSTOMERS.items():
            st.button(
                name,
                icon=":material/person:",
                help=customer["summary"],
                on_click=load_example_customer,
                args=(customer["values"],),
            )
    st.caption("  \n".join(
        f"**{name}:** {customer['summary'].lower()}." for name, customer in EXAMPLE_CUSTOMERS.items()
    ))

    input_values = {}

    def render_feature_input(feat):
        """Render appropriate input widget for a feature, with its description below."""
        # Defaults go in session state (not in value=) so the example
        # customer buttons can overwrite them without widget warnings
        key = f"input_{feat}"
        label, description = get_feature_info(feat)

        if feat in categorical_features:
            cats = categories_by_feature.get(feat, [])
            value = st.selectbox(
                label,
                options=cats,
                format_func=lambda option: OPTION_LABELS.get(option, option),
                key=key,
            )
        else:
            stats = stats_numeric.get(feat, {})
            min_val = stats.get("min", None)
            max_val = stats.get("max", None)

            # Use slider for specific features
            if feat in SLIDER_CONFIG:
                slider_conf = SLIDER_CONFIG[feat]
                # Use stats if available, otherwise use config
                actual_min = int(min_val) if min_val is not None else slider_conf["min"]
                actual_max = int(max_val) if max_val is not None else slider_conf["max"]
                st.session_state.setdefault(key, int(stats.get("median", slider_conf["default"])))

                value = st.slider(
                    label,
                    min_value=actual_min,
                    max_value=actual_max,
                    step=slider_conf["step"],
                    key=key,
                )
            else:
                # Use number input for other numeric features
                st.session_state.setdefault(key, float(stats.get("median", 0.0)))
                value = st.number_input(
                    label,
                    min_value=float(min_val) if min_val is not None else None,
                    max_value=float(max_val) if max_val is not None else None,
                    step=1.0,
                    key=key,
                )

        # Show the dataset column name too, so it can be matched with the notebooks
        st.caption(f"`{feat}` · {description}" if description else f"`{feat}`")
        return value

    # Only the groups' features the model actually uses; anything else goes in "Altri input"
    model_features = categorical_features + numeric_features
    groups = [
        (title, subtitle, [f for f in feats if f in model_features])
        for title, subtitle, feats in FEATURE_GROUPS
    ]
    grouped = {f for _, _, feats in groups for f in feats}
    other = [f for f in model_features if f not in grouped]
    if other:
        groups.append(("Altri input", "Altre variabili usate dal modello", other))
    groups = [g for g in groups if g[2]]

    # 2-column grid of groups
    for row_start in range(0, len(groups), 2):
        cols = st.columns(2, gap="large")
        for col, (title, subtitle, feats) in zip(cols, groups[row_start:row_start + 2]):
            with col:
                st.markdown(
                    f'<div class="group-title">{title}</div><div class="group-subtitle">{subtitle}</div>',
                    unsafe_allow_html=True,
                )
                for feat in feats:
                    input_values[feat] = render_feature_input(feat)

    st.markdown("<br>", unsafe_allow_html=True)

    # Step 2: Prediction Button
    section_header(
        2,
        "Stima la probabilità di sottoscrizione",
        "Il modello confronta questo cliente con gli esiti delle campagne passate.",
    )

    # Center the button
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        predict_button = st.button(
            "Stima la probabilità di sottoscrizione",
            icon=":material/insights:",
            type="primary",
            width="stretch",
        )

    # Step 3: Results
    if predict_button:
        with st.spinner("Calcolo dello score in corso..."):
            try:
                pred = make_prediction(input_values)
            except Exception as e:
                st.error(f"❌ Errore durante la predizione: {str(e)}")
                st.stop()

        st.markdown("<br>", unsafe_allow_html=True)
        section_header(
            3,
            "Risultato e azione suggerita",
            "Usa lo score per decidere in che posizione mettere il cliente nella lista delle chiamate.",
        )

        if "proba_positive" in pred:
            proba = pred["proba_positive"]
            level_class, level_label, action, action_detail = get_score_band(proba)

            # Debug info
            with st.expander("🔍 Debug Info (per sviluppatori)", expanded=False):
                st.write("**Output grezzo della predizione:**")
                st.json(pred)
                st.write("**Valori di input:**")
                st.json(input_values)

            # Dynamic color based on probability
            st.markdown(
                f"""
                <div class="prediction-card {level_class}">
                    <div class="prediction-header">
                        <div class="prediction-icon">{'🎯' if level_class == 'high' else '📈' if level_class == 'medium' else '📉'}</div>
                        <div class="prediction-level">{level_label}</div>
                    </div>
                    <div class="prediction-main">
                        <div class="prediction-value">{proba:.0%}</div>
                        <div class="prediction-label">Probabilità di sottoscrizione se chiamato</div>
                    </div>
                    <div class="probability-bar">
                        <div class="probability-fill" style="width: {proba*100}%"></div>
                    </div>
                    <div class="probability-scale">
                        <span>Bassa · sotto il 40%</span>
                        <span>Media · 40–70%</span>
                        <span>Alta · dal 70% in su</span>
                    </div>
                    <div class="prediction-action">
                        <span class="footer-label">Azione suggerita</span>
                        <span class="action-value">{action}</span>
                        <span class="action-detail">{action_detail}</span>
                    </div>
                    <div class="prediction-note">
                        <strong>Come leggere lo score.</strong> Usalo per confrontare i clienti e decidere l'ordine
                        delle chiamate, non come probabilità esatta di successo. Nelle campagne passate solo
                        circa il {BASE_SUBSCRIPTION_RATE:.0%} dei clienti chiamati ha sottoscritto; il modello dà
                        più peso alle rare risposte "sì", quindi i suoi score sono più alti di quel valore.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Fallback for regression
            st.success(f"Predizione: {pred.get('prediction', 'N/A')}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown(
        f"""
        <div class="footer">
            <p>Webapp di esempio del corso {COURSE_NAME} •
            <a href="{COURSE_REPO_URL}" target="_blank" rel="noopener">GitHub</a></p>
            <p>Powered by MLflow • Streamlit • scikit-learn</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
