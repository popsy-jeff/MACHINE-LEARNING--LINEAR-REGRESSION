"""
STUDENT PERFORMANCE PREDICTOR — Streamlit Web App
----------------------------------------------------
A browser-based app for entering a student's study habits and getting a
live Performance Index estimate from your trained linear regression model,
plus batch scoring, model performance diagnostics, and an interactive
coefficient explorer.

HOW TO RUN:
1. Install streamlit and plotly once (if you don't already have them):
       pip install streamlit plotly
2. Place this file in the SAME FOLDER as 'Student_Performance.csv'
   (your LINEAR REGRESSION project folder).
3. From that folder, run:
       streamlit run student_performance_app.py

LAYOUT:
The sidebar is navigation ONLY (page switcher). All forms — student
details, batch upload — live in the main content area for each page.

NOTE ON THE LOG FILE:
Writes to 'student_performance_predictions_log.csv' in the same folder.

ICONS:
Uses Google's Material Symbols webfont (the same icon set used by
Material UI) loaded from Google Fonts' CDN — no emoji anywhere in the UI.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Always work relative to THIS file's own folder, regardless of where the
# app is launched from (terminal, VS Code Run button, double-click, etc.)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

LOG_FILE = 'student_performance_predictions_log.csv'

LOG_COLUMNS = [
    'Timestamp', 'Hours_Studied', 'Previous_Scores', 'Sleep_Hours',
    'Sample_Papers_Practiced', 'Extracurricular_Activities', 'PredictedIndex',
]

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="wide"
)


# ----------------------------------------------------------------------
# ICONS — Material Symbols (Google's official icon set, used by Material UI)
# ----------------------------------------------------------------------
def icon(name: str, size: int = 20, color: str = "currentColor", valign: str = "middle") -> str:
    return (
        f'<span class="material-symbols-outlined" '
        f'style="font-size:{size}px; color:{color}; vertical-align:{valign}; '
        f'line-height:1;">{name}</span>'
    )


def stat_card_html(label: str, value: str, icon_name: str, color_var: str) -> str:
    """Small colour-coded stat block used on the Dashboard and other summary rows."""
    return (
        f"<div class='stat-card' style='border-color:{color_var};'>"
        f"{icon(icon_name, 22, color_var)}"
        f"<span class='stat-value' style='color:{color_var};'>{value}</span>"
        f"<span class='stat-label'>{label}</span>"
        f"</div>"
    )


def nav_card_html(title: str, desc: str, icon_name: str, color_var: str, bg_var: str) -> str:
    """Colour-coded description card used above a page-jump button."""
    return (
        f"<div class='nav-card' style='border-color:{color_var}; background:{bg_var};'>"
        f"<div class='nav-card-title'>{icon(icon_name, 18, color_var)} &nbsp;{title}</div>"
        f"<div class='nav-card-desc'>{desc}</div>"
        f"</div>"
    )


def dark_df(df, fmt: dict | None = None):
    """Wrap a DataFrame in a Styler with explicit dark cell colors.

    st.dataframe's grid otherwise picks its palette from the Streamlit
    theme (normally set via .streamlit/config.toml). Since this app is
    a single file with no separate config folder, we bake the dark
    colors directly into the Styler instead so every table stays
    readable regardless of theme detection.
    """
    styler = df.style.set_properties(**{
        'background-color': '#141B18',
        'color': '#EAEFEC',
        'border-color': 'rgba(47, 191, 143, 0.22)',
    })
    if fmt:
        styler = styler.format(fmt)
    return styler


def goto(page_name: str):
    """Programmatically switch the active sidebar page and rerun.

    We can't set st.session_state['nav_page'] directly here because the
    sidebar radio (key='nav_page') has already been instantiated earlier
    in this run — Streamlit disallows modifying a widget's own key after
    it's been created. Instead we stash the request in a plain variable
    and apply it to 'nav_page' at the very top of the next run, before
    the radio widget is (re)created.
    """
    st.session_state['nav_page_request'] = page_name
    st.rerun()


# ----------------------------------------------------------------------
# GLOBAL STYLE
# ----------------------------------------------------------------------
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/icon?family=Material+Symbols+Outlined"
          rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
          rel="stylesheet">
    <style>
        :root {
            --brand-primary:  #2FBF8F;   /* brighter teal — matches the loan app's brand */
            --brand-primary-light: rgba(47, 191, 143, 0.14);
            --brand-accent:   #E0B75C;   /* yellow/gold */
            --brand-accent-light: rgba(224, 183, 92, 0.12);
            --brand-approve:  #3FD08A;
            --brand-reject:   #F2685C;   /* red */
            --brand-reject-light: rgba(242, 104, 92, 0.10);
            --brand-purple:   #A78BFA;   /* purple */
            --brand-purple-light: rgba(167, 139, 250, 0.12);
            --brand-bg-card:  rgba(47, 191, 143, 0.07);
            --brand-border:   rgba(47, 191, 143, 0.22);
            --brand-text:     #EAEFEC;
        }

        .info-card-red {
            border-radius: 12px; padding: 16px 18px; margin-bottom: 10px;
            background: var(--brand-reject-light); border: 1px solid var(--brand-reject);
            color: var(--brand-text);
        }
        .info-card-yellow {
            border-radius: 12px; padding: 16px 18px; margin-bottom: 10px;
            background: var(--brand-accent-light); border: 1px solid var(--brand-accent);
            color: var(--brand-text);
        }
        .info-card-purple {
            border-radius: 12px; padding: 16px 18px; margin-bottom: 10px;
            background: var(--brand-purple-light); border: 1px solid var(--brand-purple);
            color: var(--brand-text);
        }
        .stat-card {
            border-radius: 12px; padding: 16px 14px; text-align: center;
            background: rgba(255,255,255,0.02); border: 1px solid;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .stat-card:hover { transform: translateY(-2px); }
        .stat-card .stat-value { font-size: 1.5rem; font-weight: 700; display: block; margin: 6px 0 2px 0; }
        .stat-card .stat-label { font-size: 0.82rem; opacity: 0.85; }
        .nav-card {
            border-radius: 12px; padding: 14px 16px; margin-bottom: 8px;
            border: 1px solid; transition: transform 0.15s ease;
        }
        .nav-card:hover { transform: translateX(3px); }
        .nav-card-title { font-weight: 600; font-size: 1rem; margin-bottom: 2px; }
        .nav-card-desc { font-size: 0.85rem; opacity: 0.85; }

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--brand-text); }

        /* Force dark mode regardless of the user's OS/browser theme or
           Streamlit's default settings — this app is designed dark-only,
           so we pin the app shell's background explicitly rather than
           relying on config.toml alone (belt-and-braces: config.toml
           handles widgets/menus, this CSS handles the page chrome). */
        [data-testid="stAppViewContainer"], [data-testid="stHeader"],
        .stApp, body {
            background-color: #0B100E !important;
        }
        [data-testid="stAppViewContainer"] > .main {
            background-color: #0B100E !important;
        }

        .block-container {
            padding-top: 1.6rem;
            padding-left: 3rem;
            padding-right: 3rem;
            max-width: 100%;
            width: 100%;
        }

        .block-container p, .block-container li {
            max-width: 1100px;
        }

        h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; }
        h1 { font-size: 1.85rem; color: var(--brand-primary); }

        div[data-testid="stMetric"] {
            background: var(--brand-bg-card);
            border: 1px solid var(--brand-border);
            border-radius: 12px;
            padding: 14px 18px;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(47, 191, 143, 0.18);
        }

        button[kind="primary"], button[kind="formSubmit"] {
            background-color: var(--brand-primary) !important;
            color: #0E1512 !important;
            border: none !important;
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }
        button[kind="primary"]:hover, button[kind="formSubmit"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(47, 191, 143, 0.35);
        }

        section[data-testid="stSidebar"] { min-width: 240px; max-width: 270px; }
        section[data-testid="stSidebar"] > div { background: #0B100E; }
        section[data-testid="stSidebar"] .block-container { padding-top: 0.5rem; }

        section[data-testid="stSidebar"] .stRadio > label { display: none; }

        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
            display: flex; flex-direction: column; gap: 6px;
        }

        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
            position: relative;
            padding: 11px 14px 11px 18px;
            border-radius: 10px;
            margin-bottom: 0;
            background: transparent;
            border: 1px solid transparent;
            cursor: pointer;
            overflow: hidden;
            transition: background 0.18s ease, border-color 0.18s ease,
                        transform 0.18s ease, box-shadow 0.18s ease;
        }

        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label::before {
            content: "";
            position: absolute; left: 0; top: 50%;
            width: 3px; height: 0%;
            background: var(--brand-primary);
            border-radius: 0 3px 3px 0;
            transform: translateY(-50%);
            transition: height 0.2s ease;
        }

        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
            background: var(--brand-primary-light);
            border-color: var(--brand-border);
            transform: translateX(3px);
        }
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover::before {
            height: 55%;
        }

        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
            background: var(--brand-primary-light);
            border-color: var(--brand-border);
            box-shadow: 0 2px 10px rgba(47, 191, 143, 0.15);
        }
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked)::before {
            height: 70%;
        }
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) p {
            color: var(--brand-primary);
            font-weight: 600;
        }

        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
            font-size: 1.05rem;
            font-weight: 500;
            transition: color 0.18s ease;
            margin: 0;
        }

        @keyframes fadeSlideIn {
            from { opacity: 0; transform: translateY(8px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .result-block { animation: fadeSlideIn 0.45s ease-out; }

        .status-pill {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 8px 16px; border-radius: 999px;
            font-weight: 600; font-size: 1.05rem;
            animation: fadeSlideIn 0.45s ease-out;
        }
        .status-approved { background: rgba(63, 208, 138, 0.16); color: var(--brand-approve); }

        .info-card {
            border-radius: 12px; padding: 16px 18px; margin-bottom: 10px;
            background: var(--brand-bg-card); border: 1px solid var(--brand-border);
            color: var(--brand-text);
        }
        .section-label {
            font-weight: 600; color: var(--brand-primary);
            display: flex; align-items: center; gap: 6px; margin-bottom: 4px;
        }
        .nav-brand {
            display: flex; align-items: center; gap: 8px;
            font-weight: 700; color: var(--brand-primary);
            font-size: 1.15rem; padding: 10px 6px 16px 6px;
            margin-bottom: 8px;
            border-bottom: 1px solid var(--brand-border);
        }

        /* ------------------------------------------------------------
           NATIVE WIDGET DARK-MODE FIXES
           No .streamlit/config.toml is used (single-file app), so
           Streamlit's native widgets — inputs, sliders, dropdowns,
           expanders, the file uploader, code blocks — default to
           light-mode chrome unless overridden here directly. Dataframe
           grids are handled separately via the dark_df() Styler helper
           above, since their cell colors aren't reachable by CSS.
           ------------------------------------------------------------ */

        /* Text / number inputs, textareas, and selectbox trigger */
        .stTextInput input, .stNumberInput input, .stTextArea textarea,
        div[data-baseweb="select"] > div {
            background-color: #141B18 !important;
            color: var(--brand-text) !important;
            border-color: var(--brand-border) !important;
        }
        .stTextInput input::placeholder, .stTextArea textarea::placeholder {
            color: rgba(234, 239, 236, 0.45) !important;
        }
        div[data-baseweb="select"] * { color: var(--brand-text) !important; }

        /* Selectbox / multiselect dropdown popover — rendered in a
           portal, so it won't inherit the app-level background rule */
        div[data-baseweb="popover"], div[data-baseweb="menu"],
        ul[role="listbox"] {
            background-color: #141B18 !important;
            color: var(--brand-text) !important;
        }
        ul[role="listbox"] li, div[data-baseweb="menu"] li {
            color: var(--brand-text) !important;
        }
        ul[role="listbox"] li:hover, div[data-baseweb="menu"] li:hover {
            background-color: var(--brand-primary-light) !important;
        }

        /* Slider value bubble + tick labels */
        div[data-testid="stSlider"] div[role="slider"] {
            color: #0E1512 !important;
        }
        div[data-testid="stSlider"] .stMarkdown, div[data-testid="stSlider"] label,
        div[data-testid="stSlider"] p {
            color: var(--brand-text) !important;
        }
        div[data-testid="stTickBar"] { color: var(--brand-text) !important; }
        div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"] {
            color: var(--brand-text) !important;
        }
        div[data-baseweb="tooltip"] {
            background-color: #141B18 !important;
            color: var(--brand-text) !important;
        }

        /* Expander header + body */
        details[data-testid="stExpander"] summary,
        details[data-testid="stExpander"] summary p,
        details[data-testid="stExpander"] svg {
            color: var(--brand-text) !important;
        }
        details[data-testid="stExpander"] {
            background-color: rgba(255,255,255,0.02) !important;
            border: 1px solid var(--brand-border) !important;
            border-radius: 10px;
        }

        /* File uploader dropzone */
        section[data-testid="stFileUploaderDropzone"] {
            background-color: #141B18 !important;
            border-color: var(--brand-border) !important;
        }
        section[data-testid="stFileUploaderDropzone"] * {
            color: var(--brand-text) !important;
        }
        section[data-testid="stFileUploaderDropzone"] button {
            color: #0E1512 !important;
        }

        /* Code blocks (st.code) */
        .stCodeBlock, .stCodeBlock pre, .stCodeBlock code, pre, code {
            background-color: #141B18 !important;
            color: #A9F0D2 !important;
        }

        /* Secondary (non-primary) buttons — keep label readable in
           both resting and hover states */
        button[kind="secondary"], .stDownloadButton button {
            color: var(--brand-text) !important;
            border-color: var(--brand-border) !important;
            background-color: transparent !important;
        }
        button[kind="secondary"]:hover, .stDownloadButton button:hover {
            color: var(--brand-primary) !important;
            border-color: var(--brand-primary) !important;
        }

        /* Dataframe / table grid header + cell text */
        div[data-testid="stDataFrame"] { color: var(--brand-text) !important; }
        div[data-testid="stDataFrame"] * { color: var(--brand-text) !important; }

        /* Metric label/value/delta */
        div[data-testid="stMetric"] label, div[data-testid="stMetric"] * {
            color: var(--brand-text) !important;
        }
        div[data-testid="stMetricValue"] { color: var(--brand-primary) !important; }

        /* Generic form / widget labels and captions */
        label, .stMarkdown p, .stCaption, [data-testid="stCaptionContainer"] {
            color: var(--brand-text) !important;
        }
        [data-testid="stCaptionContainer"] { opacity: 0.85; }

        /* Checkbox / radio labels outside the sidebar nav */
        .stCheckbox label p, .stRadio label p { color: var(--brand-text) !important; }
    </style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# 1. REBUILD THE PREPROCESSING + TRAINING PIPELINE FROM THE NOTEBOOK
# ----------------------------------------------------------------------
# The notebook's final scaled multi-feature model uses these four numeric
# predictors (Extracurricular Activities was explored via correlation but
# not included in the final fitted model, so it's excluded here too).
INPUTS = ['Previous Scores', 'Hours Studied', 'Sleep Hours', 'Sample Question Papers Practiced']
TARGET = 'Performance Index'


@st.cache_resource
def build_model():
    perf_df = pd.read_csv('Student_Performance.csv')
    perf_df = perf_df.drop_duplicates()
    perf_df = perf_df.dropna(subset=INPUTS + [TARGET])

    train_df, test_df = train_test_split(perf_df, test_size=0.2, random_state=42)

    train_inputs = train_df[INPUTS].copy()
    train_target = train_df[TARGET].copy()

    scaler = StandardScaler()
    train_inputs_scaled = scaler.fit_transform(train_inputs)

    model = LinearRegression()
    model.fit(train_inputs_scaled, train_target)

    weights_df = pd.DataFrame({
        'Feature': INPUTS,
        'Weight': model.coef_
    }).sort_values('Weight', key=abs, ascending=False)

    # ---- test-set performance, for the Model Performance page ----
    test_inputs = test_df[INPUTS].copy()
    test_target = test_df[TARGET].copy()
    test_inputs_scaled = scaler.transform(test_inputs)
    test_preds = model.predict(test_inputs_scaled)

    rmse = float(np.sqrt(mean_squared_error(test_target, test_preds)))
    mae = float(mean_absolute_error(test_target, test_preds))
    r2 = float(r2_score(test_target, test_preds))

    mean_baseline_preds = np.full(len(test_target), train_target.mean())
    mean_baseline_rmse = float(np.sqrt(mean_squared_error(test_target, mean_baseline_preds)))

    performance = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'test_size': len(test_target),
        'mean_baseline_rmse': mean_baseline_rmse,
        'test_actual': test_target.reset_index(drop=True),
        'test_pred': pd.Series(test_preds),
    }

    feature_stats = {
        f: {
            'min': float(perf_df[f].min()),
            'max': float(perf_df[f].max()),
            'mean': float(perf_df[f].mean()),
        } for f in INPUTS
    }

    return model, scaler, weights_df, performance, feature_stats, perf_df


MODEL, SCALER, WEIGHTS_DF, PERFORMANCE, FEATURE_STATS, PERF_DF = build_model()


# ----------------------------------------------------------------------
# 2. PREDICTION + LOGGING
# ----------------------------------------------------------------------
def predict_student(student_dict):
    df = pd.DataFrame({k: [v] for k, v in student_dict.items()})[INPUTS]
    scaled = SCALER.transform(df)
    predicted_index = float(MODEL.predict(scaled)[0])
    predicted_index = max(0.0, min(100.0, predicted_index))  # index is defined on 0–100
    return predicted_index, scaled


def log_prediction(student_dict, predicted_index, extracurricular):
    file_exists = os.path.isfile(LOG_FILE)
    row = {
        'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Hours_Studied': student_dict['Hours Studied'],
        'Previous_Scores': student_dict['Previous Scores'],
        'Sleep_Hours': student_dict['Sleep Hours'],
        'Sample_Papers_Practiced': student_dict['Sample Question Papers Practiced'],
        'Extracurricular_Activities': extracurricular,
        'PredictedIndex': round(predicted_index, 2),
    }
    row_df = pd.DataFrame([row], columns=LOG_COLUMNS)
    row_df.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)


def performance_band(predicted_index: float):
    """Buckets the predicted index relative to the training set's distribution."""
    q1 = PERF_DF[TARGET].quantile(0.33)
    q2 = PERF_DF[TARGET].quantile(0.67)
    if predicted_index <= q1:
        return "Lower Third", "var(--brand-accent)", "trending_down"
    elif predicted_index <= q2:
        return "Middle Third", "var(--brand-primary)", "trending_flat"
    else:
        return "Upper Third", "var(--brand-approve)", "trending_up"


# ----------------------------------------------------------------------
# 3. SIDEBAR — NAVIGATION ONLY
# ----------------------------------------------------------------------
PAGES = {
    "Dashboard": "dashboard",
    "Prediction": "insights",
    "Batch Prediction": "upload_file",
    "Model Performance": "monitoring",
    "Model Insights": "explore",
    "History & Log": "history",
    "About": "info",
}

with st.sidebar:
    st.markdown(
        f"<div class='nav-brand'>{icon('school', 24)} Performance Predictor</div>",
        unsafe_allow_html=True
    )

    if 'nav_page_request' in st.session_state:
        st.session_state['nav_page'] = st.session_state.pop('nav_page_request')

    page = st.radio(
        "Navigate",
        list(PAGES.keys()),
        label_visibility="collapsed",
        key="nav_page",
    )


# ----------------------------------------------------------------------
# 4. HEADER (shown on every page)
# ----------------------------------------------------------------------
st.markdown(
    f"<h1>{icon('school', 30, 'var(--brand-primary)')} Student Performance Predictor</h1>",
    unsafe_allow_html=True
)


# ----------------------------------------------------------------------
# 5. PAGE: DASHBOARD — landing page with summary stats and quick nav
# ----------------------------------------------------------------------
if page == "Dashboard":
    st.caption("A quick overview of the model, your prediction history, and where to go next.")

    if os.path.isfile(LOG_FILE):
        _log_df = pd.read_csv(LOG_FILE)
        _total = len(_log_df)
        _avg_idx = f"{_log_df['PredictedIndex'].mean():.1f}" if _total else "—"
        _max_idx = f"{_log_df['PredictedIndex'].max():.1f}" if _total else "—"
    else:
        _total, _avg_idx, _max_idx = 0, "—", "—"

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(stat_card_html("Students Logged", str(_total), "groups", "var(--brand-primary)"), unsafe_allow_html=True)
    with s2:
        st.markdown(stat_card_html("Avg. Predicted Index", _avg_idx, "leaderboard", "var(--brand-purple)"), unsafe_allow_html=True)
    with s3:
        st.markdown(stat_card_html("Highest Predicted", _max_idx, "trending_up", "var(--brand-accent)"), unsafe_allow_html=True)
    with s4:
        st.markdown(stat_card_html("Test R²", f"{PERFORMANCE['r2']:.2f}", "target", "var(--brand-reject)"), unsafe_allow_html=True)

    st.write("")

    st.markdown(
        f"<div class='section-label' style='font-size:1.05rem;'>"
        f"{icon('model_training', 20)} Model at a Glance</div>",
        unsafe_allow_html=True
    )
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.markdown(stat_card_html("RMSE", f"{PERFORMANCE['rmse']:.2f} pts", "rule", "var(--brand-primary)"), unsafe_allow_html=True)
    with q2:
        st.markdown(stat_card_html("MAE", f"{PERFORMANCE['mae']:.2f} pts", "straighten", "var(--brand-purple)"), unsafe_allow_html=True)
    with q3:
        st.markdown(stat_card_html("Features Used", str(len(INPUTS)), "list_alt", "var(--brand-accent)"), unsafe_allow_html=True)
    with q4:
        st.markdown(stat_card_html("Test Set Size", str(PERFORMANCE['test_size']), "science", "var(--brand-reject)"), unsafe_allow_html=True)

    st.divider()

    st.markdown(
        f"<div class='section-label' style='font-size:1.05rem;'>"
        f"{icon('explore', 20)} Jump To</div>",
        unsafe_allow_html=True
    )

    nav_items = [
        ("Prediction", "person_search", "var(--brand-primary)", "var(--brand-bg-card)",
         "Score a Single Student",
         "Fill in one student's study habits and get an instant Performance Index estimate."),
        ("Batch Prediction", "upload_file", "var(--brand-purple)", "var(--brand-purple-light)",
         "Score Many Students",
         "Upload a CSV and get predictions for a whole batch at once."),
        ("Model Performance", "monitoring", "var(--brand-reject)", "var(--brand-reject-light)",
         "Check Model Accuracy",
         "RMSE, MAE, R², actual-vs-predicted plot, and baseline comparisons."),
        ("Model Insights", "insights", "var(--brand-purple)", "var(--brand-purple-light)",
         "Explore the Regression Line",
         "Interactively see how each feature's coefficient shapes predictions."),
        ("History & Log", "history", "var(--brand-primary)", "var(--brand-bg-card)",
         "Review Past Predictions",
         "Every prediction made in this app, with a downloadable CSV log."),
        ("About", "info", "var(--brand-accent)", "var(--brand-accent-light)",
         "Read the Fine Print",
         "The model, the dataset, and its limitations."),
    ]

    nc1, nc2, nc3 = st.columns(3)
    nav_cols = [nc1, nc2, nc3]
    for i, (target, ic, color, bg, title, desc) in enumerate(nav_items):
        with nav_cols[i % 3]:
            st.markdown(nav_card_html(title, desc, ic, color, bg), unsafe_allow_html=True)
            if st.button(f"Open {target}", key=f"dash_nav_{target}", use_container_width=True):
                goto(target)

    st.divider()
    st.markdown(
        f"<div class='info-card-purple'>{icon('school', 16, 'var(--brand-purple)')} "
        f"New here? Start with <b>Prediction</b> to score a student, then visit "
        f"<b>Model Insights</b> to see why the model landed on that number."
        f"</div>",
        unsafe_allow_html=True
    )


# ----------------------------------------------------------------------
# 6. PAGE: PREDICTION
# ----------------------------------------------------------------------
elif page == "Prediction":
    st.caption("Enter the student's study habits below, then click Predict.")

    with st.form("student_form"):
        st.markdown(
            f"<div class='section-label'>{icon('menu_book', 18)} Study Habits</div>",
            unsafe_allow_html=True
        )
        p1, p2 = st.columns(2)
        with p1:
            hours_studied = st.slider("Hours Studied (daily average)", 1, 9, 5)
            previous_scores = st.slider("Previous Scores (0–100)", 40, 99, 70)
        with p2:
            sleep_hours = st.slider("Sleep Hours (daily average)", 4, 9, 7)
            sample_papers = st.slider("Sample Question Papers Practiced", 0, 9, 5)

        st.markdown(
            f"<div class='section-label' style='margin-top:14px;'>{icon('groups', 18)} Extra (log only, not used by the model)</div>",
            unsafe_allow_html=True
        )
        extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])

        st.write("")
        submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")

    if submitted:
        student = {
            'Previous Scores': float(previous_scores),
            'Hours Studied': float(hours_studied),
            'Sleep Hours': float(sleep_hours),
            'Sample Question Papers Practiced': float(sample_papers),
        }
        predicted_index, scaled_row = predict_student(student)
        log_prediction(student, predicted_index, extracurricular)
        st.session_state['last_student'] = student
        st.session_state['last_extracurricular'] = extracurricular
        st.session_state['last_index'] = predicted_index
        st.session_state['last_scaled_row'] = scaled_row

    st.divider()

    if 'last_index' in st.session_state:
        predicted_index = st.session_state['last_index']
        student = st.session_state['last_student']
        extracurricular = st.session_state['last_extracurricular']
        scaled_row = st.session_state['last_scaled_row']

        left, right = st.columns([1.15, 1], gap="large")

        with left:
            st.markdown('<div class="result-block">', unsafe_allow_html=True)

            st.markdown(
                f'<span class="status-pill status-approved">'
                f'{icon("leaderboard", 22)} Predicted Performance Index: {predicted_index:.1f} / 100</span>',
                unsafe_allow_html=True
            )

            band_label, band_color, band_icon = performance_band(predicted_index)
            st.markdown(
                f'&nbsp;<span class="status-pill" style="background:rgba(0,0,0,0.18); '
                f'color:{band_color}; border:1px solid {band_color};">'
                f'{icon(band_icon, 18, band_color)} {band_label} of cohort</span>',
                unsafe_allow_html=True
            )

            st.write("")
            st.progress(float(predicted_index) / 100.0, text="Predicted Performance Index (0–100)")
            m1, m2 = st.columns(2)
            m1.metric("Typical error (RMSE)", f"± {PERFORMANCE['rmse']:.1f} pts")
            m2.metric("Training set average", f"{PERF_DF[TARGET].mean():.1f}")
            st.caption(
                "The RMSE from the held-out test set gives a rough sense of how far "
                "off a typical estimate might be — this is not a formal confidence interval."
            )

            st.markdown(
                f"<div class='info-card' style='margin-top:10px;'>"
                f"{icon('summarize', 16)} &nbsp;"
                f"{int(student['Hours Studied'])} hrs/day studied, previous score "
                f"{int(student['Previous Scores'])}, {int(student['Sleep Hours'])} hrs/day sleep, "
                f"{int(student['Sample Question Papers Practiced'])} sample papers practiced, "
                f"extracurricular activities: {extracurricular}."
                f"</div>",
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(
                f"<div class='info-card-purple'>{icon('insights', 16, 'var(--brand-purple)')} "
                f"Curious why? Visit <b>Model Insights</b> to see how each feature "
                f"pushed this estimate up or down."
                f"</div>",
                unsafe_allow_html=True
            )
            if st.button("Open Model Insights", key="pred_to_insights", use_container_width=True):
                goto("Model Insights")

        with right:
            st.markdown(
                f"<div class='section-label'>{icon('insights', 18)} "
                f"What's driving THIS student's estimate?</div>",
                unsafe_allow_html=True
            )

            # Per-student contribution = weight_i * scaled_value_i for this
            # specific student, not just the model's global coefficients.
            contributions = scaled_row[0] * MODEL.coef_
            contrib_df = pd.DataFrame({
                'Feature': INPUTS,
                'Contribution (pts)': contributions,
            }).sort_values('Contribution (pts)', key=abs, ascending=False)
            contrib_df['Direction'] = contrib_df['Contribution (pts)'].apply(
                lambda v: '↑ raises index' if v > 0 else '↓ lowers index'
            )

            st.dataframe(
                dark_df(contrib_df, {'Contribution (pts)': '{:.2f}'}),
                hide_index=True,
                use_container_width=True
            )
            st.caption(
                "Contribution = this student's scaled value × the model's coefficient for "
                "that feature. This reflects what actually moved THIS estimate away from "
                "the training-set average index, not just what tends to matter overall."
            )

            with st.expander("Show global model coefficients instead"):
                st.dataframe(
                    dark_df(WEIGHTS_DF, {'Weight': '{:.2f}'}),
                    hide_index=True,
                    use_container_width=True
                )
                st.caption("Coefficients are on the standardized scale, so they're directly comparable across features.")

            st.markdown(
                f"<div class='info-card'>{icon('lightbulb', 16)} "
                f"<b>Why previous scores matter most:</b> across the training data it's "
                f"the single strongest predictor of the performance index, well ahead of "
                f"study habits alone."
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            f"<div class='info-card'>{icon('arrow_upward', 16, valign='text-bottom')} "
            f"Fill in the student's details above and click "
            f"<b>Predict</b> to see a result here.</div>",
            unsafe_allow_html=True
        )


# ----------------------------------------------------------------------
# 7. PAGE: BATCH PREDICTION
# ----------------------------------------------------------------------
elif page == "Batch Prediction":
    st.markdown(
        f"<div class='section-label' style='font-size:1.05rem;'>"
        f"{icon('upload_file', 20)} Batch Prediction</div>",
        unsafe_allow_html=True
    )
    st.caption(
        "Upload a CSV of multiple students to get Performance Index estimates for all of "
        "them at once. The file needs the same feature columns as the single-student form."
    )

    with st.expander("Expected CSV columns"):
        st.code(", ".join(INPUTS))
        st.caption("All four columns are required and must be numeric.")

    uploaded_file = st.file_uploader("Upload students CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.markdown(
                f"<div class='info-card' style='border-color:var(--brand-reject);'>"
                f"{icon('error', 18, 'var(--brand-reject)')} Couldn't read that file: {e}</div>",
                unsafe_allow_html=True
            )
            batch_df = None

        if batch_df is not None:
            missing = [c for c in INPUTS if c not in batch_df.columns]
            if missing:
                st.markdown(
                    f"<div class='info-card' style='border-color:var(--brand-reject);'>"
                    f"{icon('warning', 18, 'var(--brand-reject)')} Missing required column(s): "
                    f"<b>{', '.join(missing)}</b></div>",
                    unsafe_allow_html=True
                )
            else:
                st.write(f"Loaded **{len(batch_df)}** students.")
                st.dataframe(dark_df(batch_df.head()), hide_index=True, use_container_width=True)

                if st.button("Run batch prediction", type="primary"):
                    results = []
                    errors = 0
                    for _, row in batch_df.iterrows():
                        try:
                            student_row = {f: float(row[f]) for f in INPUTS}
                            predicted_index, _ = predict_student(student_row)
                            results.append({
                                **row.to_dict(),
                                'PredictedIndex': round(predicted_index, 2),
                            })
                        except Exception:
                            errors += 1
                            results.append({
                                **row.to_dict(),
                                'PredictedIndex': None,
                            })

                    results_df = pd.DataFrame(results)
                    st.session_state['batch_results'] = results_df
                    if errors:
                        st.markdown(
                            f"<div class='info-card' style='border-color:var(--brand-reject);'>"
                            f"{icon('warning', 18, 'var(--brand-reject)')} {errors} row(s) "
                            f"couldn't be scored (bad or missing values).</div>",
                            unsafe_allow_html=True
                        )

    if 'batch_results' in st.session_state:
        st.divider()
        results_df = st.session_state['batch_results']
        scored = results_df.dropna(subset=['PredictedIndex'])

        _b_avg = f"{scored['PredictedIndex'].mean():.1f}" if len(scored) else "—"
        _b_min = f"{scored['PredictedIndex'].min():.1f}" if len(scored) else "—"
        _b_max = f"{scored['PredictedIndex'].max():.1f}" if len(scored) else "—"

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(stat_card_html("Scored", str(len(scored)), "checklist", "var(--brand-primary)"), unsafe_allow_html=True)
        with m2:
            st.markdown(stat_card_html("Avg. Index", _b_avg, "leaderboard", "var(--brand-accent)"), unsafe_allow_html=True)
        with m3:
            st.markdown(stat_card_html("Lowest", _b_min, "trending_down", "var(--brand-reject)"), unsafe_allow_html=True)
        with m4:
            st.markdown(stat_card_html("Highest", _b_max, "trending_up", "var(--brand-purple)"), unsafe_allow_html=True)

        st.write("")
        st.dataframe(dark_df(results_df), hide_index=True, use_container_width=True)

        st.download_button(
            "Download results as CSV",
            data=results_df.to_csv(index=False),
            file_name="batch_performance_predictions.csv",
            mime="text/csv",
            type="primary",
        )


# ----------------------------------------------------------------------
# 8. PAGE: MODEL PERFORMANCE
# ----------------------------------------------------------------------
elif page == "Model Performance":
    st.markdown(
        f"<div class='section-label' style='font-size:1.05rem;'>"
        f"{icon('monitoring', 20)} Model Performance</div>",
        unsafe_allow_html=True
    )
    st.caption(
        f"Evaluated on a held-out test split ({PERFORMANCE['test_size']} students) "
        f"that the model never saw during training."
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(stat_card_html("RMSE", f"{PERFORMANCE['rmse']:.2f} pts", "rule", "var(--brand-primary)"), unsafe_allow_html=True)
    with m2:
        st.markdown(stat_card_html("MAE", f"{PERFORMANCE['mae']:.2f} pts", "straighten", "var(--brand-purple)"), unsafe_allow_html=True)
    with m3:
        st.markdown(stat_card_html("R²", f"{PERFORMANCE['r2']:.3f}", "target", "var(--brand-accent)"), unsafe_allow_html=True)
    with m4:
        st.markdown(stat_card_html("vs. Mean Baseline", f"{(1 - PERFORMANCE['rmse']/PERFORMANCE['mean_baseline_rmse']):.0%} lower RMSE", "compare_arrows", "var(--brand-reject)"), unsafe_allow_html=True)

    st.write("")
    st.caption(
        "RMSE and MAE are in Performance Index points (the index runs roughly 0–100) — "
        "lower is better. R² is the share of variance the model explains (1.0 = perfect, "
        "0.0 = no better than always guessing the average index)."
    )

    st.write("")
    perf_col1, perf_col2 = st.columns([1.2, 1], gap="large")

    with perf_col1:
        st.markdown(
            f"<div class='section-label'>{icon('scatter_plot', 18)} Actual vs. Predicted Index</div>",
            unsafe_allow_html=True
        )
        actual = PERFORMANCE['test_actual']
        pred = PERFORMANCE['test_pred']
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actual, y=pred, mode='markers',
            marker=dict(size=6, opacity=0.5, color='#2FBF8F'),
            name='Test students'
        ))
        line_range = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
        fig.add_trace(go.Scatter(
            x=line_range, y=line_range, mode='lines',
            line=dict(color='#F2685C', dash='dash'),
            name='Perfect prediction'
        ))
        fig.update_layout(
            xaxis_title="Actual Performance Index",
            yaxis_title="Predicted Performance Index",
            height=420,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#EAEFEC',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Points closer to the dashed line are more accurate predictions.")

    with perf_col2:
        st.markdown(
            f"<div class='section-label'>{icon('compare_arrows', 18)} Baseline Comparison</div>",
            unsafe_allow_html=True
        )
        baseline_df = pd.DataFrame({
            "Approach": ["Always predict the mean index", "This model"],
            "RMSE (pts)": [
                f"{PERFORMANCE['mean_baseline_rmse']:.2f}",
                f"{PERFORMANCE['rmse']:.2f}",
            ],
        })
        st.dataframe(dark_df(baseline_df), hide_index=True, use_container_width=True)
        st.caption(
            "A model that just guesses the average training-set index is the simplest "
            "possible baseline. The trained model should clear it by a wide margin."
        )

        st.markdown(
            f"<div class='section-label' style='margin-top:18px;'>{icon('histogram', 18)} Residual Spread</div>",
            unsafe_allow_html=True
        )
        residuals = actual - pred
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=residuals, marker_color='#A78BFA', nbinsx=25))
        fig2.add_vline(x=0, line_dash="dash", line_color="gray")
        fig2.update_layout(
            xaxis_title="Residual (Actual − Predicted, pts)",
            yaxis_title="Count",
            height=280,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#EAEFEC',
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Centered near zero and roughly symmetric is a good sign; a strong skew hints at bias.")

    st.divider()

    st.markdown(
        f"<div class='section-label' style='font-size:1.05rem;'>"
        f"{icon('insights', 20)} Global Feature Coefficients</div>",
        unsafe_allow_html=True
    )
    st.caption("Full ranking of every feature's coefficient in the trained linear regression model (standardized scale).")
    st.dataframe(
        dark_df(WEIGHTS_DF, {'Weight': '{:.2f}'}),
        hide_index=True,
        use_container_width=True
    )

    st.markdown(
        f"<div class='info-card-purple'>{icon('lightbulb', 16, 'var(--brand-purple)')} "
        f"Want to see these coefficients in action? <b>Model Insights</b> lets you drag "
        f"each one and watch the predicted index line respond in real time."
        f"</div>",
        unsafe_allow_html=True
    )
    if st.button("Open Model Insights", key="perf_to_insights"):
        goto("Model Insights")


# ----------------------------------------------------------------------
# 9. PAGE: MODEL INSIGHTS — interactive regression line explorer
# ----------------------------------------------------------------------
elif page == "Model Insights":
    st.markdown(
        f"<div class='section-label' style='font-size:1.05rem;'>"
        f"{icon('insights', 20)} Regression Line Explorer</div>",
        unsafe_allow_html=True
    )
    st.caption(
        "See how the model's fitted coefficients shape the predicted index. "
        "Sliders default to the actual trained model — drag them to explore "
        "what-if scenarios."
    )

    default_feature_idx = INPUTS.index('Previous Scores') if 'Previous Scores' in INPUTS else 0
    feature_choice = st.selectbox("Feature to explore", INPUTS, index=default_feature_idx)
    feature_idx = INPUTS.index(feature_choice)

    default_intercept = float(MODEL.intercept_)
    default_coef = float(MODEL.coef_[feature_idx])

    reset_key_b0 = f"intercept_{feature_idx}"
    reset_key_b1 = f"coef_{feature_idx}"

    top_l, top_r = st.columns([3, 1])
    with top_r:
        st.write("")
        if st.button("Reset to fitted values", use_container_width=True):
            st.session_state[reset_key_b0] = default_intercept
            st.session_state[reset_key_b1] = default_coef
            st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        intercept = st.slider(
            "Intercept (index at average student)",
            0.0, 100.0,
            st.session_state.get(reset_key_b0, default_intercept), 0.5,
            key=reset_key_b0
        )
    with col2:
        coef = st.slider(
            f"Coefficient — {feature_choice} (pts per standard deviation)",
            -30.0, 30.0,
            st.session_state.get(reset_key_b1, default_coef), 0.1,
            key=reset_key_b1
        )

    stats = FEATURE_STATS[feature_choice]
    x_raw = np.linspace(stats['min'], stats['max'], 300)
    x_std = (x_raw - stats['mean']) / (PERF_DF[feature_choice].std() if PERF_DF[feature_choice].std() != 0 else 1)
    y = intercept + coef * x_std

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_raw, y=y, mode='lines', name='Predicted index',
        line=dict(width=3, color='#2FBF8F')
    ))
    fig.add_hline(y=100, line_dash="dot", line_color="gray", annotation_text="Index ceiling (100)")
    fig.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Index floor (0)")
    fig.update_layout(
        title=f"Predicted Performance Index vs. {feature_choice}",
        xaxis_title=feature_choice,
        yaxis_title="Predicted Performance Index",
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#EAEFEC'
    )
    st.plotly_chart(fig, use_container_width=True)

    is_default = (abs(intercept - default_intercept) < 1e-6) and (abs(coef - default_coef) < 1e-6)
    st.markdown(
        f"<div class='info-card'>"
        f"{icon('functions', 16)} Current: intercept = <b>{intercept:.2f}</b>, "
        f"coefficient = <b>{coef:.2f}</b> "
        f"{'(actual fitted model values)' if is_default else '(what-if, not the fitted model)'}<br><br>"
        f"Steeper line (larger |coefficient|) → {feature_choice} moves the predicted index "
        f"more per unit change. Shifting the intercept moves the whole line up or down."
        f"</div>",
        unsafe_allow_html=True
    )

    with st.expander("Model's actual fitted values for this feature"):
        st.write(f"Fitted intercept: `{default_intercept:.4f}`")
        st.write(f"Fitted coefficient for {feature_choice}: `{default_coef:.4f}` per standardized unit")

    st.divider()

    st.markdown(
        f"<div class='section-label' style='font-size:1.05rem;'>"
        f"{icon('bar_chart', 20)} Predicted Index Across {feature_choice}'s Range</div>",
        unsafe_allow_html=True
    )
    st.caption(
        "Holding every other feature at its training-set average, this sweeps "
        f"{feature_choice} across its real observed range and shows the model's "
        "actual predicted index at each point — a direct view of the fitted model, "
        "not the what-if sliders above."
    )

    sweep_raw = np.linspace(stats['min'], stats['max'], 40)
    sweep_rows = []
    for val in sweep_raw:
        row = {f: FEATURE_STATS[f]['mean'] for f in INPUTS}
        row[feature_choice] = val
        row_df = pd.DataFrame([row])[INPUTS]
        scaled = SCALER.transform(row_df)
        pred_val = float(MODEL.predict(scaled)[0])
        sweep_rows.append({feature_choice: val, "Predicted Index": pred_val})

    sweep_df = pd.DataFrame(sweep_rows).set_index(feature_choice)
    st.line_chart(sweep_df, color="#2FBF8F", height=280)
    st.caption("All other features held at their training-set average, so this isolates the effect of this one feature alone.")


# ----------------------------------------------------------------------
# 10. PAGE: HISTORY & LOG
# ----------------------------------------------------------------------
elif page == "History & Log":
    st.markdown(
        f"<div class='section-label' style='font-size:1.05rem;'>"
        f"{icon('history', 20)} Prediction History</div>",
        unsafe_allow_html=True
    )

    if os.path.isfile(LOG_FILE):
        log_df = pd.read_csv(LOG_FILE)
        total = len(log_df)
        avg_idx = f"{log_df['PredictedIndex'].mean():.1f}" if total else "—"
        min_idx = f"{log_df['PredictedIndex'].min():.1f}" if total else "—"
        max_idx = f"{log_df['PredictedIndex'].max():.1f}" if total else "—"

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(stat_card_html("Total Logged", str(total), "database", "var(--brand-primary)"), unsafe_allow_html=True)
        with m2:
            st.markdown(stat_card_html("Average Index", avg_idx, "leaderboard", "var(--brand-accent)"), unsafe_allow_html=True)
        with m3:
            st.markdown(stat_card_html("Lowest", min_idx, "trending_down", "var(--brand-reject)"), unsafe_allow_html=True)
        with m4:
            st.markdown(stat_card_html("Highest", max_idx, "trending_up", "var(--brand-purple)"), unsafe_allow_html=True)

        st.write("")

        if total >= 2 and 'Timestamp' in log_df.columns:
            trend_df = log_df.copy()
            trend_df['Timestamp'] = pd.to_datetime(trend_df['Timestamp'], errors='coerce')
            trend_df = trend_df.dropna(subset=['Timestamp']).sort_values('Timestamp')

            st.markdown(
                f"<div class='section-label'>{icon('show_chart', 18)} Predicted Index Over Time</div>",
                unsafe_allow_html=True
            )
            st.caption("Predicted index for each logged student, in submission order.")
            st.line_chart(
                trend_df.set_index('Timestamp')[['PredictedIndex']],
                color="#A78BFA",
                height=200
            )

        st.divider()
        st.dataframe(
            dark_df(log_df.sort_values('Timestamp', ascending=False)),
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            "Download full log as CSV",
            data=log_df.to_csv(index=False),
            file_name="student_performance_predictions_log.csv",
            mime="text/csv"
        )
    else:
        st.markdown(
            f"<div class='info-card'>{icon('info', 16)} "
            f"No predictions logged yet — submit the form on the Prediction page."
            f"</div>",
            unsafe_allow_html=True
        )
        if st.button("Go make a prediction", key="log_to_pred"):
            goto("Prediction")


# ----------------------------------------------------------------------
# 11. PAGE: ABOUT
# ----------------------------------------------------------------------
elif page == "About":
    st.markdown(
        f"<div class='section-label' style='font-size:1.05rem;'>"
        f"{icon('info', 20)} About This App</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div class='info-card'>"
        f"{icon('school', 16)} <b>This is an academic / portfolio project, not a real "
        f"academic assessment tool.</b> Predicted indexes shown here should not be used "
        f"to make actual grading, placement, or intervention decisions."
        f"</div>",
        unsafe_allow_html=True
    )

    qf1, qf2, qf3, qf4 = st.columns(4)
    with qf1:
        st.markdown(stat_card_html("Algorithm", "Linear Regression", "functions", "var(--brand-primary)"), unsafe_allow_html=True)
    with qf2:
        st.markdown(stat_card_html("Features", str(len(INPUTS)), "list_alt", "var(--brand-purple)"), unsafe_allow_html=True)
    with qf3:
        st.markdown(stat_card_html("Test R²", f"{PERFORMANCE['r2']:.2f}", "target", "var(--brand-accent)"), unsafe_allow_html=True)
    with qf4:
        st.markdown(stat_card_html("Test Set Size", str(PERFORMANCE['test_size']), "science", "var(--brand-reject)"), unsafe_allow_html=True)

    st.write("")

    st.markdown(
        f"<div class='section-label' style='margin-top:14px;'>{icon('model_training', 18)} "
        f"The Model</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "A **linear regression** model trained on `Student_Performance.csv`, predicting "
        "`Performance Index` from four numeric features: previous exam scores, daily hours "
        "studied, daily sleep hours, and sample question papers practiced. Features are "
        "standardized (mean 0, standard deviation 1) before fitting so that coefficients "
        "are directly comparable across features. Extracurricular Activities was explored "
        "during EDA but not included in the final fitted model. See the **Model "
        "Performance** page for RMSE, MAE, R², and a baseline comparison on a held-out "
        "test split."
    )

    st.markdown(
        f"<div class='section-label' style='margin-top:14px;'>{icon('dataset', 18)} "
        f"The Dataset</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"A {len(PERF_DF):,}-row student study-habits dataset, pairing study behavior "
        "with a synthetic Performance Index (roughly 0–100). Previous exam scores are by "
        "far the strongest predictor — study habits refine the estimate around that "
        "baseline rather than dominating it."
    )

    st.markdown(
        f"<div class='section-label' style='margin-top:14px;'>{icon('warning', 18)} "
        f"Limitations</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "- The dataset appears to be synthetically generated for practice purposes — "
        "patterns learned from it are illustrative, not a claim about real student "
        "populations.\n"
        "- Only four numeric features are used; real academic performance depends on "
        "many factors (teaching quality, socioeconomic context, mental health, etc.) "
        "this model never sees.\n"
        "- Predictions are clipped to the 0–100 range for display, but the underlying "
        "linear model has no natural ceiling or floor — extreme inputs can produce "
        "unrealistic raw estimates.\n"
        "- No fairness, bias, or robustness auditing has been performed beyond the "
        "metrics shown on the Model Performance page."
    )

    st.markdown(
        f"<div class='section-label' style='margin-top:14px;'>{icon('code', 18)} "
        f"Tech Stack</div>",
        unsafe_allow_html=True
    )
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown(nav_card_html("Streamlit", "App framework & UI", "web", "var(--brand-primary)", "var(--brand-bg-card)"), unsafe_allow_html=True)
    with t2:
        st.markdown(nav_card_html("scikit-learn", "Model training", "psychology", "var(--brand-purple)", "var(--brand-purple-light)"), unsafe_allow_html=True)
    with t3:
        st.markdown(nav_card_html("Plotly", "Interactive charts", "show_chart", "var(--brand-accent)", "var(--brand-accent-light)"), unsafe_allow_html=True)
    with t4:
        st.markdown(nav_card_html("Pandas / NumPy", "Data processing", "table_chart", "var(--brand-reject)", "var(--brand-reject-light)"), unsafe_allow_html=True)

    st.write("")
    if st.button("Back to Dashboard", key="about_to_dash", use_container_width=True):
        goto("Dashboard")
