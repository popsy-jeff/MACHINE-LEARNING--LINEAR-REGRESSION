"""
HOUSE PRICE PREDICTOR — Streamlit Web App
-------------------------------------------
A browser-based app for entering house details and getting a live SalePrice
estimate from your trained linear regression model, plus batch scoring,
model performance diagnostics, and an interactive coefficient explorer.

HOW TO RUN:
1. Install streamlit and plotly once (if you don't already have them):
       pip install streamlit plotly
2. Place this file in the SAME FOLDER as 'house_prices_practice.csv'
   (your LINEAR REGRESSION project folder).
3. From that folder, run:
       streamlit run house_price_app.py

LAYOUT:
The sidebar is navigation ONLY (page switcher). All forms — house
details, batch upload — live in the main content area for each page.

NOTE ON THE LOG FILE:
Writes to 'house_price_predictions_log.csv' in the same folder.

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

LOG_FILE = 'house_price_predictions_log.csv'

LOG_COLUMNS = [
    'Timestamp', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
    'YearBuilt', 'FullBath', 'BedroomAbvGr', 'LotArea', 'PredictedPrice',
]

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
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
INPUTS = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt', 'FullBath']
TARGET = 'SalePrice'


@st.cache_resource
def build_model():
    house_df = pd.read_csv('house_prices_practice.csv')
    house_df = house_df.dropna(subset=INPUTS + [TARGET])

    train_df, test_df = train_test_split(house_df, test_size=0.2, random_state=42)

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
            'min': float(house_df[f].min()),
            'max': float(house_df[f].max()),
            'mean': float(house_df[f].mean()),
        } for f in INPUTS
    }

    return model, scaler, weights_df, performance, feature_stats, house_df


MODEL, SCALER, WEIGHTS_DF, PERFORMANCE, FEATURE_STATS, HOUSE_DF = build_model()


# ----------------------------------------------------------------------
# 2. PREDICTION + LOGGING
# ----------------------------------------------------------------------
def predict_house(house_dict):
    df = pd.DataFrame({k: [v] for k, v in house_dict.items()})[INPUTS]
    scaled = SCALER.transform(df)
    predicted_price = float(MODEL.predict(scaled)[0])
    return predicted_price, scaled


def log_prediction(house_dict, predicted_price):
    file_exists = os.path.isfile(LOG_FILE)
    row = {
        'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'OverallQual': house_dict['OverallQual'],
        'GrLivArea': house_dict['GrLivArea'],
        'GarageCars': house_dict['GarageCars'],
        'TotalBsmtSF': house_dict['TotalBsmtSF'],
        'YearBuilt': house_dict['YearBuilt'],
        'FullBath': house_dict['FullBath'],
        'BedroomAbvGr': house_dict.get('BedroomAbvGr', ''),
        'LotArea': house_dict.get('LotArea', ''),
        'PredictedPrice': round(predicted_price, 2),
    }
    row_df = pd.DataFrame([row], columns=LOG_COLUMNS)
    row_df.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)


def price_band(predicted_price: float):
    """Buckets the predicted price relative to the training set's price
    distribution, so the number has some context beyond a raw dollar figure."""
    q1 = HOUSE_DF[TARGET].quantile(0.33)
    q2 = HOUSE_DF[TARGET].quantile(0.67)
    if predicted_price <= q1:
        return "Lower Third", "var(--brand-accent)", "trending_down"
    elif predicted_price <= q2:
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
        f"<div class='nav-brand'>{icon('home_work', 24)} House Price Predictor</div>",
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
    f"<h1>{icon('home_work', 30, 'var(--brand-primary)')} House Price Predictor</h1>",
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
        _avg_price = f"${_log_df['PredictedPrice'].mean():,.0f}" if _total else "—"
        _max_price = f"${_log_df['PredictedPrice'].max():,.0f}" if _total else "—"
    else:
        _total, _avg_price, _max_price = 0, "—", "—"

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(stat_card_html("Houses Logged", str(_total), "house", "var(--brand-primary)"), unsafe_allow_html=True)
    with s2:
        st.markdown(stat_card_html("Avg. Predicted Price", _avg_price, "payments", "var(--brand-purple)"), unsafe_allow_html=True)
    with s3:
        st.markdown(stat_card_html("Highest Predicted", _max_price, "trending_up", "var(--brand-accent)"), unsafe_allow_html=True)
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
        st.markdown(stat_card_html("RMSE", f"${PERFORMANCE['rmse']:,.0f}", "rule", "var(--brand-primary)"), unsafe_allow_html=True)
    with q2:
        st.markdown(stat_card_html("MAE", f"${PERFORMANCE['mae']:,.0f}", "straighten", "var(--brand-purple)"), unsafe_allow_html=True)
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
        ("Prediction", "search_hands_free", "var(--brand-primary)", "var(--brand-bg-card)",
         "Price a Single House",
         "Fill in one house's details and get an instant SalePrice estimate."),
        ("Batch Prediction", "upload_file", "var(--brand-purple)", "var(--brand-purple-light)",
         "Price Many Houses",
         "Upload a CSV and get price estimates for a whole batch at once."),
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
        f"New here? Start with <b>Prediction</b> to price a house, then visit "
        f"<b>Model Insights</b> to see why the model landed on that number."
        f"</div>",
        unsafe_allow_html=True
    )


# ----------------------------------------------------------------------
# 6. PAGE: PREDICTION
# ----------------------------------------------------------------------
elif page == "Prediction":
    st.caption("Enter the house's details below, then click Predict.")

    with st.form("house_form"):
        st.markdown(
            f"<div class='section-label'>{icon('home', 18)} Structure & Quality</div>",
            unsafe_allow_html=True
        )
        p1, p2, p3 = st.columns(3)
        with p1:
            overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 5)
            year_built = st.number_input("Year Built", min_value=1870, max_value=2026, value=1995, step=1)
        with p2:
            gr_liv_area = st.number_input("Above-Ground Living Area (sq ft)", min_value=0, value=1500, step=50)
            full_bath = st.number_input("Full Bathrooms", min_value=0, max_value=6, value=2, step=1)
        with p3:
            total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, value=1000, step=50)
            garage_cars = st.number_input("Garage Capacity (cars)", min_value=0, max_value=6, value=2, step=1)

        st.markdown(
            f"<div class='section-label' style='margin-top:14px;'>{icon('grid_view', 18)} Extra (log only, not used by the model)</div>",
            unsafe_allow_html=True
        )
        e1, e2 = st.columns(2)
        with e1:
            bedroom_abv_gr = st.number_input("Bedrooms Above Ground", min_value=0, max_value=10, value=3, step=1)
        with e2:
            lot_area = st.number_input("Lot Area (sq ft)", min_value=0, value=9000, step=100)

        st.write("")
        submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")

    if submitted:
        house = {
            'OverallQual': float(overall_qual),
            'GrLivArea': float(gr_liv_area),
            'GarageCars': float(garage_cars),
            'TotalBsmtSF': float(total_bsmt_sf),
            'YearBuilt': float(year_built),
            'FullBath': float(full_bath),
            'BedroomAbvGr': float(bedroom_abv_gr),
            'LotArea': float(lot_area),
        }
        predicted_price, scaled_row = predict_house(house)
        log_prediction(house, predicted_price)
        st.session_state['last_house'] = house
        st.session_state['last_price'] = predicted_price
        st.session_state['last_scaled_row'] = scaled_row

    st.divider()

    if 'last_price' in st.session_state:
        predicted_price = st.session_state['last_price']
        house = st.session_state['last_house']
        scaled_row = st.session_state['last_scaled_row']

        left, right = st.columns([1.15, 1], gap="large")

        with left:
            st.markdown('<div class="result-block">', unsafe_allow_html=True)

            st.markdown(
                f'<span class="status-pill status-approved">'
                f'{icon("payments", 22)} Predicted Sale Price: ${predicted_price:,.0f}</span>',
                unsafe_allow_html=True
            )

            band_label, band_color, band_icon = price_band(predicted_price)
            st.markdown(
                f'&nbsp;<span class="status-pill" style="background:rgba(0,0,0,0.18); '
                f'color:{band_color}; border:1px solid {band_color};">'
                f'{icon(band_icon, 18, band_color)} {band_label} of market</span>',
                unsafe_allow_html=True
            )

            st.write("")
            m1, m2 = st.columns(2)
            m1.metric("Typical error (RMSE)", f"± ${PERFORMANCE['rmse']:,.0f}")
            m2.metric("Training set average", f"${HOUSE_DF[TARGET].mean():,.0f}")
            st.caption(
                "The RMSE from the held-out test set gives a rough sense of how far "
                "off a typical estimate might be — this is not a formal confidence interval."
            )

            st.markdown(
                f"<div class='info-card' style='margin-top:10px;'>"
                f"{icon('summarize', 16)} &nbsp;"
                f"Quality {int(house['OverallQual'])}/10, {int(house['GrLivArea'])} sq ft living area, "
                f"{int(house['TotalBsmtSF'])} sq ft basement, built {int(house['YearBuilt'])}, "
                f"{int(house['FullBath'])} full bath(s), {int(house['GarageCars'])}-car garage."
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
                f"What's driving THIS house's estimate?</div>",
                unsafe_allow_html=True
            )

            # Per-house contribution = weight_i * scaled_value_i for this
            # specific house, not just the model's global coefficients.
            contributions = scaled_row[0] * MODEL.coef_
            contrib_df = pd.DataFrame({
                'Feature': INPUTS,
                'Contribution ($)': contributions,
            }).sort_values('Contribution ($)', key=abs, ascending=False)
            contrib_df['Direction'] = contrib_df['Contribution ($)'].apply(
                lambda v: '↑ raises price' if v > 0 else '↓ lowers price'
            )

            st.dataframe(
                dark_df(contrib_df, {'Contribution ($)': '{:,.0f}'}),
                hide_index=True,
                use_container_width=True
            )
            st.caption(
                "Contribution = this house's scaled value × the model's coefficient for "
                "that feature. This reflects what actually moved THIS estimate away from "
                "the training-set average price, not just what tends to matter overall."
            )

            with st.expander("Show global model coefficients instead"):
                st.dataframe(
                    dark_df(WEIGHTS_DF, {'Weight': '{:,.0f}'}),
                    hide_index=True,
                    use_container_width=True
                )
                st.caption("Coefficients are on the standardized scale, so they're directly comparable across features.")
    else:
        st.markdown(
            f"<div class='info-card'>{icon('arrow_upward', 16, valign='text-bottom')} "
            f"Fill in the house's details above and click "
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
        "Upload a CSV of multiple houses to get price estimates for all of them at once. "
        "The file needs the same feature columns as the single-house form."
    )

    with st.expander("Expected CSV columns"):
        st.code(", ".join(INPUTS))
        st.caption("All six columns are required and must be numeric.")

    uploaded_file = st.file_uploader("Upload houses CSV", type=["csv"])

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
                st.write(f"Loaded **{len(batch_df)}** houses.")
                st.dataframe(dark_df(batch_df.head()), hide_index=True, use_container_width=True)

                if st.button("Run batch prediction", type="primary"):
                    results = []
                    errors = 0
                    for _, row in batch_df.iterrows():
                        try:
                            house_row = {f: float(row[f]) for f in INPUTS}
                            predicted_price, _ = predict_house(house_row)
                            results.append({
                                **row.to_dict(),
                                'PredictedPrice': round(predicted_price, 2),
                            })
                        except Exception:
                            errors += 1
                            results.append({
                                **row.to_dict(),
                                'PredictedPrice': None,
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
        scored = results_df.dropna(subset=['PredictedPrice'])

        _b_avg = f"${scored['PredictedPrice'].mean():,.0f}" if len(scored) else "—"
        _b_min = f"${scored['PredictedPrice'].min():,.0f}" if len(scored) else "—"
        _b_max = f"${scored['PredictedPrice'].max():,.0f}" if len(scored) else "—"

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(stat_card_html("Scored", str(len(scored)), "checklist", "var(--brand-primary)"), unsafe_allow_html=True)
        with m2:
            st.markdown(stat_card_html("Avg. Price", _b_avg, "payments", "var(--brand-accent)"), unsafe_allow_html=True)
        with m3:
            st.markdown(stat_card_html("Lowest", _b_min, "trending_down", "var(--brand-reject)"), unsafe_allow_html=True)
        with m4:
            st.markdown(stat_card_html("Highest", _b_max, "trending_up", "var(--brand-purple)"), unsafe_allow_html=True)

        st.write("")
        st.dataframe(dark_df(results_df), hide_index=True, use_container_width=True)

        st.download_button(
            "Download results as CSV",
            data=results_df.to_csv(index=False),
            file_name="batch_price_predictions.csv",
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
        f"Evaluated on a held-out test split ({PERFORMANCE['test_size']} houses) "
        f"that the model never saw during training."
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(stat_card_html("RMSE", f"${PERFORMANCE['rmse']:,.0f}", "rule", "var(--brand-primary)"), unsafe_allow_html=True)
    with m2:
        st.markdown(stat_card_html("MAE", f"${PERFORMANCE['mae']:,.0f}", "straighten", "var(--brand-purple)"), unsafe_allow_html=True)
    with m3:
        st.markdown(stat_card_html("R²", f"{PERFORMANCE['r2']:.3f}", "target", "var(--brand-accent)"), unsafe_allow_html=True)
    with m4:
        st.markdown(stat_card_html("vs. Mean Baseline", f"{(1 - PERFORMANCE['rmse']/PERFORMANCE['mean_baseline_rmse']):.0%} lower RMSE", "compare_arrows", "var(--brand-reject)"), unsafe_allow_html=True)

    st.write("")
    st.caption(
        "RMSE and MAE are in dollars — lower is better. R² is the share of price "
        "variance the model explains (1.0 = perfect, 0.0 = no better than always "
        "guessing the average price)."
    )

    st.write("")
    perf_col1, perf_col2 = st.columns([1.2, 1], gap="large")

    with perf_col1:
        st.markdown(
            f"<div class='section-label'>{icon('scatter_plot', 18)} Actual vs. Predicted Price</div>",
            unsafe_allow_html=True
        )
        actual = PERFORMANCE['test_actual']
        pred = PERFORMANCE['test_pred']
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actual, y=pred, mode='markers',
            marker=dict(size=7, opacity=0.7, color='#2FBF8F'),
            name='Test houses'
        ))
        line_range = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
        fig.add_trace(go.Scatter(
            x=line_range, y=line_range, mode='lines',
            line=dict(color='#F2685C', dash='dash'),
            name='Perfect prediction'
        ))
        fig.update_layout(
            xaxis_title="Actual SalePrice ($)",
            yaxis_title="Predicted SalePrice ($)",
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
            "Approach": ["Always predict the mean price", "This model"],
            "RMSE ($)": [
                f"{PERFORMANCE['mean_baseline_rmse']:,.0f}",
                f"{PERFORMANCE['rmse']:,.0f}",
            ],
        })
        st.dataframe(dark_df(baseline_df), hide_index=True, use_container_width=True)
        st.caption(
            "A model that just guesses the average training-set price is the simplest "
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
            xaxis_title="Residual (Actual − Predicted, $)",
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
        dark_df(WEIGHTS_DF, {'Weight': '{:,.0f}'}),
        hide_index=True,
        use_container_width=True
    )

    st.markdown(
        f"<div class='info-card-purple'>{icon('lightbulb', 16, 'var(--brand-purple)')} "
        f"Want to see these coefficients in action? <b>Model Insights</b> lets you drag "
        f"each one and watch the predicted price line respond in real time."
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
        "See how the model's fitted coefficients shape predicted price. "
        "Sliders default to the actual trained model — drag them to explore "
        "what-if scenarios."
    )

    default_feature_idx = INPUTS.index('GrLivArea') if 'GrLivArea' in INPUTS else 0
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
            "Intercept ($ at average house)",
            float(HOUSE_DF[TARGET].min()), float(HOUSE_DF[TARGET].max()),
            st.session_state.get(reset_key_b0, default_intercept), 1000.0,
            key=reset_key_b0
        )
    with col2:
        coef = st.slider(
            f"Coefficient — {feature_choice} ($ per standard deviation)",
            -150000.0, 150000.0,
            st.session_state.get(reset_key_b1, default_coef), 1000.0,
            key=reset_key_b1
        )

    x = np.linspace(-3, 3, 300)  # standardized feature range (~±3 SD)
    y = intercept + coef * x

    stats = FEATURE_STATS[feature_choice]
    x_raw = np.linspace(stats['min'], stats['max'], 300)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_raw, y=y, mode='lines', name='Predicted price',
        line=dict(width=3, color='#2FBF8F')
    ))
    fig.update_layout(
        title=f"Predicted SalePrice vs. {feature_choice}",
        xaxis_title=feature_choice,
        yaxis_title="Predicted SalePrice ($)",
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#EAEFEC'
    )
    st.plotly_chart(fig, use_container_width=True)

    is_default = (abs(intercept - default_intercept) < 1e-6) and (abs(coef - default_coef) < 1e-6)
    st.markdown(
        f"<div class='info-card'>"
        f"{icon('functions', 16)} Current: intercept = <b>${intercept:,.0f}</b>, "
        f"coefficient = <b>${coef:,.0f}</b> "
        f"{'(actual fitted model values)' if is_default else '(what-if, not the fitted model)'}<br><br>"
        f"Steeper line (larger |coefficient|) → {feature_choice} moves the predicted price "
        f"more per unit change. Shifting the intercept moves the whole line up or down."
        f"</div>",
        unsafe_allow_html=True
    )

    with st.expander("Model's actual fitted values for this feature"):
        st.write(f"Fitted intercept: `${default_intercept:,.2f}`")
        st.write(f"Fitted coefficient for {feature_choice}: `${default_coef:,.2f}` per standardized unit")

    st.divider()

    st.markdown(
        f"<div class='section-label' style='font-size:1.05rem;'>"
        f"{icon('bar_chart', 20)} Predicted Price Across {feature_choice}'s Range</div>",
        unsafe_allow_html=True
    )
    st.caption(
        "Holding every other feature at its training-set average, this sweeps "
        f"{feature_choice} across its real observed range and shows the model's "
        "actual predicted price at each point — a direct view of the fitted model, "
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
        sweep_rows.append({feature_choice: val, "Predicted Price": pred_val})

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
        avg_price = f"${log_df['PredictedPrice'].mean():,.0f}" if total else "—"
        min_price = f"${log_df['PredictedPrice'].min():,.0f}" if total else "—"
        max_price = f"${log_df['PredictedPrice'].max():,.0f}" if total else "—"

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(stat_card_html("Total Logged", str(total), "database", "var(--brand-primary)"), unsafe_allow_html=True)
        with m2:
            st.markdown(stat_card_html("Average Price", avg_price, "payments", "var(--brand-accent)"), unsafe_allow_html=True)
        with m3:
            st.markdown(stat_card_html("Lowest", min_price, "trending_down", "var(--brand-reject)"), unsafe_allow_html=True)
        with m4:
            st.markdown(stat_card_html("Highest", max_price, "trending_up", "var(--brand-purple)"), unsafe_allow_html=True)

        st.write("")

        if total >= 2 and 'Timestamp' in log_df.columns:
            trend_df = log_df.copy()
            trend_df['Timestamp'] = pd.to_datetime(trend_df['Timestamp'], errors='coerce')
            trend_df = trend_df.dropna(subset=['Timestamp']).sort_values('Timestamp')

            st.markdown(
                f"<div class='section-label'>{icon('show_chart', 18)} Predicted Price Over Time</div>",
                unsafe_allow_html=True
            )
            st.caption("Predicted price for each logged house, in submission order.")
            st.line_chart(
                trend_df.set_index('Timestamp')[['PredictedPrice']],
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
            file_name="house_price_predictions_log.csv",
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
        f"appraisal tool.</b> Predicted prices shown here should not be used to make "
        f"actual buying, selling, or lending decisions."
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
        "A **linear regression** model trained on `house_prices_practice.csv`, predicting "
        "`SalePrice` from six numeric features: overall quality, above-ground living area, "
        "garage capacity, basement area, year built, and full bathroom count. Features are "
        "standardized (mean 0, standard deviation 1) before fitting so that coefficients "
        "are directly comparable across features. See the **Model Performance** page for "
        "RMSE, MAE, R², and a baseline comparison on a held-out test split."
    )

    st.markdown(
        f"<div class='section-label' style='margin-top:14px;'>{icon('dataset', 18)} "
        f"The Dataset</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"A {len(HOUSE_DF)}-row practice sample of the classic Ames-style house price "
        "dataset, with structural and quality features for each house alongside its "
        "historical sale price. It's a small, fixed dataset — patterns learned from it "
        "reflect that specific sample and won't generalize perfectly to other housing "
        "markets."
    )

    st.markdown(
        f"<div class='section-label' style='margin-top:14px;'>{icon('warning', 18)} "
        f"Limitations</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "- Trained on a small, fixed sample — it will not generalize reliably to other "
        "housing markets, time periods, or house styles outside this dataset's range.\n"
        "- Only six numeric features are used; real appraisals weigh location, condition, "
        "renovations, and dozens of other factors this model never sees.\n"
        "- Extrapolating far outside the training data's feature ranges (see **Model "
        "Insights**) produces unreliable estimates, since linear regression has no "
        "guardrails against unrealistic inputs.\n"
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
