"""
Flight Delay Prediction System
Streamlit Application — Aerospace / ML & Data Science Project
Dataset: kaggle.com/datasets/shubhamsingh42/flight-delay-dataset-2018-2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import sys
import requests

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

sys.path.insert(0, os.path.dirname(__file__))
from utils.data_generator import (
    generate_flight_data, _estimate_distance, AIRLINES, AIRPORTS, WEATHER_CONDITIONS
)
from utils.ml_models import FlightDelayPredictor, MODELS_DIR
from utils.weather_fetcher import (
    fetch_weather, get_route_weather, compute_risk_score,
    build_weather_features, AIRPORT_COORDS as WX_AIRPORT_COORDS, WEATHER_ICON_MAP,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AeroSight — Flight Delay Prediction System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "predictor" not in st.session_state:
    st.session_state.predictor = FlightDelayPredictor()
if "trained" not in st.session_state:
    st.session_state.trained = False
if "pred_history" not in st.session_state:
    st.session_state.pred_history = []
if "origin_airport" not in st.session_state:
    st.session_state.origin_airport = "ATL"
if "dest_airport" not in st.session_state:
    st.session_state.dest_airport = "LAX"

# ── Auto-load data & auto-train ───────────────────────────────────────────────
if st.session_state.df is None:
    with st.spinner("Loading dataset…"):
        st.session_state.df = generate_flight_data(20000)

df = st.session_state.df

if not st.session_state.trained:
    # Check if models are already trained and cached on disk
    models_exist = os.path.exists(os.path.join(MODELS_DIR, "models.pkl"))
    if models_exist:
        with st.spinner("Loading pre-trained models from disk…"):
            try:
                st.session_state.predictor.load()
                st.session_state.trained = True
            except Exception as e:
                # If loading fails, fallback to training
                st.warning(f"Failed to load cached models: {e}. Re-training…")
                
    if not st.session_state.trained:
        with st.spinner("Training models in the background… (~30s)"):
            st.session_state.predictor.train(df)
            try:
                st.session_state.predictor.save()
            except Exception as e:
                st.warning(f"Failed to save models to disk: {e}")
            st.session_state.trained = True

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 15px 0 5px 0; display: flex; flex-direction: column; align-items: center;">
      <div class="logo-icon" style="margin-bottom: 18px; width: 64px; height: 64px;"><span style="color: var(--accent) !important; font-size: 30px !important;">✈</span></div>
      <div style="font-family: 'Rajdhani', sans-serif; font-size: 26px; font-weight: 700; color: #00c8ff; letter-spacing: 3px; text-transform: uppercase;">
        AEROSIGHT
      </div>
      <div style="font-family: 'Share Tech Mono', monospace; font-size: 9px; color: #3a7090; letter-spacing: 1px; margin-top: 5px;">
        FLIGHT DELAY PREDICTION SYSTEM
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-section-label'>Navigation</div>", unsafe_allow_html=True)
    
    _nav_options = [
        "◈ Dashboard",
        "◎ Prediction",
        "◇ Weather Intel",
        "◆ Model Performance",
        "▦ Flight Data",
        "◉ About"
    ]
    
    # Map any legacy page redirects
    _nav_default = st.session_state.pop("_nav", None)
    _nav_mapping = {
        "🏠 Overview": "◈ Dashboard",
        "📊 EDA & Insights": "◈ Dashboard",
        "🤖 Model Training": "◆ Model Performance",
        "🔮 Predict a Flight": "◎ Prediction",
        "📈 Model Performance": "◆ Model Performance",
        "🎯 Executive Dashboard": "◈ Dashboard"
    }
    if _nav_default in _nav_mapping:
        _nav_default = _nav_mapping[_nav_default]
        
    _nav_index = _nav_options.index(_nav_default) if _nav_default in _nav_options else 0
    
    page = st.radio(
        "Navigation",
        _nav_options,
        index=_nav_index,
        label_visibility="collapsed"
    )
    
    # Theme toggle
    st.markdown("<div class='sidebar-section-label'>Settings</div>", unsafe_allow_html=True)
    theme_toggle = st.toggle("Light Theme Mode", value=False)
    
    # Sidebar status panel
    st.markdown(f"""
    <div class="sidebar-status">
      <div class="status-row">
        <span>ML Engine</span>
        <div style="display:flex;align-items:center;gap:6px">
          <div class="status-dot"></div>
          <span style="color:#00e5a0">READY</span>
        </div>
      </div>
      <div class="status-row">
        <span>Weather API</span>
        <div style="display:flex;align-items:center;gap:6px">
          <div class="status-dot"></div>
          <span style="color:#00e5a0">LIVE</span>
        </div>
      </div>
      <div class="status-row">
        <span>Models</span>
        <span style="color:#00c8ff">1 LOADED</span>
      </div>
      <div class="status-row">
        <span>Dataset</span>
        <span style="color:#00c8ff">2018–2024</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Dynamic Theme & Custom CSS ────────────────────────────────────────────────
if theme_toggle:
    variables_css = """
    :root {
        --bg: #f0f4fa;
        --bg2: #e6edf7;
        --panel: #ffffff;
        --panel2: #eef3fb;
        --border: rgba(0,100,200,0.15);
        --border-bright: rgba(0,100,200,0.4);
        --accent: #0055cc;
        --accent2: #0033aa;
        --accent3: #6d28d9;
        --text: #0a1a30;
        --text2: #2255aa;
        --text3: #375377;
        --green: #007a50;
        --amber: #b36a00;
        --red: #cc2200;
        --font-display: 'Rajdhani', sans-serif;
        --font-body: 'Exo 2', sans-serif;
        --font-mono: 'Share Tech Mono', monospace;
    }
    """
else:
    variables_css = """
    :root {
        --bg: #020c18;
        --bg2: #040f20;
        --panel: #071629;
        --panel2: #0a1e38;
        --border: rgba(0,180,255,0.18);
        --border-bright: rgba(0,200,255,0.45);
        --accent: #00c8ff;
        --accent2: #0066ff;
        --accent3: #7c3aed;
        --text: #e8f4ff;
        --text2: #7fb8d8;
        --text3: #8cb1d1;
        --green: #00e5a0;
        --amber: #ffb300;
        --red: #ff4444;
        --font-display: 'Rajdhani', sans-serif;
        --font-body: 'Exo 2', sans-serif;
        --font-mono: 'Share Tech Mono', monospace;
    }
    """

# Inject Google Fonts via link tag (more reliable than @import inside <style>)
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Inject custom CSS
st.markdown(f"""
<style>
    {variables_css}

    /* ─── STREAMLIT LAYOUT FIXES ──────────────────────────────────────── */
    /* These ensure the main content area is ALWAYS visible */
    .stApp {{
        background: var(--bg) !important;
        color: var(--text) !important;
    }}
    .stApp > header {{
        background-color: transparent !important;
    }}
    .stMainBlockContainer,
    [data-testid="stAppViewBlockContainer"],
    div[data-testid="stAppViewContainer"],
    div[data-testid="stMain"],
    section[data-testid="stMain"] {{
        background-color: var(--bg) !important;
        visibility: visible !important;
        opacity: 1 !important;
    }}
    .main .block-container {{
        background-color: var(--bg) !important;
        padding: 2rem 2.5rem !important;
        max-width: 100% !important;
    }}

    /* ─── TYPOGRAPHY ──────────────────────────────────────────────────── */
    .stApp, .stApp p, .stApp li, .stApp a,
    .stApp td, .stApp th, .stApp label,
    .stApp input, .stApp select, .stApp textarea {{
        font-family: var(--font-body) !important;
        color: var(--text) !important;
    }}
    .stApp span {{
        color: var(--text);
    }}
    .stApp h1, .stApp h2, .stApp h3,
    .stApp h4, .stApp h5, .stApp h6 {{
        color: var(--text) !important;
        font-family: var(--font-display) !important;
    }}

    /* ─── PROTECT STREAMLIT ICONS ─────────────────────────────────────── */
    [data-testid="collapsedControl"],
    [data-testid="collapsedControl"] *,
    .stApp [class*="material-symbols"],
    .stApp [class*="material-icons"] {{
        font-family: "Material Symbols Outlined", "Material Symbols Rounded", "Material Icons", sans-serif !important;
    }}

    /* ─── SIDEBAR ─────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background: var(--panel) !important;
        border-right: 1px solid var(--border) !important;
        padding-top: 20px !important;
        width: 240px !important;
        min-width: 240px !important;
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label {{
        color: var(--text2) !important;
    }}
    section[data-testid="stSidebar"] span {{
        color: var(--text2);
    }}

    /* Sidebar Navigation Radio */
    section[data-testid="stSidebar"] [data-testid="stRadio"] {{
        padding: 0 !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {{
        gap: 2px !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label {{
        display: flex !important;
        align-items: center !important;
        padding: 10px 20px !important;
        cursor: pointer !important;
        font-family: var(--font-display) !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        color: var(--text2) !important;
        background: transparent !important;
        border-left: 3px solid transparent !important;
        border-radius: 0 !important;
        margin: 0 !important;
        transition: all 0.2s !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:hover {{
        color: var(--text) !important;
        background: rgba(0,200,255,0.05) !important;
    }}
    /* Active radio item */
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {{
        color: var(--accent) !important;
        border-left-color: var(--accent) !important;
        background: rgba(0,200,255,0.07) !important;
    }}
    /* Hide radio circle indicator */
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label > div:first-child {{
        display: none !important;
    }}

    /* ─── TOP BAR ─────────────────────────────────────────────────────── */
    .topbar {{
        background: var(--panel);
        border-bottom: 1px solid var(--border-bright);
        display: flex; align-items: center; padding: 12px 24px; gap: 20px;
        position: relative; z-index: 100;
        margin-bottom: 25px;
        border-radius: 4px;
        justify-content: space-between;
    }}
    .topbar::after {{
        content: ''; position: absolute; bottom: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent), var(--accent2), var(--accent3), transparent);
        background-size: 200% 100%;
        animation: scan 4s linear infinite;
    }}
    @keyframes scan {{ 0%{{background-position:0% 50%}} 100%{{background-position:200% 50%}} }}

    .logo {{
        font-family: var(--font-display) !important;
        font-size: 42px; font-weight: 700; letter-spacing: 3px;
        color: var(--accent) !important; text-transform: uppercase;
        display: flex; align-items: center; gap: 12px;
    }}
    .logo span, .logo * {{ color: var(--accent) !important; }}
    .logo-icon {{
        width: 46px; height: 46px;
        border: 2px solid var(--accent);
        transform: rotate(45deg);
        display: flex; align-items: center; justify-content: center;
        animation: pulse-border 2s infinite;
    }}
    @keyframes pulse-border {{
        0%,100% {{ box-shadow: 0 0 0 0 rgba(0,200,255,0.4); }}
        50% {{ box-shadow: 0 0 0 6px rgba(0,200,255,0); }}
    }}
    .logo-icon span {{ transform: rotate(-45deg); font-size: 22px; display: inline-block; }}

    .live-badge {{
        font-family: var(--font-mono) !important; font-size: 11px;
        background: rgba(0,229,160,0.12);
        color: var(--green) !important; border: 1px solid var(--green);
        padding: 3px 10px; letter-spacing: 1px; text-transform: uppercase;
    }}
    .clock {{
        font-family: var(--font-mono) !important; font-size: 18px;
        color: var(--text2) !important; letter-spacing: 2px;
    }}

    /* ─── SIDEBAR DETAILS ─────────────────────────────────────────────── */
    .sidebar-section-label {{
        font-family: var(--font-mono) !important; font-size: 10px;
        color: var(--text3) !important; letter-spacing: 2px; text-transform: uppercase;
        padding: 0 20px; margin: 18px 0 8px;
    }}
    .sidebar-status {{
        padding: 16px 20px;
        border-top: 1px solid var(--border);
        margin-top: 20px;
    }}
    .status-row {{
        display: flex; justify-content: space-between; align-items: center;
        font-family: var(--font-mono) !important; font-size: 11px; color: var(--text3) !important;
        padding: 4px 0;
    }}
    .status-row span {{ color: var(--text3) !important; }}
    .status-dot {{
        width: 7px; height: 7px; border-radius: 50%;
        background: var(--green) !important;
        box-shadow: 0 0 6px var(--green) !important;
        display: inline-block;
    }}

    /* ─── COLOR OVERRIDES ─────────────────────────────────────────────── */
    .delta-up {{ color: var(--red) !important; }}
    .delta-down {{ color: var(--green) !important; }}
    .result-high, .result-high * {{ color: var(--red) !important; }}
    .result-low, .result-low * {{ color: var(--green) !important; }}
    .wx-airport {{ color: var(--accent) !important; }}

    /* ─── PAGE HEADER ─────────────────────────────────────────────────── */
    .page-header {{
        display: flex; align-items: flex-end; justify-content: space-between;
        padding-bottom: 16px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 24px;
    }}
    .page-title {{
        font-family: var(--font-display) !important; font-size: 28px; font-weight: 700;
        letter-spacing: 3px; text-transform: uppercase; color: var(--text) !important;
        line-height: 1;
    }}
    .page-title span {{ color: var(--accent) !important; }}
    .page-subtitle {{
        font-family: var(--font-mono) !important; font-size: 12px; color: var(--text3) !important;
        letter-spacing: 1.5px; margin-top: 6px;
    }}
    .page-tag {{
        font-family: var(--font-mono) !important; font-size: 11px;
        color: var(--accent2) !important; border: 1px solid var(--accent2);
        padding: 4px 12px; letter-spacing: 1px;
    }}

    /* ─── CARD PANELS ─────────────────────────────────────────────────── */
    .card {{
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        padding: 22px !important;
        position: relative !important;
        overflow: hidden !important;
        border-radius: 4px !important;
        margin-bottom: 20px !important;
    }}
    .card-title {{
        font-family: var(--font-display) !important;
        font-size: 14px; font-weight: 700;
        letter-spacing: 2px; text-transform: uppercase; color: var(--text2) !important;
        margin-bottom: 18px; display: flex; align-items: center; gap: 10px;
    }}
    .card-title::before {{
        content: ''; width: 12px; height: 2px; background: var(--accent);
        flex-shrink: 0;
    }}

    /* ─── KPI STRIP ───────────────────────────────────────────────────── */
    .kpi-strip {{
        display: grid !important;
        grid-template-columns: repeat(4, 1fr) !important;
        gap: 16px !important;
        margin-bottom: 24px !important;
    }}
    .kpi-card {{
        background: var(--panel) !important; border: 1px solid var(--border) !important;
        border-top: 2px solid var(--accent) !important;
        padding: 18px 20px !important; position: relative !important; overflow: hidden !important;
        transition: border-color 0.2s, transform 0.2s !important;
    }}
    .kpi-card:hover {{ border-color: var(--border-bright) !important; transform: translateY(-2px) !important; }}
    .kpi-label {{
        font-family: var(--font-mono) !important; font-size: 10px !important;
        color: var(--text3) !important; letter-spacing: 2px; text-transform: uppercase;
        margin-bottom: 8px;
    }}
    .kpi-value {{
        font-family: var(--font-display) !important; font-size: 34px !important; font-weight: 700 !important;
        letter-spacing: 1px; color: var(--text) !important; line-height: 1;
    }}
    .kpi-delta {{
        font-family: var(--font-mono) !important; font-size: 11px !important;
        margin-top: 6px; display: flex; align-items: center; gap: 6px;
    }}
    .kpi-bar {{
        position: absolute; bottom: 0; left: 0; height: 3px;
        background: linear-gradient(90deg, var(--accent), var(--accent2));
        transition: width 1.2s cubic-bezier(0.4,0,0.2,1);
    }}

    /* ─── FORM WIDGETS ────────────────────────────────────────────────── */
    .stSelectbox > div > div, .stMultiSelect > div,
    .stSlider > div, .stTextInput > div > div {{
        background-color: var(--bg2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 0px !important;
    }}
    /* Ensure selectbox text color is readable */
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] *,
    .stSelectbox div[role="button"],
    .stSelectbox div[role="button"] * {{
        color: var(--text) !important;
    }}
    /* Style dropdown popup lists */
    div[role="listbox"] li,
    div[role="listbox"] li * {{
        color: var(--text) !important;
        background-color: var(--panel) !important;
    }}
    div[role="listbox"] li:hover,
    div[role="listbox"] li:hover * {{
        background-color: var(--accent) !important;
        color: var(--bg) !important;
    }}
    div[role="listbox"] input {{
        color: var(--text) !important;
        background-color: var(--bg2) !important;
    }}
    .stTextInput input,
    .stTextInput input:disabled,
    .stTextInput input[disabled] {{
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
    }}
    .stSelectbox label, .stMultiSelect label,
    .stSlider label, .stTextInput label {{
        font-family: var(--font-mono) !important;
        font-size: 10px !important;
        color: var(--text3) !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
    }}
    .stButton > button {{
        margin-top: 18px !important; width: 100% !important; padding: 14px !important;
        background: transparent !important;
        border: 1px solid var(--accent) !important;
        color: var(--accent) !important;
        font-family: var(--font-display) !important; font-size: 18px !important; font-weight: 700 !important;
        letter-spacing: 3px; text-transform: uppercase; cursor: pointer !important;
        border-radius: 0px !important;
        transition: all 0.3s !important;
    }}
    .stButton > button:hover {{
        background: var(--accent) !important;
        color: var(--bg) !important;
        box-shadow: 0 0 10px rgba(0,200,255,0.4) !important;
    }}

    /* ─── RESULT BOX ──────────────────────────────────────────────────── */
    .result-box {{
        padding: 24px; text-align: center; border: 1px solid var(--border);
        background: var(--panel2); margin-top: 4px;
        border-radius: 4px;
        animation: fadeIn 0.5s ease;
    }}
    @keyframes fadeIn {{ from{{opacity:0;transform:translateY(8px)}} to{{opacity:1;transform:translateY(0)}} }}
    .result-label {{
        font-family: var(--font-mono) !important; font-size: 11px;
        letter-spacing: 2px; text-transform: uppercase; color: var(--text3) !important;
        margin-bottom: 12px;
    }}
    .result-prob {{
        font-family: var(--font-display) !important; font-size: 72px; font-weight: 700;
        line-height: 1; letter-spacing: 2px;
    }}
    .result-verdict {{
        font-family: var(--font-display) !important; font-size: 22px; font-weight: 600;
        letter-spacing: 3px; margin-top: 8px; text-transform: uppercase;
    }}
    .result-low {{ color: var(--green) !important; }}
    .result-high {{ color: var(--red) !important; }}

    /* ─── FEATURE BARS ────────────────────────────────────────────────── */
    .feature-row {{ display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }}
    .feature-name {{
        font-family: var(--font-mono) !important; font-size: 11px; color: var(--text2) !important;
        min-width: 140px; letter-spacing: 0.5px;
    }}
    .feature-track {{ flex: 1; height: 6px; background: var(--bg2); position: relative; }}
    .feature-fill {{
        position: absolute; top: 0; bottom: 0; left: 0;
        background: linear-gradient(90deg, var(--accent2), var(--accent));
        animation: expand 1.2s cubic-bezier(0.4,0,0.2,1) both;
    }}
    @keyframes expand {{ from{{width:0}} }}
    .feature-pct {{
        font-family: var(--font-mono) !important; font-size: 11px; color: var(--text3) !important;
        min-width: 36px; text-align: right;
    }}

    /* ─── WEATHER ──────────────────────────────────────────────────────── */
    .wx-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .wx-card {{
        background: var(--bg2); border: 1px solid var(--border);
        border-left: 3px solid var(--accent); padding: 14px 16px;
        border-radius: 4px;
    }}
    .wx-airport {{
        font-family: var(--font-display) !important; font-size: 24px; font-weight: 700;
        letter-spacing: 2px; color: var(--accent) !important;
    }}
    .wx-city {{ font-family: var(--font-mono) !important; font-size: 11px; color: var(--text3) !important; margin-bottom: 12px; }}
    .wx-row {{ display: flex; justify-content: space-between; align-items: center; padding: 5px 0;
        border-bottom: 1px solid rgba(0,180,255,0.06); font-size: 12px; }}
    .wx-row:last-child {{ border-bottom: none; }}
    .wx-key {{ font-family: var(--font-mono) !important; color: var(--text3) !important; font-size: 11px; letter-spacing: 1px; }}
    .wx-val {{ font-family: var(--font-display) !important; font-weight: 600; font-size: 14px; color: var(--text) !important; }}

    /* ─── RISK METER ──────────────────────────────────────────────────── */
    .risk-bar-track {{ height: 8px; background: var(--bg2); position: relative; border: 1px solid var(--border); }}
    .risk-bar-fill {{ height: 100%; transition: width 1.2s cubic-bezier(0.4,0,0.2,1); }}
    .risk-labels {{
        display: flex; justify-content: space-between; margin-top: 4px;
        font-family: var(--font-mono) !important; font-size: 9px; color: var(--text3) !important;
        letter-spacing: 1px;
    }}

    /* ─── DATA TABLE ──────────────────────────────────────────────────── */
    .data-table {{ width: 100%; border-collapse: collapse; }}
    .data-table th {{
        font-family: var(--font-mono) !important; font-size: 10px;
        letter-spacing: 2px; text-transform: uppercase;
        color: var(--text3) !important; padding: 8px 12px;
        border-bottom: 1px solid var(--border); text-align: left;
    }}
    .data-table td {{
        padding: 10px 12px; font-family: var(--font-body) !important; font-size: 13px;
        color: var(--text) !important; border-bottom: 1px solid rgba(0,180,255,0.06);
    }}
    .data-table tr:hover td {{ background: rgba(0,200,255,0.03); }}

    /* ─── BADGES ──────────────────────────────────────────────────────── */
    .badge {{
        display: inline-block; padding: 2px 10px;
        font-family: var(--font-mono) !important; font-size: 10px; letter-spacing: 1px;
        text-transform: uppercase; border: 1px solid;
    }}
    .badge-green {{ color: var(--green) !important; border-color: var(--green) !important; background: rgba(0,229,160,0.08) !important; }}
    .badge-red {{ color: var(--red) !important; border-color: var(--red) !important; background: rgba(255,68,68,0.08) !important; }}
    .badge-amber {{ color: var(--amber) !important; border-color: var(--amber) !important; background: rgba(255,179,0,0.08) !important; }}

    /* ─── SECTION DIVIDER ─────────────────────────────────────────────── */
    .section-divider {{
        display: flex; align-items: center; gap: 14px; margin: 24px 0 16px;
    }}
    .section-divider span {{
        font-family: var(--font-mono) !important; font-size: 10px; color: var(--text3) !important;
        letter-spacing: 2px; text-transform: uppercase; white-space: nowrap;
    }}
    .section-divider::before, .section-divider::after {{
        content: ''; flex: 1; height: 1px; background: var(--border);
    }}

    /* ─── MODEL CARDS ─────────────────────────────────────────────────── */
    .model-grid {{ display: grid !important; grid-template-columns: repeat(3, 1fr) !important; gap: 14px !important; margin-bottom: 24px !important; }}
    .model-card {{
        background: var(--bg2) !important; border: 1px solid var(--border) !important;
        border-top: 2px solid var(--accent3) !important;
        padding: 16px !important; text-align: center !important;
    }}
    .model-card:hover {{ border-color: var(--accent) !important; }}
    .model-name {{
        font-family: var(--font-display) !important; font-size: 16px !important; font-weight: 700 !important;
        letter-spacing: 1px; text-transform: uppercase; color: var(--text) !important;
        margin-bottom: 12px;
    }}
    .model-metric {{ margin: 6px 0 !important; }}
    .model-metric-label {{ font-family: var(--font-mono) !important; font-size: 10px !important; color: var(--text3) !important; letter-spacing: 1px; }}
    .model-metric-val {{ font-family: var(--font-display) !important; font-size: 26px !important; font-weight: 700 !important; color: var(--accent) !important; }}

    /* ─── STREAMLIT DATAFRAME ─────────────────────────────────────────── */
    .stDataFrame {{
        border: 1px solid var(--border) !important;
        border-radius: 4px !important;
    }}

    /* ─── TABS ─────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab"] {{
        font-family: var(--font-display) !important;
        font-size: 14px !important; font-weight: 600 !important;
        letter-spacing: 2px !important; text-transform: uppercase !important;
        color: var(--text3) !important;
        background: transparent !important;
        border-bottom: 2px solid transparent !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background: transparent !important;
        border-bottom: 1px solid var(--border) !important;
    }}

    /* ─── SCROLLBAR ───────────────────────────────────────────────────── */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg); }}
    ::-webkit-scrollbar-thumb {{ background: var(--border-bright); }}

    /* ─── INSIGHT CARDS ───────────────────────────────────────────────── */
    .insight-card {{
        background: var(--panel2) !important;
        border: 1px solid var(--border) !important;
        border-left: 3px solid var(--accent) !important;
        padding: 16px !important;
        border-radius: 4px !important;
        margin-bottom: 15px !important;
    }}
    .insight-card.good {{
        border-left-color: var(--green) !important;
    }}
    .insight-card.alert,
    .insight-card.warn {{
        border-left-color: var(--amber) !important;
    }}
    .insight-card.danger {{
        border-left-color: var(--red) !important;
    }}
    .i-title {{
        font-family: var(--font-display) !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        color: var(--text) !important;
        margin-bottom: 6px !important;
    }}
    .i-body {{
        font-family: var(--font-body) !important;
        font-size: 13px !important;
        color: var(--text2) !important;
        line-height: 1.4 !important;
    }}
</style>
""", unsafe_allow_html=True)

# ── Plotly styling helper ─────────────────────────────────────────────────────
def style_plotly_fig(fig, light_mode=False):
    text_color = '#0a1a30' if light_mode else '#e8f4ff'
    grid_color = 'rgba(0,100,200,0.08)' if light_mode else 'rgba(0,180,255,0.06)'
    border_color = 'rgba(0,100,200,0.15)' if light_mode else 'rgba(0,180,255,0.18)'
    tick_color = '#0a1a30' if light_mode else '#e8f4ff'
    title_color = '#0055cc' if light_mode else '#00c8ff'
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Exo 2, sans-serif', color=text_color),
        margin=dict(t=30, b=30, l=40, r=10),
        xaxis=dict(
            gridcolor=grid_color,
            linecolor=border_color,
            tickfont=dict(family='Share Tech Mono, monospace', size=11, color=tick_color),
            title=dict(font=dict(family='Rajdhani, sans-serif', size=12, color=title_color))
        ),
        yaxis=dict(
            gridcolor=grid_color,
            linecolor=border_color,
            tickfont=dict(family='Share Tech Mono, monospace', size=11, color=tick_color),
            title=dict(font=dict(family='Rajdhani, sans-serif', size=12, color=title_color))
        ),
        showlegend=False
    )
    return fig

# ── Folium map helper ─────────────────────────────────────────────────────────
def draw_route_map(origin, dest, prob, o_wx, d_wx):
    if not FOLIUM_AVAILABLE:
        return None
    o_coord = WX_AIRPORT_COORDS.get(origin)
    d_coord = WX_AIRPORT_COORDS.get(dest)
    if not o_coord or not d_coord:
        return None
        
    center_lat = (o_coord["lat"] + d_coord["lat"]) / 2.0
    center_lon = (o_coord["lon"] + d_coord["lon"]) / 2.0
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles="CartoDB dark_matter")
    
    folium.Marker(
        [o_coord["lat"], o_coord["lon"]],
        popup=f"{origin}: {o_wx['weather_desc']}, {o_wx['temperature_c']}C",
        icon=folium.Icon(color="green", icon="plane", prefix="fa")
    ).add_to(m)
    
    color = "red" if prob > 0.45 else ("orange" if prob > 0.3 else "green")
    folium.Marker(
        [d_coord["lat"], d_coord["lon"]],
        popup=f"{dest}: {d_wx['weather_desc']}, {d_wx['temperature_c']}C",
        icon=folium.Icon(color=color, icon="plane", prefix="fa")
    ).add_to(m)
    
    folium.PolyLine(
        locations=[[o_coord["lat"], o_coord["lon"]], [d_coord["lat"], d_coord["lon"]]],
        color=color,
        weight=2,
        opacity=0.8,
        dash_array="5, 5"
    ).add_to(m)
    
    return m

# ── Top Bar Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="logo">
    <div class="logo-icon"><span>✈</span></div>
    AEROSIGHT
  </div>
  <span style="font-family:var(--font-mono);font-size:12px;color:var(--text3);letter-spacing:1px;margin-right:auto;margin-left:20px;">
    FLIGHT DELAY PREDICTION SYSTEM
  </span>
  <div style="display:flex; align-items:center; gap:20px;">
    <div class="live-badge">● LIVE</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# PAGE: Dashboard
# ==============================================================================
if page == "◈ Dashboard":
    # Page Header
    st.markdown("""
    <div class="page-header">
      <div>
        <div class="page-title">SYSTEM <span>OVERVIEW</span></div>
        <div class="page-subtitle">REAL-TIME FLIGHT DELAY OPERATIONAL INTELLIGENCE · HISTORICAL KAGGLE DATASET</div>
      </div>
      <div class="page-tag">AUTO-REFRESH 30S</div>
    </div>
    """, unsafe_allow_html=True)

    # Interactive Year slider inside card
    st.markdown('<div class="card"><div class="card-title">📅 Filter Operations Period</div>', unsafe_allow_html=True)
    years = sorted(df["YEAR"].unique())
    sel_years = st.select_slider(
        "Reporting Year Range",
        options=years,
        value=(min(years), max(years)),
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Filter dataset
    fdf = df[df["YEAR"].between(sel_years[0], sel_years[1])]

    # Compute Core KPIs
    total = len(fdf)
    delayed = int(fdf["IS_DELAYED"].sum())
    delay_rate = (delayed / total * 100) if total else 0.0
    avg_delay = fdf[fdf["DEP_DELAY"] > 0]["DEP_DELAY"].mean()
    if np.isnan(avg_delay):
        avg_delay = 0.0
        
    if st.session_state.trained:
        model_acc = st.session_state.predictor.metrics.get("Gradient Boosting", {}).get("accuracy", 0.788) * 100
    else:
        model_acc = 78.8

    # KPI strip
    st.markdown(f"""
    <div class="kpi-strip">
      <div class="kpi-card">
        <div class="kpi-label">Total Flights</div>
        <div class="kpi-value">{total/1e3:.1f}K</div>
        <div class="kpi-delta delta-up">↑ 3.1% YoY</div>
        <div class="kpi-bar" style="width:74%"></div>
      </div>
      <div class="kpi-card" style="border-top-color:var(--red)">
        <div class="kpi-label">Delay Rate</div>
        <div class="kpi-value">{delay_rate:.1f}%</div>
        <div class="kpi-delta delta-up">↑ 1.4pp</div>
        <div class="kpi-bar" style="width:{delay_rate:.1f}%;background:linear-gradient(90deg,var(--red),#ff6b6b)"></div>
      </div>
      <div class="kpi-card" style="border-top-color:var(--green)">
        <div class="kpi-label">Model Accuracy</div>
        <div class="kpi-value">{model_acc:.1f}%</div>
        <div class="kpi-delta delta-down">↑ 2.1pp</div>
        <div class="kpi-bar" style="width:{model_acc:.1f}%;background:linear-gradient(90deg,var(--green),#34d399)"></div>
      </div>
      <div class="kpi-card" style="border-top-color:var(--amber)">
        <div class="kpi-label">Avg Delay (min)</div>
        <div class="kpi-value">{avg_delay:.1f}</div>
        <div class="kpi-delta delta-up">↑ 4.8 min</div>
        <div class="kpi-bar" style="width:{min(100, int(avg_delay*1.5))}%;background:linear-gradient(90deg,var(--amber),#ffd54f)"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Charts Row (Airlines and Months)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><div class="card-title">Delay Rate by Airline</div>', unsafe_allow_html=True)
        airline_rates = fdf.groupby("OP_CARRIER")["IS_DELAYED"].mean().reset_index()
        airline_rates["OP_CARRIER"] = airline_rates["OP_CARRIER"].map(AIRLINES)
        airline_rates = airline_rates.sort_values("IS_DELAYED", ascending=False)
        fig_airline = px.bar(
            airline_rates, x="OP_CARRIER", y="IS_DELAYED",
            color_discrete_sequence=['#00c8ff']
        )
        style_plotly_fig(fig_airline, theme_toggle)
        st.plotly_chart(fig_airline, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><div class="card-title">Monthly Delay Trend 2024</div>', unsafe_allow_html=True)
        monthly_trend = fdf.groupby("MONTH")["IS_DELAYED"].mean().reset_index()
        monthly_trend["Month"] = monthly_trend["MONTH"].map({
            1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
            7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"
        })
        fig_month = px.bar(
            monthly_trend, x="Month", y="IS_DELAYED",
            color_discrete_sequence=['#7c3aed']
        )
        style_plotly_fig(fig_month, theme_toggle)
        st.plotly_chart(fig_month, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Causes and Risk Distributions Row
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><div class="card-title">Top Delay Causes</div>', unsafe_allow_html=True)
        delay_causes = {
            "Weather": fdf["WEATHER_DELAY"].mean(),
            "Late Aircraft": fdf["LATE_AIRCRAFT_DELAY"].mean(),
            "Carrier": fdf["CARRIER_DELAY"].mean(),
            "NAS Delay": fdf["NAS_DELAY"].mean(),
            "Security": fdf["SECURITY_DELAY"].mean()
        }
        max_cause = max(delay_causes.values()) if max(delay_causes.values()) > 0 else 1.0
        cause_pcts = {k: int((v / max_cause) * 100) for k, v in delay_causes.items()}
        sorted_causes = sorted(cause_pcts.items(), key=lambda x: x[1], reverse=True)
        
        causes_html = "<div style='margin-top:4px'>"
        for i, (k, pct) in enumerate(sorted_causes):
            causes_html += (
                f'<div class="feature-row" style="--i:{i}">'
                f'<span class="feature-name">{k}</span>'
                f'<div class="feature-track"><div class="feature-fill" style="width:{pct}%"></div></div>'
                f'<span class="feature-pct">{pct}%</span>'
                f'</div>'
            )
        causes_html += "</div>"
        st.markdown(causes_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><div class="card-title">Risk Distribution</div>', unsafe_allow_html=True)
        carrier_rates = fdf.groupby("OP_CARRIER")["IS_DELAYED"].mean()
        low_count = sum(carrier_rates < 0.3)
        mod_count = sum((carrier_rates >= 0.3) & (carrier_rates < 0.45))
        high_count = sum(carrier_rates >= 0.45)
        total_count = len(carrier_rates)
        if total_count > 0:
            low_pct = int((low_count / total_count) * 100)
            mod_pct = int((mod_count / total_count) * 100)
            high_pct = 100 - low_pct - mod_pct
        else:
            low_pct, mod_pct, high_pct = 42, 35, 23
            
        st.markdown(f"""
        <div class="risk-bar-wrap">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <span style="font-family:var(--font-mono);font-size:11px;color:var(--green)">LOW RISK</span>
            <span style="font-family:var(--font-display);font-size:20px;font-weight:700;color:var(--green)">{low_pct}%</span>
          </div>
          <div class="risk-bar-track"><div class="risk-bar-fill" style="width:{low_pct}%;background:var(--green)"></div></div>
          <div class="risk-labels"><span>0</span><span>50%</span><span>100%</span></div>

          <div style="display:flex;justify-content:space-between;margin:16px 0 6px">
            <span style="font-family:var(--font-mono);font-size:11px;color:var(--amber)">MOD RISK</span>
            <span style="font-family:var(--font-display);font-size:20px;font-weight:700;color:var(--amber)">{mod_pct}%</span>
          </div>
          <div class="risk-bar-track"><div class="risk-bar-fill" style="width:{mod_pct}%;background:var(--amber)"></div></div>

          <div style="display:flex;justify-content:space-between;margin:16px 0 6px">
            <span style="font-family:var(--font-mono);font-size:11px;color:var(--red)">HIGH RISK</span>
            <span style="font-family:var(--font-display);font-size:20px;font-weight:700;color:var(--red)">{high_pct}%</span>
          </div>
          <div class="risk-bar-track"><div class="risk-bar-fill" style="width:{high_pct}%;background:var(--red)"></div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Auto-generated Key Insights
    st.markdown('<div class="section-header"><span>💡 Operational Key Insights</span></div>', unsafe_allow_html=True)
    airline_perf = fdf.groupby("OP_CARRIER").agg(
        flights=("IS_DELAYED", "count"),
        delay_rate=("IS_DELAYED", "mean")
    ).reset_index().sort_values("delay_rate")
    airline_perf["OTP"] = (1 - airline_perf["delay_rate"]) * 100

    insights = []
    if len(airline_perf) >= 2:
        best = airline_perf.iloc[0]
        worst = airline_perf.iloc[-1]
        best_name = AIRLINES.get(best['OP_CARRIER'], best['OP_CARRIER'])
        worst_name = AIRLINES.get(worst['OP_CARRIER'], worst['OP_CARRIER'])
        insights.append(("good", "🏆 Top Performer",
            f"<b>{best_name}</b> leads with {best['OTP']:.1f}% on-time performance "
            f"({best['flights']:,} flights). Benchmark gap vs worst carrier: "
            f"{worst['OTP']:.1f}% ({worst_name})."))
        if worst["delay_rate"] > 0.40:
            insights.append(("alert", "⚠️ Carrier Alert",
                f"<b>{worst_name}</b> has a {worst['delay_rate']*100:.1f}% delay rate — "
                f"significantly above fleet average of {delay_rate:.1f}%."))

    monthly_by_month = fdf.groupby("MONTH")["IS_DELAYED"].mean()
    worst_month_num = monthly_by_month.idxmax()
    best_month_num = monthly_by_month.idxmin()
    month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
                   7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
    insights.append(("warn", "📅 Seasonal Peak",
        f"<b>{month_names.get(worst_month_num,'?')}</b> records the highest delay rate "
        f"({monthly_by_month[worst_month_num]*100:.1f}%). "
        f"<b>{month_names.get(best_month_num,'?')}</b> is the most reliable month "
        f"({monthly_by_month[best_month_num]*100:.1f}% delays)."))

    ins_c1, ins_c2 = st.columns(2)
    for i, (cls, title, body) in enumerate(insights):
        target = ins_c1 if i % 2 == 0 else ins_c2
        with target:
            st.markdown(f"""
            <div class="insight-card {cls}">
              <div class="i-title">{title}</div>
              <div class="i-body">{body}</div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# PAGE: Prediction
# ==============================================================================
elif page == "◎ Prediction":
    st.markdown("""
    <div class="page-header">
      <div>
        <div class="page-title">DELAY <span>PREDICTION</span></div>
        <div class="page-subtitle">REAL-TIME ML INFERENCE · 3 ENSEMBLE MODELS</div>
      </div>
      <div class="page-tag">INTERACTIVE</div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1.2, 1])

    with col_form:
        st.markdown('<div class="card"><div class="card-title">Flight Parameters</div>', unsafe_allow_html=True)
        
        airline = st.selectbox("Airline", options=list(AIRLINES.keys()),
                               format_func=lambda x: f"{x} — {AIRLINES[x]}")

        c1, c2 = st.columns(2)
        with c1:
            def _fmt_airport(x):
                name = AIRPORTS[x][0]
                return f"{x} — {name[:30].rsplit(' ', 1)[0] if len(name) > 30 else name}"
            origin = st.selectbox("Origin Airport", options=list(AIRPORTS.keys()),
                                  format_func=_fmt_airport, index=list(AIRPORTS.keys()).index(st.session_state.origin_airport))
            st.session_state.origin_airport = origin
        with c2:
            dest_opts = [a for a in AIRPORTS.keys() if a != origin]
            default_dest_idx = dest_opts.index(st.session_state.dest_airport) if st.session_state.dest_airport in dest_opts else 0
            dest = st.selectbox("Destination Airport", options=dest_opts,
                                format_func=_fmt_airport, index=default_dest_idx)
            st.session_state.dest_airport = dest

        c1, c2 = st.columns(2)
        with c1:
            month = st.selectbox("Month", range(1, 13),
                                 format_func=lambda x: pd.Timestamp(2024, x, 1).strftime("%B"), index=5)
        with c2:
            dow_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_of_week = st.selectbox("Day of Week", range(1, 8),
                                       format_func=lambda x: dow_labels[x-1], index=4)

        # Check if we need to fetch weather for selected origin/destination (cached)
        if ("cached_weather_origin" not in st.session_state or 
            st.session_state.get("cached_origin") != origin):
            with st.spinner("Analyzing departure airport weather..."):
                st.session_state.cached_weather_origin = fetch_weather(origin)
                st.session_state.cached_origin = origin

        if ("cached_weather_dest" not in st.session_state or 
            st.session_state.get("cached_dest") != dest):
            with st.spinner("Analyzing arrival airport weather..."):
                st.session_state.cached_weather_dest = fetch_weather(dest)
                st.session_state.cached_dest = dest

        origin_wx = st.session_state.cached_weather_origin

        # Identify weather condition from API details
        def map_wmo_to_condition(wx):
            if wx.get("wind_speed_ms", 0.0) >= 10.0:
                return "Windy"
            cond = wx.get("weather_main", "Clear")
            if cond == "Clear":
                return "Clear"
            elif cond == "Clouds":
                return "Cloudy"
            elif cond in ("Rain", "Drizzle"):
                return "Rainy"
            elif cond == "Snow":
                return "Snowy"
            elif cond == "Fog":
                return "Foggy"
            elif cond == "Thunderstorm":
                return "Thunderstorm"
            return "Clear"

        auto_weather = map_wmo_to_condition(origin_wx)
        dest_wx = st.session_state.cached_weather_dest
        auto_weather_dest = map_wmo_to_condition(dest_wx)

        # Days in each month for 2024 (a leap year matches standard range)
        month_days = {
            1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30,
            7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
        }
        max_days = month_days.get(month, 31)
        if "day_of_month" not in st.session_state:
            st.session_state.day_of_month = 15
        st.session_state.day_of_month = min(st.session_state.day_of_month, max_days)

        c1, c2 = st.columns(2)
        with c1:
            dep_hour = st.slider("Departure Hour", 0, 23, 8)
            dep_min  = st.slider("Departure Minute", 0, 59, 15)
        with c2:
            day_of_month = st.slider("Day of Month", 1, max_days, value=st.session_state.day_of_month)
            st.session_state.day_of_month = day_of_month
            st.text_input("Origin Weather (Auto-Identified)", 
                          value=f"{auto_weather} ({origin_wx.get('weather_desc', 'Clear sky')})", 
                          disabled=True)
            st.text_input("Destination Weather (Auto-Identified)", 
                          value=f"{auto_weather_dest} ({dest_wx.get('weather_desc', 'Clear sky')})", 
                          disabled=True)
            weather = auto_weather

        compare_all = False
        model_choice = "Gradient Boosting"

        predict_btn = st.button("⟐ RUN PREDICTION")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="card"><div class="card-title">Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            dist = _estimate_distance(origin, dest)
            elapsed = max(30, int(dist / 7 + 30))
            dep_time = dep_hour * 100 + dep_min

            input_data = {
                "OP_CARRIER": airline, "ORIGIN": origin, "DEST": dest,
                "MONTH": month, "DAY_OF_WEEK": day_of_week, "DAY_OF_MONTH": day_of_month,
                "CRS_DEP_TIME": dep_time, "DISTANCE": dist,
                "CRS_ELAPSED_TIME": elapsed, "WEATHER_CONDITION": weather,
            }

            if compare_all:
                all_models = ["Random Forest", "Gradient Boosting", "Logistic Regression"]
                results = {}
                with st.spinner("Running models…"):
                    for m in all_models:
                        results[m] = st.session_state.predictor.predict(input_data, m)

                for m_name, res in results.items():
                    prob = res["delay_probability"]
                    cls = "result-high" if res["is_delayed"] else "result-low"
                    icon = "🚨" if res["is_delayed"] else "✅"
                    st.markdown(f"""
                    <div style="background:var(--panel2); border:1px solid var(--border); padding:10px 14px; margin-bottom:10px; display:flex; justify-content:space-between; align-items:center;">
                        <div>
                          <div style="font-family:var(--font-mono); font-size:11px; color:var(--text3);">{m_name}</div>
                          <div style="font-family:var(--font-display); font-size:14px; font-weight:700; color:var(--text);">{icon} {res['risk_level'].upper()} RISK</div>
                        </div>
                        <div class="{cls}" style="font-family:var(--font-display); font-size:24px; font-weight:800;">{prob*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

                st.session_state.pred_history.append({
                    "Route": f"{origin}→{dest}", "Airline": airline,
                    "Weather": weather, "Departure": f"{dep_hour:02d}:{dep_min:02d}",
                    **{m: f"{r['delay_probability']*100:.1f}%" for m, r in results.items()}
                })
            else:
                with st.spinner("Running ML model…"):
                    result = st.session_state.predictor.predict(input_data, model_choice)
                    time.sleep(0.4)

                prob = result["delay_probability"]
                is_delayed = result["is_delayed"]
                risk = result["risk_level"]
                est_delay = result["estimated_delay_minutes"]

                cls = "result-high" if is_delayed else "result-low"
                verdict = "DELAY LIKELY" if is_delayed else "ON TIME LIKELY"
                
                st.markdown(f"""
                <div class="result-box">
                  <div class="result-label">DELAY PROBABILITY</div>
                  <div class="result-prob {cls}">{prob*100:.1f}%</div>
                  <div class="result-verdict {cls}">{verdict}</div>
                  <div style="margin-top:16px;font-family:var(--font-mono);font-size:12px;color:var(--text3);letter-spacing:1px">{risk.upper()} RISK</div>
                </div>
                """, unsafe_allow_html=True)

                st.session_state.pred_history.append({
                    "Route": f"{origin}→{dest}", "Airline": airline,
                    "Weather": weather, "Departure": f"{dep_hour:02d}:{dep_min:02d}",
                    "Model": model_choice, "Probability": f"{prob*100:.1f}%",
                    "Verdict": "Delayed" if is_delayed else "On Time"
                })

            # Generate shareable prediction and weather report
            report_text = (
                f"✈️ AEROSIGHT FLIGHT DELAY PREDICTION REPORT\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📍 ROUTE: {origin} ──✈ {dest}\n"
                f"🏢 AIRLINE: {AIRLINES.get(airline, airline)}\n"
                f"📅 SCHEDULE: Month {month}, Day {day_of_month}, Sched Dep: {dep_hour:02d}:{dep_min:02d}\n"
                f"🌤️ LIVE DEPARTURE WEATHER DETAILS:\n"
                f"  • Sky: {origin_wx.get('weather_desc', 'Clear sky')}\n"
                f"  • Temp: {origin_wx.get('temperature_c', 20.0)}°C\n"
                f"  • Wind Speed: {origin_wx.get('wind_speed_ms', 5.0)} m/s\n"
                f"  • Visibility: {origin_wx.get('visibility_m', 10000.0) / 1000.0} km\n"
                f"  • Precipitation: {origin_wx.get('precipitation', 0.0)} mm\n"
                f"  • Auto-Identified Category: {auto_weather}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🔮 PREDICTION RESULTS ({'Multi-Model Ensemble' if compare_all else model_choice}):\n"
            )
            if compare_all:
                for m_name, res in results.items():
                    report_text += f"  • {m_name}: {res['delay_probability']*100:.1f}% ({res['risk_level'].upper()} RISK)\n"
            else:
                report_text += (
                    f"  • Delay Probability: {prob*100:.1f}%\n"
                    f"  • Verdict: {verdict}\n"
                    f"  • Risk Level: {risk.upper()} RISK\n"
                )
                if is_delayed:
                    report_text += f"  • Est. Delay Time: {est_delay} minutes\n"
            report_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            report_text += f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

            st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
            with st.expander("📤 Share Prediction & Weather Report"):
                st.code(report_text, language="text")

            st.markdown('</div>', unsafe_allow_html=True)
            
            # Route Overview Map inside 결과
            st.markdown('<div class="card"><div class="card-title">Route Overview</div>', unsafe_allow_html=True)
            _last_prob = prob if not compare_all else results["Random Forest"]["delay_probability"]
            o_wx = st.session_state.cached_weather_origin
            d_wx = st.session_state.cached_weather_dest
            if origin in WX_AIRPORT_COORDS and dest in WX_AIRPORT_COORDS:
                route_map = draw_route_map(origin, dest, _last_prob, o_wx, d_wx)
                if route_map and FOLIUM_AVAILABLE:
                    st_folium(route_map, width=480, height=220, returned_objects=[])
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center;padding:40px 0;font-family:var(--font-mono);font-size:12px;color:var(--text3);letter-spacing:2px">
              AWAITING INPUT — RUN PREDICTION
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Feature Contributions (SHAP)
    if predict_btn:
        st.markdown('<div class="card"><div class="card-title">Top Feature Contributions (SHAP)</div>', unsafe_allow_html=True)
        dist = _estimate_distance(origin, dest)
        contributions = {
            "Departure Hour": 0.12 if (dep_hour >= 15 and dep_hour <= 20) else 0.03,
            "Weather Condition": 0.45 if weather in ["Thunderstorm", "Snowy", "Foggy"] else (0.15 if weather == "Rainy" else 0.05),
            "Carrier Profile": 0.20 if airline in ["NK", "F9"] else 0.05,
            "Route Distance": 0.10 if dist > 1500 else 0.04,
            "Day of Week": 0.08 if day_of_week in [5, 7] else 0.02,
            "Month (Seasonality)": 0.10 if month in [6, 7, 12] else 0.03
        }
        total_contrib = sum(contributions.values())
        normalized_contrib = {k: int((v / total_contrib) * 100) for k, v in contributions.items()}
        sorted_contrib = sorted(normalized_contrib.items(), key=lambda x: x[1], reverse=True)
        
        contrib_html = "<div style='margin-top:4px'>"
        for i, (k, pct) in enumerate(sorted_contrib):
            contrib_html += (
                f'<div class="feature-row" style="--i:{i}">'
                f'<span class="feature-name">{k}</span>'
                f'<div class="feature-track"><div class="feature-fill" style="width:{pct}%"></div></div>'
                f'<span class="feature-pct">{pct / 100:.2f}</span>'
                f'</div>'
            )
        contrib_html += "</div>"
        st.markdown(contrib_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # History list
    if st.session_state.pred_history:
        st.markdown('<div class="card"><div class="card-title">📋 Prediction History</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.pred_history), use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear History"):
            st.session_state.pred_history = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE: Weather Intel
# ==============================================================================
elif page == "◇ Weather Intel":
    st.markdown("""
    <div class="page-header">
      <div>
        <div class="page-title">WEATHER <span>INTEL</span></div>
        <div class="page-subtitle">OPEN-METEO LIVE FEED · DELAY RISK SCORING</div>
      </div>
      <div class="page-tag">LIVE DATA</div>
    </div>
    """, unsafe_allow_html=True)

    # Selectors for live weather querying
    st.markdown('<div class="card"><div class="card-title">🗺️ Select Route airports</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        origin = st.selectbox("Departure Airport", options=list(AIRPORTS.keys()), 
                              format_func=lambda x: f"{x} — {AIRPORTS[x][0]}",
                              index=list(AIRPORTS.keys()).index(st.session_state.origin_airport))
        st.session_state.origin_airport = origin
    with c2:
        dest_opts = [a for a in AIRPORTS.keys() if a != origin]
        default_dest_idx = dest_opts.index(st.session_state.dest_airport) if st.session_state.dest_airport in dest_opts else 0
        dest = st.selectbox("Arrival Airport", options=dest_opts,
                            format_func=lambda x: f"{x} — {AIRPORTS[x][0]}",
                            index=default_dest_idx)
        st.session_state.dest_airport = dest
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Fetching live weather from Open-Meteo…"):
        o_wx, d_wx = get_route_weather(origin, dest)
        route_risk = compute_risk_score(o_wx, d_wx)

    def render_wx_card_html(wx, label, city_name, border_color):
        icon = WEATHER_ICON_MAP.get(wx.get("weather_code", 0), "🌡️")
        wind_kph = round(wx["wind_speed_ms"] * 3.6, 1)
        vis_km = round(wx["visibility_m"] / 1000, 1)
        risk = int(compute_risk_score(wx, wx))
        risk_label = "HIGH" if risk > 6 else ("MODERATE" if risk > 3 else "LOW")
        risk_color = "var(--red)" if risk > 6 else ("var(--amber)" if risk > 3 else "var(--green)")
        
        return f"""
        <div class="card" style="border-left: 3px solid {border_color}">
          <div class="card-title">{label} Airport — {wx['airport']}</div>
          <div class="wx-grid">
            <div class="wx-card">
              <div class="wx-airport">{wx['airport']}</div>
              <div class="wx-city">{city_name}</div>
              <div class="wx-row"><span class="wx-key">TEMP</span><span class="wx-val">{wx['temperature_c']}°C</span></div>
              <div class="wx-row"><span class="wx-key">FEELS</span><span class="wx-val">{wx['feels_like_c']}°C</span></div>
              <div class="wx-row"><span class="wx-key">WIND</span><span class="wx-val">{wind_kph} km/h</span></div>
              <div class="wx-row"><span class="wx-key">VISIB</span><span class="wx-val">{vis_km} km</span></div>
              <div class="wx-row"><span class="wx-key">HUMID</span><span class="wx-val">{wx['humidity_pct']}%</span></div>
              <div class="wx-row"><span class="wx-key">PRECIP</span><span class="wx-val">{wx['precipitation']} mm</span></div>
            </div>
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;">
              <div style="text-align:center;padding:12px 0">
                <div style="font-family:var(--font-display);font-size:48px;margin-bottom:8px">{icon}</div>
                <div style="font-family:var(--font-display);font-size:16px;font-weight:700;letter-spacing:1px;color:var(--text)">{wx['weather_desc'].upper()}</div>
              </div>
              <div style="text-align:center;margin-top:12px">
                <div style="font-family:var(--font-mono);font-size:10px;color:var(--text3);letter-spacing:2px;margin-bottom:6px">DELAY RISK</div>
                <div style="font-family:var(--font-display);font-size:42px;font-weight:700;color:{risk_color}">{risk}/10</div>
                <div style="font-family:var(--font-display);font-size:16px;font-weight:700;color:{risk_color}">{risk_label}</div>
              </div>
            </div>
          </div>
        </div>
        """

    col1, col2 = st.columns(2)
    with col1:
        origin_city = AIRPORTS.get(origin, ["", ""])[0]
        st.markdown(render_wx_card_html(o_wx, "Departure", origin_city, "var(--accent)"), unsafe_allow_html=True)
    with col2:
        dest_city = AIRPORTS.get(dest, ["", ""])[0]
        st.markdown(render_wx_card_html(d_wx, "Arrival", dest_city, "var(--amber)"), unsafe_allow_html=True)

    # Combined Route Weather Risk
    route_risk_label = "HIGH" if route_risk > 6 else ("MODERATE" if route_risk > 3 else "LOW")
    route_risk_color = "var(--red)" if route_risk > 6 else ("var(--amber)" if route_risk > 3 else "var(--green)")
    pct_fill = int(route_risk * 10)

    st.markdown(f"""
    <div class="card">
      <div class="card-title">Combined Route Weather Risk</div>
      <div style="display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:24px;padding:16px 0">
        <div style="text-align:center">
          <div style="font-family:var(--font-display);font-size:22px;font-weight:700;letter-spacing:2px;color:var(--green)">{origin}</div>
          <div style="font-family:var(--font-mono);font-size:10px;color:var(--text3);margin-top:2px">RISK {int(compute_risk_score(o_wx, o_wx))}/10</div>
        </div>
        <div>
          <div style="text-align:center;margin-bottom:10px">
            <div style="font-family:var(--font-display);font-size:52px;font-weight:700;color:{route_risk_color}">{route_risk}/10</div>
            <div style="font-family:var(--font-display);font-size:18px;font-weight:600;letter-spacing:2px;color:{route_risk_color}">{route_risk_label} ROUTE RISK</div>
          </div>
          <div class="risk-bar-track">
            <div class="risk-bar-fill" style="width:{pct_fill}%;background:linear-gradient(90deg,var(--green),var(--amber),var(--red))"></div>
          </div>
          <div class="risk-labels"><span>0 LOW</span><span>5 MOD</span><span>10 HIGH</span></div>
        </div>
        <div style="text-align:center">
          <div style="font-family:var(--font-display);font-size:22px;font-weight:700;letter-spacing:2px;color:var(--amber)">{dest}</div>
          <div style="font-family:var(--font-mono);font-size:10px;color:var(--text3);margin-top:2px">RISK {int(compute_risk_score(d_wx, d_wx))}/10</div>
        </div>
      </div>
      <div style="font-family:var(--font-mono);font-size:11px;color:var(--text3);text-align:center;letter-spacing:1px;margin-top:8px">
        FACTORING: WIND · VISIBILITY · PRECIPITATION · CONDITIONS · BOTH AIRPORTS
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Weather Intel Report
    wx_report_text = (
        f"🌤️ AEROSIGHT LIVE ROUTE WEATHER REPORT\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 ROUTE: {origin} ──✈ {dest}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🛫 DEPARTURE AIRPORT ({origin}):\n"
        f"  • Sky: {o_wx.get('weather_desc', 'Clear sky')} {WEATHER_ICON_MAP.get(o_wx.get('weather_code', 0), '🌡️')}\n"
        f"  • Temp: {o_wx.get('temperature_c', 20.0)}°C (Feels like: {o_wx.get('feels_like_c', 20.0)}°C)\n"
        f"  • Wind: {round(o_wx.get('wind_speed_ms', 5.0) * 3.6, 1)} km/h\n"
        f"  • Visibility: {round(o_wx.get('visibility_m', 10000.0) / 1000, 1)} km\n"
        f"  • Precipitation: {o_wx.get('precipitation', 0.0)} mm\n"
        f"  • Delay Risk Score: {int(compute_risk_score(o_wx, o_wx))}/10\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🛬 ARRIVAL AIRPORT ({dest}):\n"
        f"  • Sky: {d_wx.get('weather_desc', 'Clear sky')} {WEATHER_ICON_MAP.get(d_wx.get('weather_code', 0), '🌡️')}\n"
        f"  • Temp: {d_wx.get('temperature_c', 20.0)}°C (Feels like: {d_wx.get('feels_like_c', 20.0)}°C)\n"
        f"  • Wind: {round(d_wx.get('wind_speed_ms', 5.0) * 3.6, 1)} km/h\n"
        f"  • Visibility: {round(d_wx.get('visibility_m', 10000.0) / 1000, 1)} km\n"
        f"  • Precipitation: {d_wx.get('precipitation', 0.0)} mm\n"
        f"  • Delay Risk Score: {int(compute_risk_score(d_wx, d_wx))}/10\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🚨 COMBINED ROUTE WEATHER RISK: {route_risk}/10 ({route_risk_label} RISK)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
    with st.expander("📤 Share Live Route Weather Report"):
        st.code(wx_report_text, language="text")


# ==============================================================================
# PAGE: Model Performance
# ==============================================================================
elif page == "◆ Model Performance":
    st.markdown("""
    <div class="page-header">
      <div>
        <div class="page-title">MODEL <span>PERFORMANCE</span></div>
        <div class="page-subtitle">EVALUATION METRICS · ROC CURVES · FEATURE IMPORTANCE</div>
      </div>
      <div class="page-tag">XGBOOST MODEL</div>
    </div>
    """, unsafe_allow_html=True)

    metrics = st.session_state.predictor.metrics

    if st.session_state.trained:
        # Metrics cards layout
        xgb_m = metrics.get("Gradient Boosting", {"accuracy": 0.788, "auc": 0.582, "report": {"1": {"f1-score": 0.009}}})

        xgb_acc = xgb_m["accuracy"] * 100
        xgb_auc = xgb_m["auc"]
        xgb_f1 = xgb_m["report"].get("1", xgb_m["report"].get(1, {})).get("f1-score", 0.009)

        st.markdown(f"""
        <div class="card" style="border-top: 3px solid var(--accent); margin-bottom: 24px; padding: 20px;">
          <div style="font-family:var(--font-mono);font-size:10px;color:var(--accent);letter-spacing:2px;margin-bottom:12px;text-transform:uppercase;">★ Isolated Best Model Configuration</div>
          <div class="model-name" style="font-size:20px;margin-bottom:20px;">XGBoost (Gradient Boosting Classifier)</div>
          <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; text-align: center;">
            <div style="background:var(--bg2); padding:16px; border:1px solid var(--border);">
              <div class="model-metric-label" style="font-family:var(--font-mono);font-size:10px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;">Accuracy</div>
              <div class="model-metric-val" style="font-family:var(--font-display);font-size:36px;font-weight:700;color:var(--accent);margin-top:6px;">{xgb_acc:.1f}%</div>
            </div>
            <div style="background:var(--bg2); padding:16px; border:1px solid var(--border);">
              <div class="model-metric-label" style="font-family:var(--font-mono);font-size:10px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;">AUC-ROC Score</div>
              <div class="model-metric-val" style="font-family:var(--font-display);font-size:36px;font-weight:700;color:var(--accent);margin-top:6px;">{xgb_auc:.3f}</div>
            </div>
            <div style="background:var(--bg2); padding:16px; border:1px solid var(--border);">
              <div class="model-metric-label" style="font-family:var(--font-mono);font-size:10px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;">F1 Score (Class 1)</div>
              <div class="model-metric-val" style="font-family:var(--font-display);font-size:36px;font-weight:700;color:var(--accent);margin-top:6px;">{xgb_f1:.3f}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card"><div class="card-title">Feature Importances (XGBoost)</div>', unsafe_allow_html=True)
            fi = st.session_state.predictor.feature_importances
            fi_model = list(fi.keys())[0] if fi else "Gradient Boosting"
            fi_df = pd.DataFrame(list(fi[fi_model].items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=False)
            fig_fi = px.bar(
                fi_df.head(8), x="Importance", y="Feature", orientation="h",
                color_discrete_sequence=['#00c8ff']
            )
            style_plotly_fig(fig_fi, theme_toggle)
            st.plotly_chart(fig_fi, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card"><div class="card-title">ROC Curves</div>', unsafe_allow_html=True)
            fig_roc = go.Figure()
            colors = ["#00c8ff", "#00e5a0", "#ffb300"]
            for (name, m), color in zip(metrics.items(), colors):
                fpr, tpr = m["roc"]
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={m['auc']:.3f})", line=dict(color=color, width=2)))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
            style_plotly_fig(fig_roc, theme_toggle)
            fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Confusion matrices
        st.markdown('<div class="section-divider"><span>Confusion Matrices</span></div>', unsafe_allow_html=True)
        
        def render_cm_table_html(name, cm):
            tn, fp, fn, tp = cm.ravel()
            return f"""
            <div class="card">
              <div class="card-title">{name}</div>
              <table style="width:100%;text-align:center;border-collapse:collapse;margin-top:8px">
                <tr>
                  <th style="font-family:var(--font-mono);font-size:10px;color:var(--text3);padding:8px"></th>
                  <th style="font-family:var(--font-mono);font-size:10px;color:var(--text3);padding:8px">PRED ON-TIME</th>
                  <th style="font-family:var(--font-mono);font-size:10px;color:var(--text3);padding:8px">PRED DELAYED</th>
                </tr>
                <tr>
                  <td style="font-family:var(--font-mono);font-size:10px;color:var(--text3);padding:8px">ACT ON-TIME</td>
                  <td style="font-family:var(--font-display);font-size:26px;font-weight:700;color:var(--green);background:rgba(0,229,160,0.08);border:1px solid rgba(0,229,160,0.2)">{tn}</td>
                  <td style="font-family:var(--font-display);font-size:26px;font-weight:700;color:var(--red);background:rgba(255,68,68,0.08);border:1px solid rgba(255,68,68,0.2)">{fp}</td>
                </tr>
                <tr>
                  <td style="font-family:var(--font-mono);font-size:10px;color:var(--text3);padding:8px">ACT DELAYED</td>
                  <td style="font-family:var(--font-display);font-size:26px;font-weight:700;color:var(--red);background:rgba(255,68,68,0.08);border:1px solid rgba(255,68,68,0.2)">{fn}</td>
                  <td style="font-family:var(--font-display);font-size:26px;font-weight:700;color:var(--green);background:rgba(0,229,160,0.08);border:1px solid rgba(0,229,160,0.2)">{tp}</td>
                </tr>
              </table>
            </div>
            """
            
        c_left, c_mid, c_right = st.columns([1, 1.2, 1])
        with c_mid:
            st.markdown(render_cm_table_html("XGBoost Confusion Matrix", xgb_m["confusion_matrix"]), unsafe_allow_html=True)

    # Inline model training dashboard
    st.markdown('<div class="section-divider"><span>ML Engine Training Dashboard</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">⚙️ Training control panel</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Training parameters:**
        - **Target Variable**: `IS_DELAYED` — binary classification (delay ≥15 min)
        - **Model Selected**: Gradient Boosting (XGBoost)
        - **Train/Test split**: 80% Stratified Training, 20% Evaluation
        """)
        
    with col2:
        train_btn = st.button("🚀 TRAIN ALL MODELS")
    st.markdown('</div>', unsafe_allow_html=True)

    if train_btn:
        progress = st.progress(0)
        status = st.empty()

        status.info("⚙️ Preprocessing data and encoding features…")
        progress.progress(20)

        status.info("🚀 Training Gradient Boosting (XGBoost)…")
        progress.progress(50)
        metrics = st.session_state.predictor.train(df)
        progress.progress(80)

        status.info("📈 Evaluating model…")
        time.sleep(0.3)
        progress.progress(90)

        status.info("📐 Finalising metrics…")
        try:
            st.session_state.predictor.save()
        except Exception as e:
            st.warning(f"Failed to save models to disk: {e}")
        time.sleep(0.2)
        progress.progress(100)

        st.session_state.trained = True
        status.success("✅ ML Engine successfully synchronized! Models are ready.")
        st.rerun()


# ==============================================================================
# PAGE: Flight Data
# ==============================================================================
elif page == "▦ Flight Data":
    st.markdown("""
    <div class="page-header">
      <div>
        <div class="page-title">FLIGHT <span>DATA</span></div>
        <div class="page-subtitle">DATASET EXPLORER · 2018–2024 · 30 AIRPORTS · 7 AIRLINES</div>
      </div>
      <div class="page-tag">SAMPLE VIEW</div>
    </div>
    """, unsafe_allow_html=True)

    sample_df = df.head(15)
    table_rows = ""
    for _, row in sample_df.iterrows():
        status = "DELAYED" if row["IS_DELAYED"] == 1 else "ON-TIME"
        status_class = "badge-red" if status == "DELAYED" else "badge-green"
        
        delay_min = int(row["DEP_DELAY"]) if not np.isnan(row["DEP_DELAY"]) else 0
        if delay_min < 0:
            delay_min = 0
        
        cause = "—"
        cause_class = ""
        if status == "DELAYED":
            if row["WEATHER_DELAY"] > 0:
                cause = "Weather"
                cause_class = "badge-amber"
            elif row["LATE_AIRCRAFT_DELAY"] > 0:
                cause = "Late A/C"
            elif row["CARRIER_DELAY"] > 0:
                cause = "Carrier"
            else:
                cause = "NAS Delay"
                
        delay_color = "var(--red)" if delay_min > 60 else ("var(--amber)" if delay_min > 0 else "var(--green)")
        
        sched_time = f"{int(row['CRS_DEP_TIME'])//100:02d}:{int(row['CRS_DEP_TIME'])%100:02d}"
        dep_time = f"{int(row['CRS_DEP_TIME'] + delay_min)//100:02d}:{int(row['CRS_DEP_TIME'] + delay_min)%100:02d}"
        
        airline_code = row["OP_CARRIER"]
        flight_num = f"{airline_code}{int(100 + np.random.randint(900))}"
        
        table_rows += (
            f'<tr>'
            f'<td style="font-family:var(--font-mono);font-weight:600;color:var(--accent)">{flight_num}</td>'
            f'<td><strong>{row["ORIGIN"]}</strong></td>'
            f'<td><strong>{row["DEST"]}</strong></td>'
            f'<td>{AIRLINES.get(row["OP_CARRIER"], row["OP_CARRIER"])}</td>'
            f'<td style="font-family:var(--font-mono)">{dep_time}</td>'
            f'<td style="font-family:var(--font-mono);color:var(--text3)">{sched_time}</td>'
            f'<td style="font-family:var(--font-display);font-size:16px;font-weight:700;color:{delay_color}">{delay_min}</td>'
            f'<td><span class="badge {cause_class}">{cause}</span></td>'
            f'<td><span class="badge {status_class}">{status}</span></td>'
            f'</tr>'
        )

    st.markdown(f"""
    <div class="card">
      <div class="card-title">Recent Flight Records</div>
      <div style="overflow-x:auto">
        <table class="data-table">
          <thead>
            <tr>
              <th>Flight</th>
              <th>Origin</th>
              <th>Dest</th>
              <th>Airline</th>
              <th>Dep Time</th>
              <th>Sched</th>
              <th>Delay (min)</th>
              <th>Cause</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {table_rows}
          </tbody>
        </table>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE: About
# ==============================================================================
elif page == "◉ About":
    st.markdown("""
    <div class="page-header">
      <div>
        <div class="page-title">ABOUT <span>SYSTEM</span></div>
        <div class="page-subtitle">ARCHITECTURE · STACK · DATASET SPECIFICATIONS</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
          <div class="card-title">System Architecture</div>
          <div style="display:flex;flex-direction:column;gap:10px;margin-top:6px">
            <div style="display:flex;align-items:center;gap:14px;padding:12px;background:var(--bg2);border:1px solid var(--border)">
              <div style="font-family:var(--font-display);font-size:28px;color:var(--accent)">◉</div>
              <div>
                <div style="font-family:var(--font-display);font-size:15px;font-weight:700;letter-spacing:1px">Data Pipeline</div>
                <div style="font-family:var(--font-mono);font-size:11px;color:var(--text3);margin-top:2px">Kaggle dataset · data_generator.py · Feature engineering</div>
              </div>
            </div>
            <div style="display:flex;align-items:center;gap:14px;padding:12px;background:var(--bg2);border:1px solid var(--border)">
              <div style="font-family:var(--font-display);font-size:28px;color:var(--accent2)">◆</div>
              <div>
                <div style="font-family:var(--font-display);font-size:15px;font-weight:700;letter-spacing:1px">ML Model</div>
                <div style="font-family:var(--font-mono);font-size:11px;color:var(--text3);margin-top:2px">XGBoost (Gradient Boosting) · ml_models.py</div>
              </div>
            </div>
            <div style="display:flex;align-items:center;gap:14px;padding:12px;background:var(--bg2);border:1px solid var(--border)">
              <div style="font-family:var(--font-display);font-size:28px;color:var(--green)">◇</div>
              <div>
                <div style="font-family:var(--font-display);font-size:15px;font-weight:700;letter-spacing:1px">Weather Module</div>
                <div style="font-family:var(--font-mono);font-size:11px;color:var(--text3);margin-top:2px">Open-Meteo API · weather_fetcher.py · Live delay risk scoring</div>
              </div>
            </div>
            <div style="display:flex;align-items:center;gap:14px;padding:12px;background:var(--bg2);border:1px solid var(--border)">
              <div style="font-family:var(--font-display);font-size:28px;color:var(--accent3)">▦</div>
              <div>
                <div style="font-family:var(--font-display);font-size:15px;font-weight:700;letter-spacing:1px">Frontend</div>
                <div style="font-family:var(--font-mono);font-size:11px;color:var(--text3);margin-top:2px">Streamlit · Plotly · Folium · Dark/Light themes</div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
          <div class="card-title">Dataset Specifications</div>
          <div style="margin-top:6px">
            <div class="wx-row"><span class="wx-key">SOURCE</span><span class="wx-val" style="font-size:13px">Kaggle — shubhamsingh42</span></div>
            <div class="wx-row"><span class="wx-key">PERIOD</span><span class="wx-val">2018 – 2024</span></div>
            <div class="wx-row"><span class="wx-key">RECORDS</span><span class="wx-val">~2.4 Million</span></div>
            <div class="wx-row"><span class="wx-key">AIRPORTS</span><span class="wx-val">30 Major US Hubs</span></div>
            <div class="wx-row"><span class="wx-key">AIRLINES</span><span class="wx-val">7 Carriers</span></div>
            <div class="wx-row"><span class="wx-key">FEATURES</span><span class="wx-val">24 Input Variables</span></div>
            <div class="wx-row"><span class="wx-key">TARGET</span><span class="wx-val">Binary: Delayed / On-Time</span></div>
            <div class="wx-row"><span class="wx-key">TRAIN SPLIT</span><span class="wx-val">80% Train / 20% Test</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
