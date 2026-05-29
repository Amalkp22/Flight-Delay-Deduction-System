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
from plotly.subplots import make_subplots
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
    page_title="Flight Delay Prediction System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Animated header ── */
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        animation: fadeSlideDown 0.6s ease;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute; inset: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.04), transparent);
        animation: shimmer 3s infinite;
    }
    @keyframes shimmer { 0%{transform:translateX(-100%)} 100%{transform:translateX(100%)} }
    @keyframes fadeSlideDown { from{opacity:0;transform:translateY(-18px)} to{opacity:1;transform:translateY(0)} }

    .main-header h1 { font-size: 2.4rem; font-weight: 700; margin: 0; }
    .main-header p  { font-size: 1rem; opacity: 0.8; margin-top: 0.5rem; }

    /* ── Metric cards with hover ── */
    .metric-card {
        background: linear-gradient(135deg, #1e2a3a, #243447);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 16px rgba(0,0,0,0.25);
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        cursor: default;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 28px rgba(52,152,219,0.3);
        border-left-color: #5dade2;
    }
    .metric-card .label { font-size: 0.78rem; color: #aab; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #fff; }
    .metric-card .delta { font-size: 0.85rem; margin-top: 0.2rem; color: #7fb3d3; }

    /* ── Risk colours ── */
    .risk-low      { color: #00b894; font-weight: 700; }
    .risk-moderate { color: #f39c12; font-weight: 700; }
    .risk-high     { color: #e17055; font-weight: 700; }
    .risk-veryhigh { color: #d63031; font-weight: 700; }

    /* ── Prediction result box ── */
    .prediction-box {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
        animation: fadeIn 0.4s ease;
    }
    @keyframes fadeIn { from{opacity:0;transform:scale(0.97)} to{opacity:1;transform:scale(1)} }
    .prediction-box h2 { font-size: 1.8rem; margin: 0.5rem 0; }
    .prediction-box .prob { font-size: 3rem; font-weight: 800; }
    .delayed-box { background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; }
    .ontime-box  { background: linear-gradient(135deg, #00b894, #00cec9); color: white; }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.3rem; font-weight: 600; color: #5dade2;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.4rem; margin: 1.5rem 0 1rem 0;
    }

    /* ── Sidebar ── */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #12263a 100%);
    }
    div[data-testid="stSidebar"] * { color: white !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] { font-weight: 600; color: #aaa; transition: color 0.2s; }
    .stTabs [aria-selected="true"] { color: #3498db !important; border-bottom: 3px solid #3498db; }

    /* ── Info tags ── */
    .info-tag {
        display: inline-block;
        background: rgba(52,152,219,0.15);
        color: #5dade2;
        border: 1px solid rgba(52,152,219,0.3);
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 0.15rem;
        transition: background 0.2s;
    }
    .info-tag:hover { background: rgba(52,152,219,0.3); }

    /* ── Workflow step cards ── */
    .step-card {
        background: linear-gradient(135deg, #1a2535, #1e2d40);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        height: 140px;
        border-top: 3px solid #3498db;
        transition: transform 0.2s, box-shadow 0.2s;
        display: flex; flex-direction: column; justify-content: center;
    }
    .step-card:hover { transform: translateY(-5px); box-shadow: 0 8px 24px rgba(52,152,219,0.25); }

    /* ── Page title ── */
    .page-title {
        color: #007BFF !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        animation: fadeSlideDown 0.4s ease;
    }

    /* ── Filter bar ── */
    .filter-bar {
        background: linear-gradient(135deg, #1a2535, #1e2d40);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(52,152,219,0.2);
    }

    /* ── Compare model cards ── */
    .model-compare-card {
        background: linear-gradient(135deg, #1a2535, #1e2d40);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border-top: 3px solid #3498db;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Airport coordinates (lat/lon for map) ────────────────────────────────────
AIRPORT_COORDS = {
    "ATL": (33.6407, -84.4277, "Atlanta Hartsfield-Jackson"),
    "LAX": (33.9425, -118.4081, "Los Angeles International"),
    "ORD": (41.9742, -87.9073, "Chicago O'Hare"),
    "DFW": (32.8998, -97.0403, "Dallas/Fort Worth"),
    "DEN": (39.8561, -104.6737, "Denver International"),
    "JFK": (40.6413, -73.7781, "New York JFK"),
    "SFO": (37.6213, -122.3790, "San Francisco International"),
    "SEA": (47.4502, -122.3088, "Seattle-Tacoma"),
    "MIA": (25.7959, -80.2870, "Miami International"),
    "BOS": (42.3656, -71.0096, "Boston Logan"),
    "LAS": (36.0840, -115.1537, "Las Vegas Harry Reid"),
    "MCO": (28.4312, -81.3081, "Orlando International"),
    "CLT": (35.2140, -80.9431, "Charlotte Douglas"),
    "PHX": (33.4373, -112.0078, "Phoenix Sky Harbor"),
    "IAH": (29.9902, -95.3368, "Houston George Bush"),
    "EWR": (40.6895, -74.1745, "Newark Liberty"),
    "MSP": (44.8848, -93.2223, "Minneapolis-Saint Paul"),
    "DTW": (42.2124, -83.3534, "Detroit Metropolitan"),
    "PHL": (39.8744, -75.2424, "Philadelphia International"),
    "LGA": (40.7769, -73.8740, "New York LaGuardia"),
    "FLL": (26.0742, -80.1506, "Fort Lauderdale"),
    "BWI": (39.1754, -76.6684, "Baltimore/Washington"),
    "SLC": (40.7884, -111.9778, "Salt Lake City"),
    "MDW": (41.7868, -87.7522, "Chicago Midway"),
    "IAD": (38.9531, -77.4565, "Washington Dulles"),
    "DCA": (38.8521, -77.0377, "Ronald Reagan Washington"),
    "HNL": (21.3245, -157.9251, "Honolulu Daniel K. Inouye"),
    "ANC": (61.1743, -149.9963, "Ted Stevens Anchorage"),
    "PDX": (45.5887, -122.5975, "Portland International"),
    "STL": (38.7487, -90.3700, "St. Louis Lambert"),
}

# ── Live weather helper functions ────────────────────────────────────────────
def get_weather(lat, lon, api_key):
    """Fetch current weather from OpenWeatherMap API."""
    try:
        url = (f"https://api.openweathermap.org/data/2.5/weather"
               f"?lat={lat}&lon={lon}&appid={api_key}&units=metric")
        r = requests.get(url, timeout=6)
        d = r.json()
        if r.status_code != 200:
            return None
        return {
            "temp":       round(d["main"]["temp"]),
            "feels_like": round(d["main"]["feels_like"]),
            "humidity":   d["main"]["humidity"],
            "wind_kph":   round(d["wind"]["speed"] * 3.6, 1),
            "visibility": round(d.get("visibility", 10000) / 1000, 1),
            "condition":  d["weather"][0]["description"].title(),
            "icon_code":  d["weather"][0]["main"],
        }
    except Exception:
        return None

def weather_delay_risk(w):
    """Return (score 0-100, label, hex colour) based on weather params."""
    if not w:
        return 0, "Unknown", "#888888"
    score = 0
    if w["wind_kph"] > 65:   score += 40
    elif w["wind_kph"] > 45: score += 20
    elif w["wind_kph"] > 28: score += 10
    if w["visibility"] < 1:  score += 40
    elif w["visibility"] < 5: score += 20
    elif w["visibility"] < 8: score += 10
    if "Thunderstorm" in w["icon_code"]: score += 20
    elif "Snow" in w["icon_code"]:       score += 15
    elif "Rain" in w["icon_code"]:       score += 10
    label  = "High"     if score > 50 else "Moderate" if score > 25 else "Low"
    colour = "#e74c3c"  if score > 50 else "#f39c12"  if score > 25 else "#2ecc71"
    return score, label, colour

def weather_icon(icon_code):
    icons = {
        "Clear": "☀️", "Clouds": "☁️", "Rain": "🌧️",
        "Drizzle": "🌦️", "Thunderstorm": "⛈️", "Snow": "❄️",
        "Mist": "🌫️", "Fog": "🌁", "Haze": "🌫️",
    }
    return icons.get(icon_code, "🌡️")

def _wx_popup_html(wx: dict, label: str, iata: str) -> str:
    """Build a compact HTML popup card for a weather-annotated airport marker."""
    live_badge = (
        '<span style="background:#2ecc71;color:#000;font-size:0.65rem;'
        'padding:1px 5px;border-radius:4px;font-weight:700">LIVE</span>'
        if wx.get("live") else
        '<span style="background:#e74c3c;color:#fff;font-size:0.65rem;'
        'padding:1px 5px;border-radius:4px;font-weight:700">OFFLINE</span>'
    )
    icon = WEATHER_ICON_MAP.get(wx.get("weather_code", 0), "🌡️")
    wind_kph = round(wx["wind_speed_ms"] * 3.6, 1)
    vis_km   = round(wx["visibility_m"] / 1000, 1)
    return f"""
    <div style="font-family:Inter,sans-serif;min-width:200px;
                background:#1a2535;color:#eee;border-radius:10px;padding:10px 14px">
      <div style="font-size:1rem;font-weight:700;margin-bottom:4px">
        {label} &nbsp;{iata} &nbsp;{live_badge}
      </div>
      <div style="font-size:1.6rem">{icon}
        <span style="font-size:0.9rem;vertical-align:middle">{wx['weather_desc']}</span>
      </div>
      <table style="margin-top:6px;font-size:0.82rem;width:100%;border-collapse:collapse">
        <tr><td>🌡️ Temp</td><td><b>{wx['temperature_c']}°C</b>
            (feels {wx['feels_like_c']}°C)</td></tr>
        <tr><td>💨 Wind</td><td><b>{wind_kph} km/h</b></td></tr>
        <tr><td>👁️ Visibility</td><td><b>{vis_km} km</b></td></tr>
        <tr><td>💧 Humidity</td><td><b>{wx['humidity_pct']}%</b></td></tr>
        <tr><td>🌧️ Precip</td><td><b>{wx['precipitation']} mm</b></td></tr>
        <tr><td>☁️ Cloud cover</td><td><b>{wx['clouds_pct']}%</b></td></tr>
        <tr><td>📡 Updated</td><td>{wx['fetched_at']}</td></tr>
      </table>
    </div>"""


def draw_route_map(origin, dest, delay_prob,
                   origin_wx: dict = None, dest_wx: dict = None):
    """Draw folium map with route line, weather popups, and risk overlay."""
    if not FOLIUM_AVAILABLE:
        return None
    o_info = AIRPORT_COORDS.get(origin)
    d_info = AIRPORT_COORDS.get(dest)
    if not o_info or not d_info:
        return None

    mid_lat = (o_info[0] + d_info[0]) / 2
    mid_lon = (o_info[1] + d_info[1]) / 2
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=4,
                   tiles="CartoDB dark_matter")

    route_color = ("#e74c3c" if delay_prob > 0.65
                   else "#f39c12" if delay_prob > 0.40
                   else "#2ecc71")

    # Route line
    folium.PolyLine(
        locations=[[o_info[0], o_info[1]], [d_info[0], d_info[1]]],
        color=route_color, weight=3.5, opacity=0.9,
        tooltip=f"Delay probability: {delay_prob*100:.1f}%",
        dash_array="8 4" if delay_prob > 0.5 else None,
    ).add_to(m)

    # Departure marker — with live weather popup if available
    dep_popup = None
    if origin_wx:
        dep_popup = folium.Popup(
            folium.IFrame(_wx_popup_html(origin_wx, "🛫 Departure", origin),
                          width=240, height=230),
            max_width=260,
        )
    folium.Marker(
        location=[o_info[0], o_info[1]],
        tooltip=f"🛫 {o_info[2]} ({origin}) — click for weather",
        popup=dep_popup,
        icon=folium.Icon(color="red", icon="plane", prefix="fa"),
    ).add_to(m)

    # Arrival marker — with live weather popup if available
    arr_popup = None
    if dest_wx:
        arr_popup = folium.Popup(
            folium.IFrame(_wx_popup_html(dest_wx, "🛬 Arrival", dest),
                          width=240, height=230),
            max_width=260,
        )
    folium.Marker(
        location=[d_info[0], d_info[1]],
        tooltip=f"🛬 {d_info[2]} ({dest}) — click for weather",
        popup=arr_popup,
        icon=folium.Icon(color="blue", icon="flag", prefix="fa"),
    ).add_to(m)

    # All other airports as small dots
    for code, (lat, lon, name) in AIRPORT_COORDS.items():
        if code in (origin, dest):
            continue
        folium.CircleMarker(
            location=[lat, lon], radius=4,
            color="#5dade2", fill=True, fill_color="#5dade2",
            fill_opacity=0.4, weight=1,
            tooltip=f"{code} — {name}",
        ).add_to(m)

    return m


# ── Session state ─────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "predictor" not in st.session_state:
    st.session_state.predictor = FlightDelayPredictor()
if "trained" not in st.session_state:
    st.session_state.trained = False
if "pred_history" not in st.session_state:
    st.session_state.pred_history = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ Flight Delay System")
    st.markdown("---")
    _nav_default = st.session_state.pop("_nav", None)
    _nav_options = ["🏠 Overview", "📊 EDA & Insights", "🤖 Model Training",
                    "🔮 Predict a Flight", "📈 Model Performance"]
    _nav_index = _nav_options.index(_nav_default) if _nav_default in _nav_options else 0
    page = st.radio(
        "Navigation",
        _nav_options,
        index=_nav_index,
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("Kaggle Flight Delay 2018–2024")
    st.markdown("**Technology**")
    st.markdown("Random Forest · Gradient Boosting · Logistic Regression")
    st.markdown("---")

    


# ── Auto-load data & auto-train ───────────────────────────────────────────────
if st.session_state.df is None:
    with st.spinner("Loading dataset…"):
        st.session_state.df = generate_flight_data(20000)

df = st.session_state.df

if not st.session_state.trained:
    with st.spinner("Training models in the background… (~30s)"):
        st.session_state.predictor.train(df)
        st.session_state.trained = True


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class="main-header">
        <h1>✈️ Flight Delay Prediction System</h1>
        <p>Machine Learning · Aerospace · 2018–2024 Historical Flight Data</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Interactive year filter ───────────────────────────────────────────────
    years = sorted(df["YEAR"].unique())
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
    sel_years = st.select_slider(
        "📅 Filter by Year Range",
        options=years,
        value=(min(years), max(years)),
    )
    st.markdown('</div>', unsafe_allow_html=True)
    fdf = df[df["YEAR"].between(sel_years[0], sel_years[1])]

    # ── KPI cards ────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    total     = len(fdf)
    delayed   = int(fdf["IS_DELAYED"].sum())
    cancelled = int(fdf["CANCELLED"].sum())
    avg_delay = fdf[fdf["DEP_DELAY"] > 0]["DEP_DELAY"].mean()

    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Total Flights</div>
            <div class="value">{total:,}</div>
            <div class="delta">📅 {sel_years[0]} – {sel_years[1]}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        pct = delayed / total * 100
        st.markdown(f"""<div class="metric-card">
            <div class="label">Delayed Flights</div>
            <div class="value">{delayed:,}</div>
            <div class="delta">🔴 {pct:.1f}% of all flights</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Cancelled Flights</div>
            <div class="value">{cancelled:,}</div>
            <div class="delta">❌ {cancelled/total*100:.1f}% cancellation rate</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Avg Delay (when delayed)</div>
            <div class="value">{avg_delay:.0f} min</div>
            <div class="delta">⏱️ Departure delay</div>
        </div>""", unsafe_allow_html=True)

    # ── About + chart ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">About This Project</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown("""
        **Problem Statement**  
        Flight delays cost airlines **billions** annually and cause significant passenger inconvenience.
        This system predicts whether a flight will be delayed (≥15 min) before departure using historical
        patterns and real-time inputs.

        **Dataset**  
        Based on the [Kaggle Flight Delay Dataset 2018–2024](https://kaggle.com/datasets/shubhamsingh42/flight-delay-dataset-2018-2024),
        containing millions of domestic US flight records with departure/arrival delays, cancellations,
        weather conditions, and carrier information.

        **ML Approach**
        - **Binary Classification**: Delayed (≥15 min) vs On-Time
        - **3 Models compared**: Random Forest, Gradient Boosting, Logistic Regression
        - **Features**: Airline, Origin, Destination, Time, Distance, Weather, Day of Week
        """)
        tags = ["Random Forest", "Gradient Boosting", "Scikit-learn", "Plotly", "Streamlit",
                "Pandas", "NumPy", "Aerospace", "2018–2024"]
        tags_html = "".join(f'<span class="info-tag">{t}</span>' for t in tags)
        st.markdown(tags_html, unsafe_allow_html=True)
    #with c2:
        #delay_dist = fdf["DEP_DELAY"].dropna().clip(-20, 200)
        #fig = px.histogram(
            #delay_dist, nbins=60,
            #color_discrete_sequence=["#3498db"],
            #title="Departure Delay Distribution",
            #labels={"value": "Delay (minutes)", "count": "Flights"},
        #)
        #fig.add_vline(x=15, line_dash="dash", line_color="red",
                      #annotation_text="Delay threshold (15 min)")
        #fig.update_layout(showlegend=False, height=300, margin=dict(t=40, b=20),
                          #paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          #font_color="#ccc")
        #st.plotly_chart(fig, use_container_width=True)

    # ── Quick nav buttons ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Jump to:**")
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        if st.button("📊 Explore EDA & Insights", use_container_width=True):
            st.session_state["_nav"] = "📊 EDA & Insights"
            st.rerun()
    with qc2:
        if st.button("🔮 Predict a Flight", use_container_width=True):
            st.session_state["_nav"] = "🔮 Predict a Flight"
            st.rerun()
    with qc3:
        if st.button("📈 View Model Performance", use_container_width=True):
            st.session_state["_nav"] = "📈 Model Performance"
            st.rerun()

    # ── Workflow steps ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Workflow</div>', unsafe_allow_html=True)
    steps = [
        ("1️⃣", "Data Ingestion", "Load 2018–2024 flight records with 24+ features"),
        ("2️⃣", "EDA & Feature Eng.", "Analyze delay patterns, encode categoricals"),
        ("3️⃣", "Model Training", "Train RF, GBM & LR classifiers on 80% data"),
        ("4️⃣", "Evaluation", "Compare accuracy, AUC-ROC, F1 across models"),
        ("5️⃣", "Prediction", "Input a flight → get delay probability instantly"),
    ]
    cols = st.columns(5)
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div style="font-size:1.8rem">{icon}</div>
                <div style="font-weight:700;font-size:0.9rem;margin:0.3rem 0;color:#fff">{title}</div>
                <div style="font-size:0.75rem;color:#8ab">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA & Insights
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Insights":
    st.markdown('<h2 class="page-title">📊 Exploratory Data Analysis & Insights</h2>', unsafe_allow_html=True)

    # ── Global interactive filters ────────────────────────────────────────────
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        all_airlines = sorted(df["OP_CARRIER"].unique())
        sel_airlines = st.multiselect("✈️ Filter Airlines", all_airlines, default=all_airlines)
    with fc2:
        years = sorted(df["YEAR"].unique())
        sel_yr = st.select_slider("📅 Year Range", options=years, value=(min(years), max(years)))
    with fc3:
        all_weather = sorted(df["WEATHER_CONDITION"].unique())
        sel_wx = st.multiselect("🌤️ Weather", all_weather, default=all_weather)
    st.markdown('</div>', unsafe_allow_html=True)

    fdf = df[
        df["OP_CARRIER"].isin(sel_airlines) &
        df["YEAR"].between(sel_yr[0], sel_yr[1]) &
        df["WEATHER_CONDITION"].isin(sel_wx)
    ]
    st.caption(f"Showing **{len(fdf):,}** flights after filters")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["⏱️ Delay Patterns", "✈️ Airline Analysis", "🗺️ Route Analysis", "🌦️ Weather Impact"]
    )

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            monthly = fdf.groupby("MONTH")["IS_DELAYED"].mean().reset_index()
            monthly["Month"] = pd.to_datetime(monthly["MONTH"], format="%m").dt.strftime("%b")
            fig = px.bar(monthly, x="Month", y="IS_DELAYED",
                         title="Delay Rate by Month",
                         color="IS_DELAYED", color_continuous_scale="RdYlGn_r",
                         labels={"IS_DELAYED": "Delay Rate"})
            fig.update_layout(height=320, showlegend=False, coloraxis_showscale=False,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            dow_map = {1:"Mon",2:"Tue",3:"Wed",4:"Thu",5:"Fri",6:"Sat",7:"Sun"}
            dow = fdf.groupby("DAY_OF_WEEK")["IS_DELAYED"].mean().reset_index()
            dow["Day"] = dow["DAY_OF_WEEK"].map(dow_map)
            fig = px.bar(dow, x="Day", y="IS_DELAYED",
                         title="Delay Rate by Day of Week",
                         color="IS_DELAYED", color_continuous_scale="RdYlGn_r",
                         labels={"IS_DELAYED": "Delay Rate"})
            fig.update_layout(height=320, showlegend=False, coloraxis_showscale=False,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            _hourly_df = fdf.copy()
            _hourly_df["DEP_HOUR"] = (_hourly_df["CRS_DEP_TIME"] // 100)
            hourly = _hourly_df.groupby("DEP_HOUR")["IS_DELAYED"].mean().reset_index()
            fig = px.line(hourly, x="DEP_HOUR", y="IS_DELAYED",
                          title="Delay Rate by Departure Hour",
                          markers=True,
                          labels={"DEP_HOUR": "Hour of Day", "IS_DELAYED": "Delay Rate"},
                          color_discrete_sequence=["#e74c3c"])
            fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            yearly = fdf.groupby("YEAR")["IS_DELAYED"].mean().reset_index()
            fig = px.line(yearly, x="YEAR", y="IS_DELAYED",
                          title="Yearly Delay Rate Trend",
                          markers=True,
                          labels={"IS_DELAYED": "Delay Rate"},
                          color_discrete_sequence=["#2980b9"])
            if 2020 in yearly["YEAR"].values:
                fig.add_annotation(x=2020, y=yearly[yearly["YEAR"]==2020]["IS_DELAYED"].values[0],
                                    text="COVID-19", showarrow=True, arrowhead=2, font_color="#e74c3c")
            fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
            st.plotly_chart(fig, use_container_width=True)

        # ── Interactive heatmap: Hour vs Day ─────────────────────────────────
        st.markdown('<div class="section-header">🔥 Delay Heatmap: Hour × Day of Week</div>', unsafe_allow_html=True)
        _hm = fdf.copy()
        _hm["DEP_HOUR"] = _hm["CRS_DEP_TIME"] // 100
        heatmap_data = _hm.groupby(["DAY_OF_WEEK", "DEP_HOUR"])["IS_DELAYED"].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index="DAY_OF_WEEK", columns="DEP_HOUR", values="IS_DELAYED")
        heatmap_pivot.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        fig_hm = px.imshow(heatmap_pivot, color_continuous_scale="RdYlGn_r",
                           title="Delay Rate by Hour & Day (darker = higher delay)",
                           labels={"x": "Departure Hour", "y": "Day", "color": "Delay Rate"},
                           aspect="auto")
        fig_hm.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
        st.plotly_chart(fig_hm, use_container_width=True)

    with tab2:
        airline_stats = fdf.groupby("OP_CARRIER").agg(
            delay_rate=("IS_DELAYED", "mean"),
            avg_delay=("DEP_DELAY", "mean"),
            total_flights=("IS_DELAYED", "count"),
            cancellation_rate=("CANCELLED", "mean"),
        ).reset_index()
        airline_stats["Airline"] = airline_stats["OP_CARRIER"].map({k: v for k, v in AIRLINES.items()})

        # ── Sort control ──────────────────────────────────────────────────────
        sort_by = st.radio("Sort airlines by:", ["Delay Rate", "Avg Delay", "Total Flights", "Cancellation Rate"],
                           horizontal=True)
        sort_map = {"Delay Rate": "delay_rate", "Avg Delay": "avg_delay",
                    "Total Flights": "total_flights", "Cancellation Rate": "cancellation_rate"}
        airline_stats = airline_stats.sort_values(sort_map[sort_by], ascending=False)

        fig = px.scatter(airline_stats, x="delay_rate", y="avg_delay",
                         size="total_flights", color="cancellation_rate",
                         hover_name="Airline",
                         title="Airline Performance: Delay Rate vs Avg Delay",
                         labels={"delay_rate": "Delay Rate", "avg_delay": "Avg Departure Delay (min)",
                                 "cancellation_rate": "Cancellation Rate"},
                         color_continuous_scale="Reds", size_max=40)
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(airline_stats, x="Airline", y=sort_map[sort_by],
                      color=sort_map[sort_by], color_continuous_scale="RdYlGn_r",
                      title=f"{sort_by} by Airline",
                      labels={sort_map[sort_by]: sort_by})
        fig2.update_layout(height=320, coloraxis_showscale=False,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        origin_stats = fdf.groupby("ORIGIN").agg(
            delay_rate=("IS_DELAYED", "mean"),
            total_flights=("IS_DELAYED", "count"),
        ).reset_index()
        origin_stats["Airport"] = origin_stats["ORIGIN"].map({k: v[0] for k, v in AIRPORTS.items()})
        origin_stats["lat"] = origin_stats["ORIGIN"].map({k: v[1] for k, v in AIRPORTS.items()})
        origin_stats["lon"] = origin_stats["ORIGIN"].map({k: v[2] for k, v in AIRPORTS.items()})

        # ── Top N slider ──────────────────────────────────────────────────────
        top_n = st.slider("Show top N airports in bar chart", 5, 20, 10)

        fig = px.scatter_geo(origin_stats, lat="lat", lon="lon",
                             size="total_flights", color="delay_rate",
                             hover_name="Airport",
                             color_continuous_scale="RdYlGn_r", scope="usa",
                             title="US Airport Delay Rates (Bubble = Flight Volume)",
                             labels={"delay_rate": "Delay Rate"})
        fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            top_origins = origin_stats.nlargest(top_n, "delay_rate")
            fig3 = px.bar(top_origins, x="ORIGIN", y="delay_rate",
                          title=f"Top {top_n} Airports by Delay Rate",
                          color="delay_rate", color_continuous_scale="Reds",
                          labels={"delay_rate": "Delay Rate"})
            fig3.update_layout(height=300, coloraxis_showscale=False,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            dist_delay = fdf.copy()
            dist_delay["dist_bin"] = pd.cut(fdf["DISTANCE"], bins=5)
            dd = dist_delay.groupby("dist_bin", observed=False)["IS_DELAYED"].mean().reset_index()
            dd["dist_bin"] = dd["dist_bin"].astype(str)
            fig4 = px.bar(dd, x="dist_bin", y="IS_DELAYED",
                          title="Delay Rate by Distance",
                          color="IS_DELAYED", color_continuous_scale="Blues",
                          labels={"IS_DELAYED": "Delay Rate", "dist_bin": "Distance (miles)"})
            fig4.update_layout(height=300, coloraxis_showscale=False,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
            st.plotly_chart(fig4, use_container_width=True)

    with tab4:
        weather_stats = fdf.groupby("WEATHER_CONDITION").agg(
            delay_rate=("IS_DELAYED", "mean"),
            avg_delay=("DEP_DELAY", "mean"),
            count=("IS_DELAYED", "count"),
        ).reset_index().sort_values("delay_rate", ascending=True)

        fig = px.bar(weather_stats, x="delay_rate", y="WEATHER_CONDITION",
                     orientation="h",
                     title="Delay Rate by Weather Condition",
                     color="delay_rate", color_continuous_scale="RdYlGn_r",
                     labels={"delay_rate": "Delay Rate", "WEATHER_CONDITION": "Weather"})
        fig.update_layout(height=350, coloraxis_showscale=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig2 = px.pie(weather_stats, values="count", names="WEATHER_CONDITION",
                          title="Flight Distribution by Weather",
                          color_discrete_sequence=px.colors.qualitative.Set3)
            fig2.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            weather_delay_types = fdf.groupby("WEATHER_CONDITION")[
                ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "LATE_AIRCRAFT_DELAY"]
            ].mean().reset_index()
            fig3 = px.bar(weather_delay_types, x="WEATHER_CONDITION",
                          y=["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "LATE_AIRCRAFT_DELAY"],
                          title="Avg Delay by Type & Weather",
                          labels={"value": "Avg Delay (min)", "variable": "Delay Type"},
                          barmode="stack")
            fig3.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
            st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Training
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.markdown('<h2 class="page-title">🤖 Machine Learning Model Training</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("""
        **Models trained:**
        - 🌲 **Random Forest** — Ensemble of decision trees, robust to noise
        - 📈 **Gradient Boosting** — Sequential boosting, high accuracy
        - 📐 **Logistic Regression** — Baseline linear classifier

        **Target variable:** `IS_DELAYED` — 1 if departure delay ≥ 15 minutes, else 0

        **Train / Test split:** 80% / 20% (stratified)
        """)

        features = {
            "MONTH": "Month of year (seasonality)",
            "DAY_OF_WEEK": "Day of week (1=Mon…7=Sun)",
            "DAY_OF_MONTH": "Day of month",
            "CRS_DEP_TIME": "Scheduled departure time (HHMM)",
            "DISTANCE": "Flight distance (miles)",
            "CRS_ELAPSED_TIME": "Scheduled flight duration",
            "OP_CARRIER_ENC": "Airline (encoded)",
            "ORIGIN_ENC": "Origin airport (encoded)",
            "DEST_ENC": "Destination airport (encoded)",
            "WEATHER_ENC": "Weather condition (encoded)",
        }
        st.markdown("**Feature Set:**")
        feat_df = pd.DataFrame(list(features.items()), columns=["Feature", "Description"])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    with col2:
        delayed_count = int(df["IS_DELAYED"].sum())
        ontime_count  = int((df["IS_DELAYED"] == 0).sum())
        fig = px.pie(
            values=[delayed_count, ontime_count],
            names=["Delayed", "On-Time"],
            title="Class Distribution",
            color_discrete_sequence=["#e74c3c", "#00b894"],
            hole=0.4,
        )
        fig.update_layout(height=420, legend=dict(font=dict(size=15)),
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div style="background:#1c3a5e;border-radius:8px;padding:1rem 1.2rem;font-size:1.1rem;color:white;">
            📊 Dataset: <b>{len(df):,}</b> flights &nbsp;|&nbsp;
            Delayed: <b>{delayed_count/len(df)*100:.1f}%</b> &nbsp;|&nbsp;
            On-Time: <b>{ontime_count/len(df)*100:.1f}%</b>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        status.info("⚙️ Preprocessing data and encoding features…")
        progress.progress(10)

        status.info("🌲 Training Random Forest…")
        progress.progress(25)
        metrics = st.session_state.predictor.train(df)
        progress.progress(70)

        status.info("📈 Evaluating all models…")
        time.sleep(0.3)
        progress.progress(90)

        status.info("📐 Finalising metrics…")
        time.sleep(0.2)
        progress.progress(100)

        st.session_state.trained = True
        status.success("✅ All 3 models trained successfully!")

        st.markdown('<div class="section-header">Training Results Summary</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for col, (name, m) in zip(cols, metrics.items()):
            with col:
                icon = "🌲" if "Forest" in name else ("📈" if "Boost" in name else "📐")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">{icon} {name}</div>
                    <div class="value">{m['accuracy']*100:.1f}%</div>
                    <div class="delta">AUC-ROC: {m['auc']:.3f}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        gc1, gc2 = st.columns(2)
        with gc1:
            if st.button("🔮 Go to Predict a Flight", use_container_width=True, type="primary"):
                st.session_state["_nav"] = "🔮 Predict a Flight"
                st.rerun()
        with gc2:
            if st.button("📈 View Model Performance", use_container_width=True):
                st.session_state["_nav"] = "📈 Model Performance"
                st.rerun()

    elif st.session_state.trained:
        st.success("✅ Models already trained! Go to **Model Performance** or **Predict a Flight**.")
    else:
        st.warning("⚠️ Click **Train All Models** to begin. Training on 20K records takes ~30 seconds.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Predict a Flight
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict a Flight":
    st.markdown('<h2 class="page-title">🔮 Flight Delay Prediction</h2>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1.2, 1])

    with col_form:
        st.markdown('<div class="section-header">Flight Details</div>', unsafe_allow_html=True)

        airline = st.selectbox("Airline", options=list(AIRLINES.keys()),
                               format_func=lambda x: f"{x} — {AIRLINES[x]}")

        c1, c2 = st.columns(2)
        with c1:
            def _fmt_airport(x):
                name = AIRPORTS[x][0]
                return f"{x} — {name[:30].rsplit(' ', 1)[0] if len(name) > 30 else name}"
            origin = st.selectbox("Origin Airport", options=list(AIRPORTS.keys()),
                                  format_func=_fmt_airport)
        with c2:
            dest_opts = [a for a in AIRPORTS.keys() if a != origin]
            dest = st.selectbox("Destination Airport", options=dest_opts,
                                format_func=_fmt_airport)

        c1, c2 = st.columns(2)
        with c1:
            month = st.selectbox("Month", range(1, 13),
                                 format_func=lambda x: pd.Timestamp(2024, x, 1).strftime("%B"))
        with c2:
            dow_labels = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            day_of_week = st.selectbox("Day of Week", range(1, 8),
                                       format_func=lambda x: dow_labels[x-1])

        c1, c2 = st.columns(2)
        with c1:
            dep_hour = st.slider("Departure Hour", 0, 23, 8)
            dep_min  = st.select_slider("Departure Minute", options=[0, 15, 30, 45])
        with c2:
            day_of_month = st.slider("Day of Month", 1, 31, 15)
            weather = st.selectbox("Weather Condition", WEATHER_CONDITIONS)

        # ── Compare all 3 models toggle ───────────────────────────────────────
        compare_all = st.checkbox("🔁 Compare all 3 models", value=False)
        if not compare_all:
            model_choice = st.selectbox("ML Model", ["Random Forest", "Gradient Boosting", "Logistic Regression"])
        else:
            model_choice = "Random Forest"

        predict_btn = st.button("🔮 Predict Delay", type="primary", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            dist    = _estimate_distance(origin, dest)
            elapsed = max(30, int(dist / 7 + 30))
            dep_time = dep_hour * 100 + dep_min

            input_data = {
                "OP_CARRIER": airline, "ORIGIN": origin, "DEST": dest,
                "MONTH": month, "DAY_OF_WEEK": day_of_week, "DAY_OF_MONTH": day_of_month,
                "CRS_DEP_TIME": dep_time, "DISTANCE": dist,
                "CRS_ELAPSED_TIME": elapsed, "WEATHER_CONDITION": weather,
            }

            if compare_all:
                # ── Side-by-side model comparison ─────────────────────────────
                all_models = ["Random Forest", "Gradient Boosting", "Logistic Regression"]
                results = {}
                with st.spinner("Running all 3 models…"):
                    for m in all_models:
                        results[m] = st.session_state.predictor.predict(input_data, m)

                for m_name, res in results.items():
                    prob = res["delay_probability"]
                    box_class = "delayed-box" if res["is_delayed"] else "ontime-box"
                    icon = "🚨" if res["is_delayed"] else "✅"
                    st.markdown(f"""
                    <div class="model-compare-card">
                        <div style="font-size:0.85rem;color:#aab;margin-bottom:0.3rem">{m_name}</div>
                        <div style="font-size:1.8rem;font-weight:800;color:{'#e74c3c' if res['is_delayed'] else '#00b894'}">{prob*100:.1f}%</div>
                        <div style="font-size:0.9rem;color:#ccc">{icon} {res['risk_level']} risk · ~{res['estimated_delay_minutes']} min</div>
                    </div>""", unsafe_allow_html=True)

                # Comparison bar chart
                fig_cmp = go.Figure(go.Bar(
                    x=list(results.keys()),
                    y=[r["delay_probability"] * 100 for r in results.values()],
                    marker_color=["#e74c3c" if r["is_delayed"] else "#00b894" for r in results.values()],
                    text=[f"{r['delay_probability']*100:.1f}%" for r in results.values()],
                    textposition="outside",
                ))
                fig_cmp.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% threshold")
                fig_cmp.update_layout(title="Delay Probability by Model", height=280,
                                      yaxis_title="Probability (%)", yaxis_range=[0, 110],
                                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      font_color="#ccc")
                st.plotly_chart(fig_cmp, use_container_width=True)

                # Save to history
                st.session_state.pred_history.append({
                    "Route": f"{origin}→{dest}", "Airline": airline,
                    "Weather": weather, "Departure": f"{dep_hour:02d}:{dep_min:02d}",
                    **{m: f"{r['delay_probability']*100:.1f}%" for m, r in results.items()}
                })

            else:
                with st.spinner("Running ML model…"):
                    result = st.session_state.predictor.predict(input_data, model_choice)
                    time.sleep(0.4)

                prob       = result["delay_probability"]
                is_delayed = result["is_delayed"]
                risk       = result["risk_level"]
                est_delay  = result["estimated_delay_minutes"]

                # Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    title={"text": "Delay Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#e74c3c" if is_delayed else "#00b894"},
                        "steps": [
                            {"range": [0, 25],  "color": "#d5f5e3"},
                            {"range": [25, 50], "color": "#fdebd0"},
                            {"range": [50, 75], "color": "#fadbd8"},
                            {"range": [75, 100],"color": "#f1948a"},
                        ],
                        "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.75, "value": 50}
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(t=40, b=0),
                                        paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
                st.plotly_chart(fig_gauge, use_container_width=True)

                box_class = "delayed-box" if is_delayed else "ontime-box"
                label = "🚨 FLIGHT LIKELY DELAYED" if is_delayed else "✅ FLIGHT LIKELY ON TIME"
                sub   = f"Estimated Delay: <b>~{est_delay} minutes</b>" if is_delayed else "Low delay probability"
                st.markdown(f"""
                <div class="prediction-box {box_class}">
                    <div>{label}</div>
                    <div class="prob">{prob*100:.1f}%</div>
                    <div>{sub}</div>
                    <div style="margin-top:0.5rem;font-size:0.9rem">Risk Level: {risk}</div>
                </div>""", unsafe_allow_html=True)

                summary = pd.DataFrame({
                    "Parameter": ["Route", "Distance", "Model Used", "Weather", "Departure"],
                    "Value": [f"{origin} → {dest}", f"{dist:,} miles", model_choice,
                              weather, f"{dep_hour:02d}:{dep_min:02d}"]
                })
                st.markdown("**Flight Summary**")
                st.dataframe(summary, hide_index=True, use_container_width=True)

                st.session_state.pred_history.append({
                    "Route": f"{origin}→{dest}", "Airline": airline,
                    "Weather": weather, "Departure": f"{dep_hour:02d}:{dep_min:02d}",
                    "Model": model_choice, "Probability": f"{prob*100:.1f}%",
                    "Verdict": "Delayed" if is_delayed else "On Time"
                })

        else:
            st.markdown("""
            <div style="background:linear-gradient(135deg,#1a2535,#1e2d40);border-radius:12px;padding:2rem;
                        text-align:center;color:#8ab;min-height:300px;display:flex;flex-direction:column;
                        justify-content:center;border:1px solid rgba(52,152,219,0.2)">
                <div style="font-size:4rem">✈️</div>
                <div style="font-size:1.1rem;margin-top:1rem;color:#ccc">Fill in the flight details and click
                    <b style="color:#3498db">Predict Delay</b></div>
                <div style="font-size:0.85rem;margin-top:0.5rem">or enable <b>Compare all 3 models</b> to see side-by-side results</div>
            </div>""", unsafe_allow_html=True)

    # ── Prediction history ────────────────────────────────────────────────────
    if st.session_state.pred_history:
        st.markdown('<div class="section-header">📋 Prediction History</div>', unsafe_allow_html=True)
        hist_df = pd.DataFrame(st.session_state.pred_history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear History"):
            st.session_state.pred_history = []
            st.rerun()

    # ── Route Map & Live Weather ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🗺️ Flight Route Map &amp; 🌦️ Live Weather Forecast</div>',
                unsafe_allow_html=True)

    # ── Fetch live weather for both airports (cached per route in session) ────
    _wx_key = f"wx_{origin}_{dest}"
    if _wx_key not in st.session_state or st.button("🔄 Refresh Weather", key="refresh_wx"):
        with st.spinner("Fetching live weather from Open-Meteo…"):
            o_wx, d_wx = get_route_weather(origin, dest)
        st.session_state[_wx_key] = (o_wx, d_wx)
    else:
        o_wx, d_wx = st.session_state[_wx_key]

    map_col, wx_col = st.columns([3, 2])

    with map_col:
        if not FOLIUM_AVAILABLE:
            st.warning("📦 Install map libraries to enable the route map:\n```\npip install folium streamlit-folium\n```")
        elif origin not in AIRPORT_COORDS or dest not in AIRPORT_COORDS:
            st.info("ℹ️ Route map is available for major US airports. Select a supported origin and destination.")
        else:
            _last_prob = 0.5
            if st.session_state.pred_history:
                _last_entry = st.session_state.pred_history[-1]
                try:
                    _last_prob = float(_last_entry.get("Probability", "50%").replace("%", "")) / 100
                except Exception:
                    pass

            route_map = draw_route_map(origin, dest, _last_prob, o_wx, d_wx)
            if route_map:
                _route_color = ("🔴 High delay risk" if _last_prob > 0.65
                                else "🟡 Moderate delay risk" if _last_prob > 0.40
                                else "🟢 Low delay risk")
                risk_score = compute_risk_score(o_wx, d_wx)
                st.caption(
                    f"Route: **{origin} → {dest}** · Line colour: {_route_color} · "
                    f"Weather Risk Score: **{risk_score}/10** · "
                    f"_Click airport markers for live weather_"
                )
                st_folium(route_map, width=700, height=420, returned_objects=[])
            else:
                st.info("Select valid airports to display the route map.")

    with wx_col:
        def _render_wx_card(wx: dict, label_emoji: str, label: str):
            iata       = wx["airport"]
            icon       = WEATHER_ICON_MAP.get(wx.get("weather_code", 0), "🌡️")
            wind_kph   = round(wx["wind_speed_ms"] * 3.6, 1)
            vis_km     = round(wx["visibility_m"] / 1000, 1)
            live_badge = (
                '<span style="background:#2ecc71;color:#000;font-size:0.68rem;'
                'padding:1px 6px;border-radius:4px;font-weight:700;margin-left:6px">LIVE</span>'
                if wx.get("live") else
                '<span style="background:#e74c3c;color:#fff;font-size:0.68rem;'
                'padding:1px 6px;border-radius:4px;font-weight:700;margin-left:6px">OFFLINE</span>'
            )

            # Weather risk for this single airport
            _single_risk = compute_risk_score(wx, wx)   # same airport both sides → /2 by design
            risk_colour  = ("#e74c3c" if _single_risk > 6
                            else "#f39c12" if _single_risk > 3
                            else "#2ecc71")
            risk_label   = ("High" if _single_risk > 6
                            else "Moderate" if _single_risk > 3
                            else "Low")

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a2535,#1e2d40);border-radius:12px;
                        padding:1rem 1.2rem;margin-bottom:0.9rem;
                        border-left:4px solid {risk_colour}">
              <div style="font-size:0.78rem;color:#aab;text-transform:uppercase;letter-spacing:0.5px">
                {label_emoji} {label} — {iata} {live_badge}
              </div>
              <div style="font-size:1.6rem;margin:0.35rem 0">
                {icon} <span style="color:#fff;font-weight:700;font-size:1rem">{wx['weather_desc']}</span>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.35rem;
                          font-size:0.87rem;color:#ccc;margin-top:0.4rem">
                <div>🌡️ <b>{wx['temperature_c']}°C</b> (feels {wx['feels_like_c']}°C)</div>
                <div>💨 <b>{wind_kph} km/h</b></div>
                <div>👁️ Vis <b>{vis_km} km</b></div>
                <div>💧 Humidity <b>{wx['humidity_pct']}%</b></div>
                <div>🌧️ Precip <b>{wx['precipitation']} mm</b></div>
                <div>☁️ Clouds <b>{wx['clouds_pct']}%</b></div>
              </div>
              <div style="margin-top:0.75rem;background:{risk_colour};border-radius:8px;
                          padding:0.35rem 0.8rem;text-align:center;color:white;
                          font-weight:700;font-size:0.88rem">
                Weather Delay Risk: {risk_label} ({_single_risk}/10)
              </div>
              <div style="font-size:0.72rem;color:#667;margin-top:0.4rem;text-align:right">
                Updated {wx['fetched_at']} · Open-Meteo
              </div>
            </div>""", unsafe_allow_html=True)

        _render_wx_card(o_wx, "🛫", "Departure")
        _render_wx_card(d_wx, "🛬", "Arrival")

        # ── Combined route risk summary ────────────────────────────────────────
        route_risk   = compute_risk_score(o_wx, d_wx)
        route_colour = "#e74c3c" if route_risk > 6 else "#f39c12" if route_risk > 3 else "#2ecc71"
        route_label  = "High" if route_risk > 6 else "Moderate" if route_risk > 3 else "Low"
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a2535,#1e2d40);border-radius:12px;
                    padding:0.9rem 1.2rem;border:2px solid {route_colour};text-align:center">
          <div style="font-size:0.78rem;color:#aab;text-transform:uppercase;letter-spacing:0.5px">
            Combined Route Weather Risk
          </div>
          <div style="font-size:2.2rem;font-weight:800;color:{route_colour};margin:0.3rem 0">
            {route_risk}/10
          </div>
          <div style="color:{route_colour};font-weight:700">{route_label} Risk</div>
          <div style="font-size:0.78rem;color:#888;margin-top:0.3rem">
            Factoring wind, visibility, precipitation &amp; conditions at both airports
          </div>
        </div>""", unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Performance
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown('<h2 class="page-title">📈 Model Performance & Evaluation</h2>', unsafe_allow_html=True)

    metrics = st.session_state.predictor.metrics

    # Summary table
    st.markdown('<div class="section-header">Performance Summary</div>', unsafe_allow_html=True)
    summary_rows = []
    for name, m in metrics.items():
        r = m["report"]
        summary_rows.append({
            "Model": name,
            "Accuracy": f"{m['accuracy']*100:.2f}%",
            "AUC-ROC": f"{m['auc']:.4f}",
            "Precision (Delayed)": f"{r.get('1', r.get(1,{})).get('precision', 0):.4f}",
            "Recall (Delayed)": f"{r.get('1', r.get(1,{})).get('recall', 0):.4f}",
            "F1-Score (Delayed)": f"{r.get('1', r.get(1,{})).get('f1-score', 0):.4f}",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        names = list(metrics.keys())
        accs  = [m["accuracy"] * 100 for m in metrics.values()]
        aucs  = [m["auc"] for m in metrics.values()]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Accuracy (%)", x=names, y=accs, marker_color="#3498db"))
        fig.add_trace(go.Bar(name="AUC-ROC (%)", x=names, y=[a * 100 for a in aucs], marker_color="#e74c3c"))
        fig.update_layout(barmode="group", title="Model Comparison: Accuracy & AUC-ROC",
                          height=320, yaxis_title="Score (%)",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        colors = ["#3498db", "#e74c3c", "#2ecc71"]
        for (name, m), color in zip(metrics.items(), colors):
            fpr, tpr = m["roc"]
            fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                      name=f"{name} (AUC={m['auc']:.3f})",
                                      line=dict(color=color, width=2)))
        fig2.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                  name="Random", line=dict(dash="dash", color="gray")))
        fig2.update_layout(title="ROC Curves", xaxis_title="False Positive Rate",
                           yaxis_title="True Positive Rate", height=320,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
        st.plotly_chart(fig2, use_container_width=True)

    # Confusion matrices
    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (name, m) in zip(cols, metrics.items()):
        with col:
            cm = m["confusion_matrix"]
            fig3 = px.imshow(cm, text_auto=True,
                             x=["On-Time","Delayed"], y=["On-Time","Delayed"],
                             color_continuous_scale="Blues", title=name,
                             labels={"x":"Predicted","y":"Actual"})
            fig3.update_layout(height=280, coloraxis_showscale=False,
                               paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
            st.plotly_chart(fig3, use_container_width=True)

    # Feature importances with interactive model selector
    fi = st.session_state.predictor.feature_importances
    if fi:
        st.markdown('<div class="section-header">Feature Importances</div>', unsafe_allow_html=True)
        fi_cols = st.columns([2, 1])
        with fi_cols[0]:
            model_name = st.selectbox("Select model", list(fi.keys()))
        with fi_cols[1]:
            chart_type = st.radio("Chart type", ["Horizontal Bar", "Treemap"], horizontal=True)

        fi_df = pd.DataFrame(list(fi[model_name].items()),
                             columns=["Feature", "Importance"]).sort_values("Importance")

        if chart_type == "Horizontal Bar":
            fig4 = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                          title=f"Feature Importances — {model_name}",
                          color="Importance", color_continuous_scale="Blues")
            fig4.update_layout(height=350, coloraxis_showscale=False,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
        else:
            fig4 = px.treemap(fi_df, path=["Feature"], values="Importance",
                              title=f"Feature Importances — {model_name}",
                              color="Importance", color_continuous_scale="Blues")
            fig4.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc")

        st.plotly_chart(fig4, use_container_width=True)

