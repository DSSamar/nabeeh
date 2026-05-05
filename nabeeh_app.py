"""
NABEEH — Smart Energy Management System
Streamlit Deployment App
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
from datetime import datetime
import holidays

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NABEEH — Smart Energy Management",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Montserrat:wght@300;400;500;600;700&display=swap');

/* Dark theme */
:root {
    --bg: #0B132B;
    --card: #1C2541;
    --accent: #00D4FF;
    --purple: #7B2CBF;
    --green: #10b981;
    --yellow: #f59e0b;
    --red: #ef4444;
    --text: #ffffff;
    --muted: rgba(255,255,255,0.55);
}

.stApp {
    background: var(--bg) !important;
    font-family: 'Montserrat', sans-serif;
}

/* Header */
.nabeeh-header {
    background: linear-gradient(135deg, #0B132B 0%, #1C2541 60%, #0d1a3a 100%);
    border-radius: 20px;
    padding: 36px 48px;
    margin-bottom: 28px;
    border: 1px solid rgba(0,212,255,0.15);
    position: relative;
    overflow: hidden;
}
.nabeeh-header::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 300px; height: 300px; border-radius: 50%;
    background: radial-gradient(circle, rgba(0,212,255,0.08) 0%, transparent 70%);
}
.nabeeh-title {
    font-family: 'Orbitron', monospace;
    font-size: 52px; font-weight: 900;
    background: linear-gradient(90deg, #00D4FF, #ffffff, #7B2CBF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: 8px; margin: 0;
}
.nabeeh-subtitle {
    font-size: 16px; color: rgba(255,255,255,0.6);
    letter-spacing: 3px; text-transform: uppercase;
    margin-top: 8px; font-weight: 300;
}
.nabeeh-tagline {
    font-size: 13px; color: #00D4FF;
    margin-top: 6px; font-style: italic;
}

/* Cards */
.card {
    background: var(--card);
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(0,212,255,0.12);
    margin-bottom: 16px;
}
.card-title {
    font-family: 'Orbitron', monospace;
    font-size: 13px; font-weight: 700;
    color: var(--accent); letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 16px;
}

/* Result metric */
.metric-big {
    background: linear-gradient(135deg, #1C2541, #0B132B);
    border-radius: 16px; padding: 24px;
    border: 1px solid rgba(0,212,255,0.2);
    text-align: center;
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 36px; font-weight: 900;
    color: var(--accent);
}
.metric-unit  { font-size: 14px; color: var(--muted); margin-top: 4px; }
.metric-label { font-size: 12px; color: var(--muted); letter-spacing: 2px; text-transform: uppercase; margin-top: 6px; }

/* ANFIS badge */
.badge-no     { background: rgba(16,185,129,0.15); border: 1px solid #10b981; color: #10b981; border-radius: 30px; padding: 8px 24px; font-weight: 700; font-size: 18px; display:inline-block; }
.badge-mild   { background: rgba(245,158,11,0.15); border: 1px solid #f59e0b; color: #f59e0b; border-radius: 30px; padding: 8px 24px; font-weight: 700; font-size: 18px; display:inline-block; }
.badge-strong { background: rgba(239,68,68,0.15);  border: 1px solid #ef4444; color: #ef4444;  border-radius: 30px; padding: 8px 24px; font-weight: 700; font-size: 18px; display:inline-block; }

/* Weather card */
.weather-row {
    display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0;
}
.w-item {
    background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 12px; padding: 14px 20px;
    text-align: center; flex: 1; min-width: 80px;
}
.w-val   { font-size: 22px; font-weight: 700; color: white; }
.w-label { font-size: 10px; color: var(--muted); letter-spacing: 1px; margin-top: 4px; }

/* Divider */
.divider { border: none; border-top: 1px solid rgba(0,212,255,0.1); margin: 20px 0; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00D4FF, #0066aa) !important;
    color: #0B132B !important; font-weight: 800 !important;
    font-family: 'Montserrat', sans-serif !important;
    border: none !important; border-radius: 12px !important;
    padding: 14px 32px !important; font-size: 15px !important;
    letter-spacing: 1px !important; width: 100%;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0,212,255,0.3) !important;
}

/* Input labels */
label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color: rgba(255,255,255,0.7) !important;
    font-size: 13px !important; font-weight: 500 !important;
}

/* Hide streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}

/* Section headers */
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 14px; font-weight: 700;
    color: var(--accent); letter-spacing: 3px;
    text-transform: uppercase;
    padding: 0 0 12px 0;
    border-bottom: 1px solid rgba(0,212,255,0.15);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

OPENWEATHER_API_KEY = "943986a004ca2d335f6c8fa00513286b"  # ← Replace with your free key from openweathermap.org

def get_weather(city: str) -> dict | None:
    """Fetch current weather from OpenWeatherMap API."""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        resp = requests.get(url, timeout=30) 
        if resp.status_code == 200:
            data = resp.json()
            return {
                "temp"     : round(data["main"]["temp"], 1),
                "humidity" : data["main"]["humidity"],
                "wind"     : round(data["wind"]["speed"], 1),
                "desc"     : data["weather"][0]["description"].title(),
                "icon"     : data["weather"][0]["main"],
                "clouds"   : data["clouds"]["all"],        # % cloud cover 0-100
                "lat"      : data["coord"]["lat"],          # latitude
                "lon"      : data["coord"]["lon"],          # longitude
            }
        else:
            st.error(f"API Error {resp.status_code}: {resp.json().get('message','Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None


def get_weather_icon(desc: str) -> str:
    icons = {"Clear":"☀️","Clouds":"☁️","Rain":"🌧️","Snow":"❄️","Thunderstorm":"⛈️","Drizzle":"🌦️","Mist":"🌫️","Fog":"🌫️"}
    return icons.get(desc, "🌤️")


def calculate_strglo(lat: float, hour: int, clouds_pct: int) -> float:
    """
    Estimate solar irradiance StrGlo [W/m²] from:
    - lat       : latitude of the city
    - hour      : current hour (0-23)
    - clouds_pct: cloud cover % from OpenWeatherMap (0-100)

    Formula:
    1. Solar elevation angle based on hour (simplified)
    2. Clear-sky irradiance × (1 - cloud_factor)
    """
    # Solar elevation: simplified sinusoidal peak at solar noon (hour=12)
    import math
    # Hour angle from solar noon
    hour_angle = abs(hour - 12)
    # Max elevation depends on latitude — higher lat = lower sun
    max_elev = max(5, 90 - abs(lat))
    # Elevation follows cosine from noon
    elevation = max_elev * math.cos(math.radians(hour_angle * 7.5))
    elevation = max(0, elevation)  # can't be negative (nighttime)

    # Clear-sky irradiance at this elevation
    clear_sky = 1000 * math.sin(math.radians(elevation))
    clear_sky = max(0, clear_sky)

    # Cloud reduction factor
    cloud_factor = clouds_pct / 100.0
    strglo = clear_sky * (1 - 0.75 * cloud_factor)

    return round(strglo, 1)


def is_off_day(dt: datetime) -> int:
    ch_holidays = holidays.Switzerland(prov="ZH", years=dt.year)
    return int(dt.weekday() >= 5 or dt in ch_holidays)


def get_tou(hour: int) -> int:
    if 10 <= hour <= 16:               return 2
    elif 6 <= hour < 10 or 17 <= hour <= 19: return 1
    else:                              return 0


def cyclic(val, max_val):
    return np.sin(2 * np.pi * val / max_val), np.cos(2 * np.pi * val / max_val)


def build_feature_row(temp, humidity, wind, dt: datetime) -> np.ndarray:
    hour_sin, hour_cos   = cyclic(dt.hour, 24)
    dow_sin,  dow_cos    = cyclic(dt.weekday(), 7)
    month_sin, month_cos = cyclic(dt.month, 12)
    off_day = is_off_day(dt)
    
    # ترتيب الـ features يطابق التدريب بالضبط:
    # T_degC, Hr_pctHr, WVs_m_s, hour_sin, hour_cos,
    # dow_sin, dow_cos, month_sin, month_cos, is_off_day
    return np.array([[
        temp,       # T_degC
        humidity,   # Hr_pctHr
        wind,       # WVs_m_s
        hour_sin,   # hour_sin
        hour_cos,   # hour_cos
        dow_sin,    # dow_sin
        dow_cos,    # dow_cos
        month_sin,  # month_sin
        month_cos,  # month_cos
        off_day     # is_off_day
    ]], dtype=np.float32)


def predict_consumption(X: np.ndarray) -> float:
    if os.path.exists("xgboost_nabeeh.json"):
        import pandas as pd
        import json
        from xgboost import XGBRegressor

        with open("feature_names.json", "r") as f:
            feature_names = json.load(f)

        X_df = pd.DataFrame(X, columns=feature_names)
        model = XGBRegressor()
        model.load_model("xgboost_nabeeh.json")
        return float(model.predict(X_df)[0])
    else:
        base = 180000
        temp_effect = (X[0][0] - 10) * 800
        off = X[0][9]
        demo = base + temp_effect - (off * 40000) + np.random.normal(0, 3000)
        return max(50000, demo)

def solar_pv(strglo: float, p_rated=100, pr=0.80) -> float:
    """E_PV in Wh."""
    return p_rated * 1000 * pr * (strglo / 1000.0)


def carbon(consumption_wh: float, e_pv_wh: float) -> tuple:
    EF = 0.000039  # kg CO2 / Wh  =  39 g/kWh (Swiss grid, Our World in Data / Ember 2026 
    co2_base    = consumption_wh * EF
    co2_pv      = max(0, consumption_wh - e_pv_wh) * EF
    co2_saved   = co2_base - co2_pv
    return co2_base, co2_pv, co2_saved


def anfis_recommendation(consumption: float, pv_share: float, hour: int, tou: int, temp: float) -> tuple:
    """Rule-based ANFIS recommendation (mirrors the trained model logic)."""
    score = 0
    if consumption > 200000: score += 2
    elif consumption > 150000: score += 1
    if pv_share < 0.1: score += 1
    if tou == 2: score += 2
    elif tou == 1: score += 1
    if temp > 30: score += 1

    if score >= 4:   return 2, "Strong Action ⚡", "badge-strong", "Immediately reduce non-essential loads. Shift heavy equipment to off-peak hours."
    elif score >= 2: return 1, "Mild Action ⚠️",   "badge-mild",   "Consider reducing lighting and HVAC intensity. Monitor consumption closely."
    else:            return 0, "No Action ✅",      "badge-no",     "Energy usage is optimal. No intervention needed at this time."


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="nabeeh-header">
    <p class="nabeeh-title">NABEEH</p>
    <p class="nabeeh-subtitle">Smart Energy Management System</p>
    <p class="nabeeh-tagline">"Predicting Energy. Powering the Future." · نبيه</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT — Two columns
# ═══════════════════════════════════════════════════════════════════════════════
col_input, col_results = st.columns([1, 1.6], gap="large")

# ───────────────────────────────────────────────────────────────────────────────
# LEFT COLUMN — INPUTS
# ───────────────────────────────────────────────────────────────────────────────
with col_input:

    # ── Weather section ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📍 Location & Time</div>', unsafe_allow_html=True)

    city = st.text_input("City", value="Zurich", placeholder="Enter city name...")

    col_d, col_t = st.columns(2)
    with col_d:
        selected_date = st.date_input("Date", value=datetime.now().date())
    with col_t:
        selected_hour = st.slider("Hour", 0, 23, datetime.now().hour)

    selected_dt = datetime.combine(selected_date, datetime.min.time()).replace(hour=selected_hour)

    # Fetch weather button
    fetch_btn = st.button("🌤️  Fetch Weather Automatically", use_container_width=True)

    # Weather state
    if "weather" not in st.session_state:
        st.session_state.weather = None

    if fetch_btn:
        with st.spinner("Fetching weather..."):
            w = get_weather(city)
            if w:
                st.session_state.weather = w
                st.success(f"Weather fetched for {city}!")
            else:
                st.warning("Could not fetch weather — please enter manually below or check your API key.")

    # Show fetched weather
    if st.session_state.weather:
        w = st.session_state.weather
        icon = get_weather_icon(w["icon"])
        st.markdown(f"""
        <div class="card">
            <div class="card-title">{icon} Current Weather — {city}</div>
            <div class="weather-row">
                <div class="w-item"><div class="w-val">{w['temp']}°C</div><div class="w-label">Temperature</div></div>
                <div class="w-item"><div class="w-val">{w['humidity']}%</div><div class="w-label">Humidity</div></div>
                <div class="w-item"><div class="w-val">{w['wind']} m/s</div><div class="w-label">Wind</div></div>
                <div class="w-item"><div class="w-val">{w['clouds']}%</div><div class="w-label">Cloud Cover</div></div>
                <div class="w-item"><div class="w-val" style="font-size:14px">{w['desc']}</div><div class="w-label">Condition</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        temp_val     = w["temp"]
        humidity_val = w["humidity"]
        wind_val     = w["wind"]
    else:
        st.markdown('<div class="section-header" style="margin-top:16px">🔧 Manual Input</div>', unsafe_allow_html=True)
        temp_val     = st.number_input("🌡️ Temperature (°C)",  value=12.0, min_value=-20.0, max_value=45.0, step=0.5)
        humidity_val = st.number_input("💧 Humidity (%)",       value=70.0, min_value=0.0,  max_value=100.0, step=1.0)
        wind_val     = st.number_input("💨 Wind Speed (m/s)",   value=3.0,  min_value=0.0,  max_value=30.0,  step=0.1)

    # ── Solar note ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);
    border-radius:10px;padding:12px 16px;margin-top:8px">
        <span style="color:#f59e0b;font-size:12px;font-weight:600">☀️ Solar irradiance is calculated automatically
        from your city's location, time, and cloud cover.</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Predict button ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  Run NABEEH Prediction", use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN — RESULTS
# ───────────────────────────────────────────────────────────────────────────────
with col_results:

    if predict_btn:

        # Build features
        X = build_feature_row(temp_val, humidity_val, wind_val, selected_dt)
        tou = get_tou(selected_hour)
        off = is_off_day(selected_dt)

        # Auto-calculate StrGlo from weather data
        if st.session_state.weather:
            w = st.session_state.weather
            strglo = calculate_strglo(w["lat"], selected_hour, w["clouds"])
        else:
            strglo = calculate_strglo(47.3769, selected_hour, 50)  # Default: Zurich coords

        # Predict
        with st.spinner("Running predictions..."):
            consumption = predict_consumption(X)
            e_pv        = solar_pv(strglo)
            pv_share    = min(e_pv / max(consumption, 1), 1.0)
            co2_base, co2_pv, co2_saved = carbon(consumption, e_pv)
            rec_score, rec_label, rec_class, rec_advice = anfis_recommendation(
                consumption, pv_share, selected_hour, tou, temp_val
            )

        # ── Context bar ─────────────────────────────────────────────────────
        tou_labels = {0:"Off-Peak 🟢", 1:"Part-Peak 🟡", 2:"Peak 🔴"}
        off_label  = "🏖️ Off Day" if off else "🏢 Working Day"
        st.markdown(f"""
        <div class="card" style="padding:16px 24px">
            <div style="display:flex;gap:24px;flex-wrap:wrap;align-items:center">
                <div><span style="color:var(--muted);font-size:11px;letter-spacing:1px">DATE & TIME</span><br>
                     <span style="color:white;font-weight:700">{selected_dt.strftime('%a, %b %d %Y  %H:00')}</span></div>
                <div><span style="color:var(--muted);font-size:11px;letter-spacing:1px">TOU</span><br>
                     <span style="color:white;font-weight:700">{tou_labels[tou]}</span></div>
                <div><span style="color:var(--muted);font-size:11px;letter-spacing:1px">DAY TYPE</span><br>
                     <span style="color:white;font-weight:700">{off_label}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 4 metric cards ───────────────────────────────────────────────────
        m1, m2 = st.columns(2)

        with m1:
            st.markdown(f"""
            <div class="metric-big">
                <div style="font-size:28px;margin-bottom:6px">⚡</div>
                <div class="metric-value">{consumption/1000:.1f}</div>
                <div class="metric-unit">kWh</div>
                <div class="metric-label">Predicted Consumption</div>
            </div>""", unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-big">
                <div style="font-size:28px;margin-bottom:6px">☀️</div>
                <div class="metric-value" style="color:#f59e0b">{e_pv/1000:.1f}</div>
                <div class="metric-unit">kWh</div>
                <div class="metric-label">Solar PV Generation</div>
            </div>""", unsafe_allow_html=True)

        m3, m4 = st.columns(2)
        with m3:
            st.markdown(f"""
            <div class="metric-big">
                <div style="font-size:28px;margin-bottom:6px">🌿</div>
                <div class="metric-value" style="color:#10b981">{co2_saved:.2f}</div>
                <div class="metric-unit">kg CO₂ saved</div>
                <div class="metric-label">Carbon Reduction</div>
            </div>""", unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-big">
                <div style="font-size:28px;margin-bottom:6px">🔆</div>
                <div class="metric-value" style="color:#7B2CBF">{pv_share*100:.1f}%</div>
                <div class="metric-unit">of consumption</div>
                <div class="metric-label">PV Share</div>
            </div>""", unsafe_allow_html=True)

        # ── Merged Interpretation Card ───────────────────────────────────────
        avg_consumption = 150000
        diff_pct = ((consumption - avg_consumption) / avg_consumption) * 100
        diff_label = ("📈 " + str(round(diff_pct,1)) + "% above") if diff_pct > 0 else ("📉 " + str(round(abs(diff_pct),1)) + "% below")
        diff_color = "#ef4444" if diff_pct > 10 else "#10b981" if diff_pct < -10 else "#f59e0b"
        office_rooms_str = "{:,}".format(int(consumption / 500))
        ac_units_str     = "{:,}".format(int(consumption / 1500))
        computers_str    = "{:,}".format(int(consumption / 150))

        if rec_score == 2:
            ac = "#ef4444"; ai = "🔴"; at = "Action Needed Now!"
            atxt = "Consumption is significantly above average. Turn off lights in empty rooms, reduce air conditioning, and delay heavy equipment until off-peak hours."
        elif rec_score == 1:
            ac = "#f59e0b"; ai = "🟡"; at = "Stay Alert"
            atxt = "Energy usage is moderate. Consider reducing HVAC intensity slightly. Avoid starting high-power equipment for the next hour."
        else:
            ac = "#10b981"; ai = "🟢"; at = "All Good!"
            atxt = "Energy usage is efficient right now. No action needed. Great time to run heavy equipment or charge devices."

        smsg = "excellent — most of your energy is from the sun!" if pv_share > 0.3 else "good — some free energy from the sun." if pv_share > 0.05 else "minimal due to low sunlight or cloud cover."
        ac_days_str = str(max(1, int(co2_saved / 40)))
        pv_pct_str = str(int(pv_share * 100))
        cons_kw_str = "{:.0f}".format(consumption / 1000)

        merged_html = (
            '<div style="background:linear-gradient(135deg,#1C2541,#0B132B);border-radius:20px;'
            'padding:32px 40px;border:1px solid rgba(0,212,255,0.15);margin-top:8px">'

            # Title
            '<div style="font-family:Orbitron,monospace;font-size:13px;font-weight:700;color:#00D4FF;'
            'letter-spacing:3px;text-transform:uppercase;margin-bottom:8px">💬 What Does This Mean?</div>'

            # Compared to average
            '<div style="margin-bottom:20px;padding:12px 16px;background:rgba(0,0,0,0.2);border-radius:10px">'
            '<div style="color:rgba(255,255,255,0.4);font-size:10px;letter-spacing:2px;margin-bottom:4px">COMPARED TO AVERAGE</div>'
            '<div style="font-size:16px;font-weight:700;color:' + diff_color + '">' + diff_label + ' the typical hourly consumption for this building type</div>'
            '</div>'

            # 3 main cards
            '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:20px">'

            # Energy Usage card
            '<div style="background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.15);border-radius:14px;padding:20px;text-align:center">'
            '<div style="font-size:32px;margin-bottom:8px">🏢</div>'
            '<div style="color:white;font-size:14px;font-weight:600;margin-bottom:6px">Energy Usage</div>'
            '<div style="color:rgba(255,255,255,0.6);font-size:12px;line-height:1.6">'
            'About <strong style="color:#00D4FF">' + cons_kw_str + ' kWh</strong> this hour<br>'
            '≈ <strong style="color:#00D4FF">' + office_rooms_str + '</strong> office rooms<br>'
            '<span style="font-size:10px;color:rgba(255,255,255,0.3)">~500 Wh/room · ASHRAE</span>'
            '</div></div>'

            # Solar Power card
            '<div style="background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.15);border-radius:14px;padding:20px;text-align:center">'
            '<div style="font-size:32px;margin-bottom:8px">☀️</div>'
            '<div style="color:white;font-size:14px;font-weight:600;margin-bottom:6px">Solar Power</div>'
            '<div style="color:rgba(255,255,255,0.6);font-size:12px;line-height:1.6">'
            'Covering <strong style="color:#f59e0b">' + pv_pct_str + '%</strong> of your needs<br>'
            '≈ <strong style="color:#f59e0b">' + ac_units_str + '</strong> AC units offset<br>'
            '<span style="font-size:10px;color:rgba(255,255,255,0.3)">~1,500 Wh/unit · ASHRAE</span>'
            '</div></div>'

            # CO2 Saved card
            '<div style="background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.15);border-radius:14px;padding:20px;text-align:center">'
            '<div style="font-size:32px;margin-bottom:8px">🌿</div>'
            '<div style="color:white;font-size:14px;font-weight:600;margin-bottom:6px">CO₂ Saved</div>'
            '<div style="color:rgba(255,255,255,0.6);font-size:12px;line-height:1.6">'
            '<strong style="color:#10b981">' + "{:.1f}".format(co2_saved) + ' kg</strong> CO₂ avoided<br>'
            'Like turning off <strong style="color:#10b981">' + computers_str + '</strong> computers<br>'
            '<span style="font-size:10px;color:rgba(255,255,255,0.3)">~150 Wh/unit · Energy Star</span>'
            '</div></div>'

            '</div>'

            # ANFIS Action bar
            '<div style="background:rgba(255,255,255,0.03);border-left:4px solid ' + ac + ';border-radius:0 14px 14px 0;padding:20px 24px">'
            '<div style="font-size:16px;font-weight:700;color:' + ac + ';margin-bottom:8px">' + ai + ' ' + at + '</div>'
            '<div style="color:rgba(255,255,255,0.7);font-size:13px;line-height:1.7">' + atxt + '</div>'
            '</div>'

            '</div>'
        )
        st.markdown(merged_html, unsafe_allow_html=True)

        # ── ANFIS Recommendation ─────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🧠 ANFIS Recommendation</div>
            <div style="text-align:center;margin:12px 0">
                <span class="{rec_class}">{rec_label}</span>
            </div>
            <p style="color:rgba(255,255,255,0.7);font-size:14px;text-align:center;margin-top:12px">
                {rec_advice}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Carbon Footprint Breakdown ────────────────────────────────────────
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🌿 Carbon Footprint Breakdown</div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;text-align:center">
                <div>
                    <div style="font-size:20px;font-weight:800;color:#ef4444">{co2_base:.2f}</div>
                    <div style="font-size:11px;color:var(--muted);margin-top:4px">kg CO₂ Baseline</div>
                </div>
                <div>
                    <div style="font-size:20px;font-weight:800;color:#f59e0b">{co2_pv:.2f}</div>
                    <div style="font-size:11px;color:var(--muted);margin-top:4px">kg CO₂ With PV</div>
                </div>
                <div>
                    <div style="font-size:20px;font-weight:800;color:#10b981">{co2_saved:.2f}</div>
                    <div style="font-size:11px;color:var(--muted);margin-top:4px">kg CO₂ Saved</div>
                </div>
            </div>
            <div style="margin-top:16px;background:rgba(0,0,0,0.2);border-radius:8px;height:8px;overflow:hidden">
                <div style="height:100%;width:{pv_share*100:.0f}%;background:linear-gradient(90deg,#10b981,#00D4FF);border-radius:8px"></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:4px">
                <span style="font-size:10px;color:var(--muted)">Grid CO₂</span>
                <span style="font-size:10px;color:#10b981">{pv_share*100:.1f}% offset by solar</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Placeholder before prediction
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;color:rgba(255,255,255,0.3)">
            <div style="font-size:72px;margin-bottom:24px">🔋</div>
            <div style="font-family:'Orbitron',monospace;font-size:16px;letter-spacing:3px">
                AWAITING INPUT
            </div>
            <div style="font-size:13px;margin-top:12px;max-width:300px;margin-left:auto;margin-right:auto">
                Enter location & time on the left, then click <strong style="color:#00D4FF">Run NABEEH Prediction</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:rgba(0,212,255,0.1);margin-top:32px">
<div style="text-align:center;padding:16px;color:rgba(255,255,255,0.25);font-size:11px;letter-spacing:2px">
    NABEEH · Smart Energy Management System · PNU · CCIS · Data Science · 1447–1448H<br>
    <span style="color:rgba(0,212,255,0.4)">Supervised by Dr. Zuhaira Muhammad Zain</span>
</div>
""", unsafe_allow_html=True)
