import subprocess
import sys
import time
import re
from IPython.display import display, HTML

# =============================================================================
# 1. INSTALL CLOUDFLARED BINARY & PYTHON DEPENDENCIES
# =============================================================================
print("Installing cloudflared binary on Colab Linux environment...")
subprocess.check_call("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb", shell=True)
subprocess.check_call("dpkg -i cloudflared-linux-amd64.deb", shell=True)

print("Installing required Python dependencies...")
dependencies = [
    "streamlit",
    "plotly",
    "xgboost",
    "pandas",
    "numpy",
    "scikit-learn",
    "requests",
    "pydeck",
    "feedparser"
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + dependencies)
print("All system and Python packages installed successfully.\n")

# =============================================================================
# 2. WRITE STREAMLIT APP CODE TO app.py
# =============================================================================
app_code = r'''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pydeck as pdk
from datetime import datetime, timedelta
import feedparser
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FreightSense AI · India Road Freight Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0b0e14; color: #e6edf3; }
    
    /* Metric Card Styling */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    .metric-label { font-size: 0.85rem; color: #8b949e; text-transform: uppercase; font-weight: 600; }
    .metric-value { font-size: 1.9rem; font-weight: 700; color: #58a6ff; margin: 0.3rem 0; }
    .metric-sub { font-size: 0.8rem; color: #7d8590; }
    
    /* Risk Badges */
    .badge-low { background: #13231b; color: #3fb950; border: 1px solid #238636; padding: 4px 10px; border-radius: 20px; font-weight: 600; }
    .badge-medium { background: #272015; color: #d29922; border: 1px solid #9e6a03; padding: 4px 10px; border-radius: 20px; font-weight: 600; }
    .badge-high { background: #2e1518; color: #f85149; border: 1px solid #da3633; padding: 4px 10px; border-radius: 20px; font-weight: 600; }

    /* Zone Detail Box */
    .zone-box {
        background-color: #161b22;
        border-left: 5px solid #58a6ff;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 25px;
    }
    .news-box {
        background: #11161d;
        border: 1px solid #21262d;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# INDIAN CORRIDORS & ZONE DATA
# -----------------------------------------------------------------------------
INDIAN_CORRIDORS = {
    "Delhi NCR → Mumbai (NH48 Golden Quadrilateral)": {
        "zone": "North-West Corridor",
        "base_demand": 750, "dist_km": 1411,
        "lat1": 28.6139, "lon1": 77.2090, "lat2": 19.0760, "lon2": 72.8777,
        "weather_default": 2.2, "fuel_idx": 94.5,
        "cargo": "Textiles, Auto Components, FMCG, Electronics",
        "description": "Primary economic engine connecting Delhi-NCR to JNPT Port & Mumbai financial hub."
    },
    "Mumbai → Bengaluru (NH48 Industrial Belt)": {
        "zone": "West-South Corridor",
        "base_demand": 620, "dist_km": 980,
        "lat1": 19.0760, "lon1": 72.8777, "lat2": 12.9716, "lon2": 77.5946,
        "weather_default": 3.8, "fuel_idx": 102.1,
        "cargo": "Pharmaceuticals, Tech Hardware, Engineering Goods",
        "description": "High-density manufacturing route linking Western ports to South India tech manufacturing hubs."
    },
    "Kolkata → Delhi (NH19 Eastern Freight Way)": {
        "zone": "East-North Corridor",
        "base_demand": 540, "dist_km": 1530,
        "lat1": 22.5726, "lon1": 88.3639, "lat2": 28.6139, "lon2": 77.2090,
        "weather_default": 4.1, "fuel_idx": 93.8,
        "cargo": "Steel, Heavy Machinery, Coal, Agri Commodities",
        "description": "Heavy industrial raw material route bridging mineral-rich eastern belts to northern consumption zones."
    },
    "Chennai → Bengaluru (NH75 Auto Belt)": {
        "zone": "South Corridor",
        "base_demand": 480, "dist_km": 346,
        "lat1": 13.0827, "lon1": 80.2707, "lat2": 12.9716, "lon2": 77.5946,
        "weather_default": 1.5, "fuel_idx": 100.2,
        "cargo": "Automobiles, Precision Machinery, Apparel",
        "description": "Short-haul high-frequency automotive corridor connecting Chennai port cluster with Bengaluru assembly plants."
    },
    "Ahmedabad → Mumbai (NH48 Western Express)": {
        "zone": "West Corridor",
        "base_demand": 680, "dist_km": 525,
        "lat1": 23.0225, "lon1": 72.5714, "lat2": 19.0760, "lon2": 72.8777,
        "weather_default": 2.0, "fuel_idx": 92.4,
        "cargo": "Chemicals, Petrochemicals, Textiles, Processed Food",
        "description": "Dense industrial highway experiencing high daily heavy commercial vehicle movement."
    }
}

TRUCK_CAPACITY_TONS = 22.0

# -----------------------------------------------------------------------------
# REAL-TIME SIMULATION & MODEL ENGINE PIPELINE
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_india_data(corridor_name, horizon_days, is_auto_signals, manual_fuel, manual_weather):
    cfg = INDIAN_CORRIDORS[corridor_name]
    base = cfg["base_demand"]
    
    end_hist = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    start_hist = end_hist - timedelta(days=365)
    all_dates = pd.date_range(start_hist, end_hist + timedelta(days=horizon_days), freq="D")
    n = len(all_dates)
    
    rng = np.random.default_rng(seed=abs(hash(corridor_name)) % (2**31))
    
    t = np.arange(n)
    dow = np.array([d.weekday() for d in all_dates])
    
    # Regional Weather & Fuel setup
    if is_auto_signals:
        weather_base = cfg["weather_default"]
        fuel_base = cfg["fuel_idx"]
    else:
        weather_base = manual_weather
        fuel_base = manual_fuel

    seasonal = base + 0.15 * base * np.sin(2 * np.pi * t / 365.25) + rng.normal(0, base * 0.04, n)
    fuel_price = fuel_base + rng.normal(0, 0.8, n)
    weather_idx = np.clip(weather_base + rng.normal(0, 0.4, n), 0, 5)
    
    demand = seasonal - 2.5 * (fuel_price - 95.0) - 12.0 * weather_idx
    demand = np.clip(demand, 100, None)
    
    df = pd.DataFrame({
        "date": all_dates,
        "true_demand": np.round(demand, 1),
        "fuel_price": np.round(fuel_price, 2),
        "weather_idx": np.round(weather_idx, 2),
        "dow": dow,
        "is_future": all_dates > pd.Timestamp(end_hist)
    })
    
    # Feature Engineering
    for lag in [1, 3, 7, 14]:
        df[f"lag_{lag}"] = df["true_demand"].shift(lag)
    df["roll_mean_7"] = df["true_demand"].shift(1).rolling(7, min_periods=1).mean()
    df["roll_std_7"] = df["true_demand"].shift(1).rolling(7, min_periods=1).std().fillna(0)
    
    return df.bfill().ffill()

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("FreightSense AI Controls")

selected_corridor = st.sidebar.selectbox(
    "Select Indian Transport Corridor", 
    list(INDIAN_CORRIDORS.keys())
)

selected_model_name = st.sidebar.selectbox(
    "Select Predictive Model Engine",
    ["XGBoost Regressor", "Random Forest Regressor", "Gradient Boosting Regressor"]
)

horizon_days = st.sidebar.slider(
    "Forecast Timeframe Horizon (Max 90 Days)", 
    min_value=7, 
    max_value=90, 
    value=30, 
    step=1,
    help="Select forecast range up to 90 days max"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Real-Time Signal Settings")
auto_signals = st.sidebar.checkbox("Automatic Zone-Specific Signals", value=True)

if not auto_signals:
    manual_fuel = st.sidebar.slider("Diesel Price Index (INR / L)", 80.0, 120.0, 95.0)
    manual_weather = st.sidebar.slider("Monsoon / Weather Severity (0 to 5 Index)", 0.0, 5.0, 2.0)
else:
    manual_fuel = INDIAN_CORRIDORS[selected_corridor]["fuel_idx"]
    manual_weather = INDIAN_CORRIDORS[selected_corridor]["weather_default"]

# Process Data & Model
df = generate_india_data(selected_corridor, horizon_days, auto_signals, manual_fuel, manual_weather)

feature_cols = ["lag_1", "lag_3", "lag_7", "lag_14", "roll_mean_7", "roll_std_7", "fuel_price", "weather_idx"]
train_df = df[~df["is_future"]]

if selected_model_name == "XGBoost Regressor":
    model = XGBRegressor(n_estimators=120, max_depth=5, learning_rate=0.05, random_state=42)
elif selected_model_name == "Random Forest Regressor":
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
else:
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)

model.fit(train_df[feature_cols], train_df["true_demand"])
df["predicted_demand"] = model.predict(df[feature_cols])

# -----------------------------------------------------------------------------
# MAIN LAYOUT: TITLE & 3D INTERACTIVE INDIA MAP (TOP SECTION)
# -----------------------------------------------------------------------------
st.title("FreightSense AI · India Freight & Demand Forecasting")
st.caption("3D Spatial Logistics Engine & Zone-Specific Predictive Analytics")

# MAP SECTION
st.subheader("3D Interactive Freight Map of India")

corridor_info = INDIAN_CORRIDORS[selected_corridor]

# Prepare PyDeck Map Data
arc_data = []
node_data = []

for name, info in INDIAN_CORRIDORS.items():
    is_active = (name == selected_corridor)
    arc_data.append({
        "name": name,
        "from_lat": info["lat1"], "from_lon": info["lon1"],
        "to_lat": info["lat2"], "to_lon": info["lon2"],
        "color": [255, 100, 0, 255] if is_active else [0, 150, 255, 150]
    })
    node_data.append({"name": name + " Origin", "lat": info["lat1"], "lon": info["lon1"], "color": [255, 255, 255]})
    node_data.append({"name": name + " Dest", "lat": info["lat2"], "lon": info["lon2"], "color": [0, 255, 150]})

arc_layer = pdk.Layer(
    "ArcLayer",
    data=arc_data,
    get_source_position=["from_lon", "from_lat"],
    get_target_position=["to_lon", "to_lat"],
    get_source_color="color",
    get_target_color="color",
    get_width=6,
    pickable=True
)

scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=node_data,
    get_position=["lon", "lat"],
    get_fill_color="color",
    get_radius=25000,
    pickable=True
)

view_state = pdk.ViewState(
    latitude=21.5937,
    longitude=78.9629,
    zoom=4.3,
    pitch=45,
    bearing=-10
)

st.pydeck_chart(pdk.Deck(
    layers=[arc_layer, scatter_layer],
    initial_view_state=view_state,
    tooltip={"text": "{name}"},
    map_style="mapbox://styles/mapbox/dark-v10"
))

# -----------------------------------------------------------------------------
# ZONE DEMAND DESCRIPTION BOX (EXPANDS ON SELECTION)
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="zone-box">
    <h3 style="margin-top:0; color: #58a6ff;">Zone Profile: {corridor_info['zone']} ({selected_corridor})</h3>
    <p style="font-size: 1.05rem; margin-bottom: 10px;"><b>Overview:</b> {corridor_info['description']}</p>
    <div style="display: flex; gap: 30px; flex-wrap: wrap;">
        <div><b>Primary Cargo:</b> {corridor_info['cargo']}</div>
        <div><b>Route Distance:</b> {corridor_info['dist_km']} KM</div>
        <div><b>Regional Fuel Price:</b> ₹{corridor_info['fuel_idx']} / Liter</div>
        <div><b>Monsoon Severity Index:</b> {corridor_info['weather_default']} / 5.0</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA DASHBOARD WITH METRICS & GRAPH (BELOW MAP)
# -----------------------------------------------------------------------------
st.subheader("Freight Demand Dashboard & Analytics")

fut = df[df["is_future"]]
tot_vol = fut["predicted_demand"].sum()
avg_daily = fut["predicted_demand"].mean()
req_trucks = int(np.ceil(avg_daily / TRUCK_CAPACITY_TONS))

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Predicted Volume</div><div class="metric-value">{tot_vol:,.0f} Tons</div><div class="metric-sub">Next {horizon_days} Days Horizon</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Required Daily Fleet</div><div class="metric-value">{req_trucks} Trucks / Day</div><div class="metric-sub">22-Ton HCV Vehicles</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Active Fuel Benchmark</div><div class="metric-value">₹{fut["fuel_price"].mean():.2f} / L</div><div class="metric-sub">Regional Average Price</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Selected Engine</div><div class="metric-value" style="font-size:1.2rem; margin-top:0.6rem;">{selected_model_name}</div><div class="metric-sub">Active Forecast Model</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 3D STYLED INTERACTIVE DEMAND GRAPH
st.markdown(f"### Interactive Demand Trend & Forecast Window ({horizon_days}-Day Horizon)")

fig = go.Figure()

plot_df = df.iloc[-(120 + horizon_days):]
hist_data = plot_df[~plot_df["is_future"]]
fut_data = plot_df[plot_df["is_future"]]

fig.add_trace(go.Scatter(
    x=hist_data["date"], 
    y=hist_data["true_demand"], 
    mode="lines", 
    name="Historical Demand (Tons)", 
    line=dict(color="#58a6ff", width=2)
))

fig.add_trace(go.Scatter(
    x=fut_data["date"], 
    y=fut_data["predicted_demand"], 
    mode="lines+markers", 
    name=f"Predicted Demand ({selected_model_name})", 
    line=dict(color="#3fb950", width=3, dash="solid")
))

fig.update_layout(
    template="plotly_dark",
    height=480,
    margin=dict(l=20, r=20, t=30, b=20),
    hovermode="x unified",
    xaxis_title="Timeline / Date",
    yaxis_title="Freight Volume (Metric Tons)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# REAL-TIME LOGISTICS NEWS FEED
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Live India Freight & Logistics News Feed")

try:
    feed = feedparser.parse("https://news.google.com/rss/search?q=india+freight+logistics+trucking&hl=en-IN&gl=IN&ceid=IN:en")
    entries = feed.entries[:4]
    
    col_n1, col_n2 = st.columns(2)
    for i, entry in enumerate(entries):
        with col_n1 if i % 2 == 0 else col_n2:
            st.markdown(f"""
            <div class="news-box">
                <a href="{entry.link}" target="_blank" style="color:#58a6ff; font-weight:600; text-decoration:none;">{entry.title}</a>
                <p style="font-size:0.8rem; color:#7d8590; margin-top:5px;">Published: {entry.published}</p>
            </div>
            """, unsafe_allow_html=True)
except Exception as e:
    st.info("Live news feed updating... connect internet or re-run cell to refresh RSS stream.")
'''

with open("app.py", "w") as f:
    f.write(app_code)

print("'app.py' written successfully.")

# =============================================================================
# 3. START STREAMLIT & CAPTURE CLOUDFLARE TUNNEL URL
# =============================================================================
print("Starting Streamlit application in background...")
subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"])
time.sleep(3)

print("Generating public Cloudflare Tunnel link...\n")
tunnel_proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://localhost:8501"], 
    stdout=subprocess.PIPE, 
    stderr=subprocess.STDOUT, 
    text=True
)

tunnel_url = None
for line in iter(tunnel_proc.stdout.readline, ''):
    match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
    if match:
        tunnel_url = match.group(0)
        break

if tunnel_url:
    display(HTML(f'''
    <div style="background-color: #161b22; border: 2px solid #38d430; border-radius: 10px; padding: 20px; font-family: Arial, sans-serif; margin: 10px 0;">
        <h3 style="color: #38d430; margin-top: 0;">Your App is Live!</h3>
        <p style="color: #e6edf3; font-size: 16px;">Click the button below to open your updated interactive Streamlit Dashboard:</p>
        <a href="{tunnel_url}" target="_blank" style="background-color: #238636; color: white; padding: 12px 24px; text-decoration: none; font-size: 18px; font-weight: bold; border-radius: 6px; display: inline-block;">OPEN DASHBOARD APP</a>
        <p style="color: #8b949e; font-size: 12px; margin-bottom: 0; margin-top: 15px;">Direct Link: <a href="{tunnel_url}" target="_blank" style="color: #58a6ff;">{tunnel_url}</a></p>
    </div>
    '''))
else:
    print("Failed to automatically detect tunnel URL. Check cloudflared output.")
