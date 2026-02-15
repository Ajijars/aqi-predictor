import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import requests
import os
import glob



# ML imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


# ================= LIVE DATA MEMORY =================
HISTORY_CSV = "live_history.csv"

def append_live_history(record):
    df = pd.DataFrame([record])
    if os.path.exists(HISTORY_CSV):
        df.to_csv(HISTORY_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(HISTORY_CSV, index=False)

def load_live_history():
    if os.path.exists(HISTORY_CSV):
        return pd.read_csv(HISTORY_CSV, parse_dates=['datetime'])
    return pd.DataFrame(columns=['datetime','AQI'])

# Streamlit page config
st.set_page_config(page_title="Delhi AQI Dashboard", layout="wide", initial_sidebar_state="collapsed")

# ===== LIVE SETTINGS =====
WAQI_TOKEN = "6f73ae7ea07f5e29f9e9eaed3e2d6dcbb1fb4b5a"
CITY = "delhi"
HISTORY_HOURS = 24
HISTORICAL_FOLDER = "historical_data"

# CSS Styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .aqi-box-good { background: linear-gradient(135deg, #52C77F 0%, #3AAE5C 100%); color: white; }
    .aqi-box-satisfactory { background: linear-gradient(135deg, #B8E6C4 0%, #7FD4A6 100%); color: black; }
    .aqi-box-moderate { background: linear-gradient(135deg, #FFE66D 0%, #FFD700 100%); color: black; }
    .aqi-box-poor { background: linear-gradient(135deg, #FF9F43 0%, #FF6C00 100%); color: white; }
    .aqi-box-verypoor { background: linear-gradient(135deg, #FF6B6B 0%, #C92A2A 100%); color: white; }
    .aqi-box-severe { background: linear-gradient(135deg, #8B0000 0%, #4D0000 100%); color: white; }
    .aqi-display {
        padding: 40px 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 60px;
        font-weight: bold;
        margin: 20px 0;
    }
    .advice-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #007bff;
    }
    </style>
""", unsafe_allow_html=True)


def fetch_live_waqi_city(token):
    """
    Fetch Delhi city AQI by averaging ALL WAQI stations
    Also fetch pollutants from central Delhi station (/feed/delhi)
    """
    try:
        # ---- CITY AQI FROM ALL STATIONS ----
        map_url = f"https://api.waqi.info/map/bounds/?token={token}&latlng=28.40,76.80,28.90,77.40"
        r = requests.get(map_url, timeout=10)
        data = r.json()

        if data.get("status") != "ok":
            return None

        stations = data.get("data", [])
        aqi_values = [float(s["aqi"]) for s in stations if str(s["aqi"]).isdigit()]
        
        if len(aqi_values) == 0:
            return None
            
        city_aqi = np.mean(aqi_values)

        # ---- POLLUTANTS FROM MAIN DELHI FEED ----
        feed_url = f"https://api.waqi.info/feed/delhi/?token={token}"
        r2 = requests.get(feed_url, timeout=10)
        feed = r2.json()

        pollutants = {}
        if feed.get("status") == "ok":
            iaqi = feed["data"].get("iaqi", {})
            pollutants = {k: v.get("v") for k, v in iaqi.items()}

        return {
            "datetime": pd.Timestamp.now(),
            "AQI": round(city_aqi, 1),
            "pollutants": pollutants
        }

    except Exception as e:
        st.error(f"WAQI error: {e}")
        return None




def load_historical_data_from_folder():
    import re

    files = glob.glob(f"{HISTORICAL_FOLDER}/*.xlsx")

    if len(files) == 0:
        st.warning("No historical Excel files found.")
        return pd.DataFrame(columns=['datetime','AQI'])

    month_map = {
        "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
        "july":7,"august":8,"september":9,"october":10,"november":11,"december":12
    }

    all_rows = []

    for file in files:
        fname = file.lower()
        # detect month
        month_num = None
        for m in month_map:
            if m in fname:
                month_num = month_map[m]
                break
        if month_num is None:
            st.warning(f"Month not found in filename: {file}")
            continue

        year = 2025

        parsed = False

        # Strategy A: matrix parser (day x hour heatmap)
        try:
            raw = pd.read_excel(file, header=None)
            # look for a header row where many cells look like hours (e.g., '00:00' or '0')
            hour_header_row = None
            for i in range(min(8, len(raw))):
                row_vals = raw.iloc[i].astype(str).str.strip().tolist()
                cnt = 0
                for v in row_vals[1:25]:
                    if re.match(r"^\s*\d{1,2}(:\d{2})?(:\d{2})?\s*$", str(v)):
                        cnt += 1
                if cnt >= 10:
                    hour_header_row = i
                    break

            if hour_header_row is not None:
                df = pd.read_excel(file, header=hour_header_row)
                # rename first col to 'day'
                df = df.rename(columns={df.columns[0]: 'day'})
                # melt
                df_long = df.melt(id_vars=['day'], var_name='hour', value_name='AQI')
                df_long = df_long.dropna()
                # ensure numeric AQI
                df_long['AQI'] = pd.to_numeric(df_long['AQI'], errors='coerce')
                df_long = df_long[df_long['AQI'].between(0,500)]
                # parse hour - extract numeric hour value
                df_long['hour'] = df_long['hour'].astype(str).str.strip()
                df_long['hour'] = df_long['hour'].str.extract(r'(\d{1,2})', expand=False)
                df_long['hour'] = pd.to_numeric(df_long['hour'], errors='coerce')
                df_long['day'] = pd.to_numeric(df_long['day'], errors='coerce')
                df_long = df_long.dropna(subset=['day', 'hour', 'AQI'])
                # Ensure day and hour are integers
                df_long['day'] = df_long['day'].astype(int)
                df_long['hour'] = df_long['hour'].astype(int)
                # Build datetime using proper pandas constructor
                df_long['datetime'] = pd.to_datetime(
                    pd.DataFrame({
                        'year': year,
                        'month': month_num,
                        'day': df_long['day'],
                        'hour': df_long['hour'],
                        'minute': 0,
                        'second': 0
                    })
                )
                all_rows.append(df_long[['datetime','AQI']])
                parsed = True
        except Exception as e:
            st.warning(f"Matrix parse failed for {file}: {e}")

        if parsed:
            continue

        # Strategy B: header-detection fallback (search for AQI header row)
        try:
            raw = pd.read_excel(file, header=None)
            header_row = None
            for i in range(min(15, len(raw))):
                row_values = raw.iloc[i].astype(str).str.lower().tolist()
                if any('aqi' in cell for cell in row_values):
                    header_row = i
                    break
            if header_row is None:
                st.warning(f"Header not found in {file}")
                continue
            df = pd.read_excel(file, header=header_row)
            df.columns = [str(c).strip().lower() for c in df.columns]
            # find datetime and aqi columns
            dt_col = None
            for col in df.columns:
                if any(k in col for k in ['date','time','datetime','from']):
                    dt_col = col
                    break
            aqi_col = None
            for col in df.columns:
                if 'aqi' in col:
                    aqi_col = col
                    break
            if dt_col is None or aqi_col is None:
                st.warning(f"Columns not found in {file}")
                continue
            df['datetime'] = pd.to_datetime(df[dt_col], errors='coerce')
            df['AQI'] = pd.to_numeric(df[aqi_col], errors='coerce')
            df2 = df[['datetime','AQI']].dropna()
            df2 = df2[df2['AQI'].between(0,500)]
            if not df2.empty:
                all_rows.append(df2)
                parsed = True
        except Exception as e:
            st.warning(f"Fallback parse failed for {file}: {e}")

        if not parsed:
            st.warning(f"Failed reading {file}")

    if len(all_rows) == 0:
        st.error("Historical parsing failed.")
        st.stop()

    combined = pd.concat(all_rows)
    combined = combined.sort_values("datetime").reset_index(drop=True)

    st.success(f"Loaded {len(combined)} hourly AQI records from 2025 dataset")

    return combined

def train_model_from_history(df):
    # Requires at least ~50 rows to train reasonably
    if df.shape[0] < 30:
        return None, None
    df = df.sort_values('datetime').reset_index(drop=True)
    df['Hour'] = df['datetime'].dt.hour
    df['Day'] = df['datetime'].dt.day
    df['Month'] = df['datetime'].dt.month
    df['DayOfWeek'] = df['datetime'].dt.weekday
    df['Lag_1'] = df['AQI'].shift(1)
    df['Lag_3'] = df['AQI'].shift(3)
    df['Lag_24'] = df['AQI'].shift(24)
    df['Rolling_6'] = df['AQI'].rolling(6,min_periods=1).mean()
    df['Rolling_24'] = df['AQI'].rolling(24,min_periods=1).mean()
    df = df.dropna().reset_index(drop=True)
    if df.shape[0] < 10:
        return None, None
    features = ['Hour','Day','Month','DayOfWeek','Lag_1','Lag_3','Lag_24','Rolling_6','Rolling_24']
    X = df[features]
    y = df['AQI']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model_local = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    model_local.fit(X_train, y_train)
    preds = model_local.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    # Save model for reuse
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_local, f)
    return model_local, mae

def predict_next_24_hours(history_df, trained_model):

    if trained_model is None or len(history_df) < 48:
        return None

    df = history_df.sort_values('datetime').copy().reset_index(drop=True)

    future_preds = []
    future_times = []

    working_df = df.copy()

    for i in range(24):

        future_time = working_df.iloc[-1]['datetime'] + timedelta(hours=1)

        # REAL historical lag logic
        Lag_1 = working_df.iloc[-1]['AQI']
        Lag_3 = working_df.iloc[-3]['AQI']
        Lag_24 = df.iloc[-24]['AQI']   # critical fix

        Rolling_6 = working_df.tail(6)['AQI'].mean()
        Rolling_24 = working_df.tail(24)['AQI'].mean()

        input_features = pd.DataFrame({
            'Hour':[future_time.hour],
            'Day':[future_time.day],
            'Month':[future_time.month],
            'DayOfWeek':[future_time.weekday()],
            'Lag_1':[Lag_1],
            'Lag_3':[Lag_3],
            'Lag_24':[Lag_24],
            'Rolling_6':[Rolling_6],
            'Rolling_24':[Rolling_24]
        })

        pred = trained_model.predict(input_features)[0]
        pred = max(0, min(500, pred))

        new_row = pd.DataFrame({'datetime':[future_time],'AQI':[pred]})
        working_df = pd.concat([working_df, new_row], ignore_index=True)

        future_times.append(future_time)
        future_preds.append(pred)

    return pd.DataFrame({
        'datetime': future_times,
        'predicted_AQI': future_preds
    })

# ===== LOAD TRAINED MODEL =====
@st.cache_resource
def load_model():
    import os, pickle
    BASE_DIR = os.path.dirname(__file__)
    model_path = os.path.join(BASE_DIR, "model.pkl")

    if not os.path.exists(model_path):
        st.error("model.pkl missing in repository!")
        st.stop()

    with open(model_path, "rb") as f:
        return pickle.load(f)

trained_model = load_model()

# Note: synthetic data functions removed; app uses live WAQI data only.

# Get AQI category and color
def get_aqi_category(aqi_value):
    if aqi_value <= 50:
        return "Good", "#52C77F", "aqi-box-good"
    elif aqi_value <= 100:
        return "Satisfactory", "#B8E6C4", "aqi-box-satisfactory"
    elif aqi_value <= 200:
        return "Moderate", "#FFE66D", "aqi-box-moderate"
    elif aqi_value <= 300:
        return "Poor", "#FF9F43", "aqi-box-poor"
    elif aqi_value <= 400:
        return "Very Poor", "#FF6B6B", "aqi-box-verypoor"
    else:
        return "Severe", "#8B0000", "aqi-box-severe"

# Note: old random predictor removed. Real predictor will be added below train function.

# Health advice based on AQI
def get_health_advice(aqi_value):
    advice = {}
    
    if aqi_value <= 50:
        advice = {
            "General Public": ["✅ Air quality is good", "👍 Enjoy outdoor activities"],
            "Children": ["✅ Outdoor play is safe", "🏃 Encourage physical activities"],
            "Elderly": ["✅ No restrictions", "🚴 Regular exercise recommended"],
            "Asthma/Heart": ["✅ Minimal impact", "🏥 Continue normal routine"]
        }
    elif aqi_value <= 100:
        advice = {
            "General Public": ["Activity can continue normally", "No major health impacts"],
            "Children": ["Light outdoor activities allowed", "Avoid strenuous exercise"],
            "Elderly": ["Normal activities safe", "Stay hydrated"],
            "Asthma/Heart": ["Use prescribed inhalers", "Avoid intense exercise"]
        }
    elif aqi_value <= 200:
        advice = {
            "General Public": ["⚠️ Reduce prolonged outdoor activity", "Consider using masks"],
            "Children": ["🏠 Limit outdoor play", "Use N95 masks if outside"],
            "Elderly": ["⚠️ Avoid vigorous activities", "Wear protective masks"],
            "Asthma/Heart": ["⚠️ Avoid outdoor exertion", "Keep inhalers handy", "Minimize exposure"]
        }
    elif aqi_value <= 300:
        advice = {
            "General Public": ["🚫 Avoid outdoor activities", "Wear N95 masks outdoors", "Keep windows closed"],
            "Children": ["🏠 Stay indoors if possible", "Use air purifier", "Wear N95 masks"],
            "Elderly": ["🏠 Minimize outdoor activities", "Use air-purifier", "Stay hydrated"],
            "Asthma/Heart": ["🚫 Avoid outdoor activities", "Keep medication accessible", "Use air purifier indoors"]
        }
    elif aqi_value <= 400:
        advice = {
            "General Public": ["🚫 Avoid all outdoor activity", "Wear N95/P100 mask if must go out", "Close all windows", "Use air purifier", "Keep car windows closed"],
            "Children": ["🏠 Stay indoors completely", "Use air purifier", "Wear N95 mask", "Limit to essential outings only"],
            "Elderly": ["🏠 Stay indoors", "Use high-quality air purifier", "Wear P100 mask if necessary", "Avoid exertion"],
            "Asthma/Heart": ["🚫 Stay indoors", "Use air purifier continuously", "Have prescribed medications ready", "Limit movement", "Consult doctor if symptoms worsen"]
        }
    else:
        advice = {
            "General Public": ["🚫 HAZARDOUS - Avoid all outdoor activity", "Wear P100/N99 mask if absolutely necessary", "Close all windows/doors", "Use high-grade air purifier", "Limit outdoor time to minimum"],
            "Children": ["🏠 DO NOT go outside", "Use HEPA air purifier", "Wear N95 mask if emergency outing", "Avoid all strenuous activity"],
            "Elderly": ["🏠 STAY INDOORS", "Use best available air purifier", "Limit any movement", "Monitor health closely", "Call doctor if breathing issues"],
            "Asthma/Heart": ["🏥 CONSULT DOCTOR IMMEDIATELY", "Use prescribed medications proactively", "DO NOT go outside", "Use air purifier continuously", "Have emergency contact ready"]
        }
    
    return advice

# ================= FINAL DATA PIPELINE =================

# 1️⃣ Fetch LIVE city AQI (avg of all stations + pollutants)
live = fetch_live_waqi_city(WAQI_TOKEN)
# Save live reading into memory
if live:
    append_live_history({
        "datetime": live["datetime"],
        "AQI": live["AQI"]
    })



# 3️⃣ Load historical dataset
historical_df = load_historical_data_from_folder()

# 4️⃣ Merge historical + accumulated live history
live_history_df = load_live_history()

combined_df = pd.concat([historical_df, live_history_df], ignore_index=True)



combined_df["datetime"] = pd.to_datetime(combined_df["datetime"], utc=True).dt.tz_localize(None)
combined_df = combined_df.dropna().drop_duplicates("datetime").sort_values("datetime")

history = combined_df.copy()

if len(combined_df) < 50:
    st.warning("Collecting live data... come back later 🙂")
    st.stop()



# 6️⃣ TRUE LAST 24 HOURS FILTER (REAL TIME WINDOW)

combined_df = combined_df.sort_values("datetime").reset_index(drop=True)

now = combined_df["datetime"].max()
last_24h_time = now - timedelta(hours=24)

recent_df = combined_df[
    combined_df["datetime"] >= last_24h_time
].copy()

recent_df = recent_df.sort_values("datetime").reset_index(drop=True)





# 7️⃣ Forecast next 24h
trained_model, _ = train_model_from_history(combined_df)
future_df = predict_next_24_hours(combined_df, trained_model)


# 8️⃣ Current AQI MUST be LIVE
current_aqi = float(live["AQI"]) if live else float(recent_df.iloc[-1]["AQI"])

# Validate data before computing stats
if recent_df.empty or recent_df['AQI'].isna().all():
    st.error("❌ No valid AQI data available. Historical files may be corrupted or WAQI API is not responding.")
    st.stop()

# Now compute stats from last 24h dataset
avg_aqi = recent_df['AQI'].mean()
max_aqi = recent_df['AQI'].max()
min_aqi = recent_df['AQI'].min()

# Handle NaN values in stats
if pd.isna(avg_aqi):
    avg_aqi = current_aqi
if pd.isna(max_aqi):
    max_aqi = current_aqi
if pd.isna(min_aqi):
    min_aqi = current_aqi

current_category, current_color, current_style = get_aqi_category(current_aqi)

# HEADER
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🌫️ Delhi Air Quality Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #666;'>Monitor Last 24 Hours AQI & Predict Next 24 Hours</h3>", unsafe_allow_html=True)
st.markdown("---")

# AQI legend (compact)
st.markdown("""
<div style='display:flex;gap:8px;justify-content:center;margin-bottom:8px;'>
    <div style='padding:8px;border-radius:6px;background:#52C77F;color:#fff;'>Good (0-50)</div>
    <div style='padding:8px;border-radius:6px;background:#B8E6C4;color:#000;'>Satisfactory (51-100)</div>
    <div style='padding:8px;border-radius:6px;background:#FFE66D;color:#000;'>Moderate (101-200)</div>
    <div style='padding:8px;border-radius:6px;background:#FF9F43;color:#fff;'>Poor (201-300)</div>
    <div style='padding:8px;border-radius:6px;background:#FF6B6B;color:#fff;'>Very Poor (301-400)</div>
    <div style='padding:8px;border-radius:6px;background:#8B0000;color:#fff;'>Severe (401-500)</div>
</div>
""", unsafe_allow_html=True)

# CURRENT AQI STATUS
col1, col2, col3 = st.columns([2, 2, 2])

with col2:
    st.markdown(f"""
    <div class='aqi-display {current_style}'>
        {current_aqi:.0f}
        <div style='font-size: 30px; margin-top: 10px;'>{current_category}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# METRICS
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Current AQI", f"{current_aqi:.0f}")
with col2:
    st.metric("📈 24h Average", f"{avg_aqi:.0f}")
with col3:
    st.metric("⬆️ Peak AQI", f"{max_aqi:.0f}")
with col4:
    st.metric("⬇️ Low AQI", f"{min_aqi:.0f}")

st.markdown("---")

# LAST 24 HOURS CHART
st.subheader("📉 Last 24 Hours AQI Trend")
if not recent_df.empty and not recent_df['datetime'].isna().all() and not recent_df['AQI'].isna().all():
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=recent_df['datetime'],
        y=recent_df['AQI'],
        mode='lines+markers',
        name='Last 24h AQI',
        line=dict(color='#636EFA', width=3),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(99, 110, 250, 0.1)'
    ))
    fig1.update_layout(
        height=400,
        hovermode='x unified',
        xaxis_title='Time',
        yaxis_title='AQI',
        template='plotly_white',
        margin=dict(l=0, r=0, t=0, b=0)
    )
    fig1.update_xaxes(type="date", tickformat="%H:%M\n%b %d")
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.warning("⚠️ Insufficient data for 24h trend chart")

st.markdown("---")

# NEXT 24 HOURS PREDICTION
st.subheader("🔮 Next 24 Hours AQI Prediction")
if future_df is not None and not future_df.empty and not future_df['datetime'].isna().all() and not future_df['predicted_AQI'].isna().all():
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=future_df['datetime'],
        y=future_df['predicted_AQI'],
        mode='lines+markers',
        name='Predicted AQI',
        line=dict(color='#EF553B', width=3, dash='dash'),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(239, 85, 59, 0.1)'
    ))
    fig2.update_layout(
        height=400,
        hovermode='x unified',
        xaxis_title='Time',
        yaxis_title='AQI',
        template='plotly_white',
        margin=dict(l=0, r=0, t=0, b=0)
    )
    fig2.update_xaxes(type="date", tickformat="%H:%M\n%b %d")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("⚠️ Model prediction failed. Using historical data only.")

st.markdown("---")

# COMBINED GRAPH
st.subheader("📊 48-Hour AQI Overview (Past 24h + Forecast)")
if not recent_df.empty and future_df is not None and not future_df.empty:
    fig3 = go.Figure()

    if not recent_df['datetime'].isna().all() and not recent_df['AQI'].isna().all():
        fig3.add_trace(go.Scatter(
            x=recent_df['datetime'],
            y=recent_df['AQI'],
            mode='lines+markers',
            name='Last 24h (Actual)',
            line=dict(color='#636EFA', width=3),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(99, 110, 250, 0.1)'
        ))

    if not future_df['datetime'].isna().all() and not future_df['predicted_AQI'].isna().all():
        fig3.add_trace(go.Scatter(
            x=future_df['datetime'],
            y=future_df['predicted_AQI'],
            mode='lines+markers',
            name='Next 24h (Forecast)',
            line=dict(color='#EF553B', width=3, dash='dash'),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(239, 85, 59, 0.1)'
        ))

    fig3.update_layout(
        height=500,
        hovermode='x unified',
        xaxis_title='Time',
        yaxis_title='AQI',
        template='plotly_white',
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(x=0.01, y=0.99)
    )
    fig3.update_xaxes(type="date", tickformat="%H:%M\n%b %d")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("⚠️ Insufficient data for 48-hour overview")

st.markdown("---")

# REAL POLLUTANTS FROM WAQI
st.subheader("🔬 Pollutant Composition")

pollutants = {}
if live and live.get("pollutants"):
    mapping = {'pm25':'PM2.5','pm10':'PM10','no2':'NO₂','o3':'O₃','so2':'SO₂','co':'CO'}
    for k,label in mapping.items():
        val = live["pollutants"].get(k)
        if val is not None:
            pollutants[label] = float(val)

col1, col2 = st.columns([2, 3])
with col1:
    # Display only non-NaN pollutants
    for label, value in pollutants.items():
        if not pd.isna(value):
            st.metric(f"{label} (µg/m³)", f"{value:.1f}")
    if all(pd.isna(v) for v in pollutants.values()):
        st.info("Pollutant data not available from live feed")
with col2:
    # Pie chart for pollutant composition (visual) - only if we have data
    valid_pollutants = {k: v for k, v in pollutants.items() if not pd.isna(v)}
    if valid_pollutants and len(valid_pollutants) > 0:
        fig_pollutants = go.Figure(data=[
            go.Pie(labels=list(valid_pollutants.keys()), values=list(valid_pollutants.values()), hole=0.4,
                   marker=dict(colors=['#FF6B6B', '#FF9F43', '#FFE66D', '#52C77F', '#1E90FF', '#9D4EDD']))
        ])
        fig_pollutants.update_traces(textinfo='percent+label')
        fig_pollutants.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
        st.plotly_chart(fig_pollutants, use_container_width=True)
    else:
        st.info("No valid pollutant data to display")

st.markdown("---")

# KEY INSIGHTS
st.subheader("💡 Key Insights & Alerts")

col1, col2 = st.columns(2)

with col1:
    # Trend Analysis - handle edge cases
    if len(recent_df) >= 5:
        trend = "📈 INCREASING" if recent_df['AQI'].iloc[-1] > recent_df['AQI'].iloc[-5] else "📉 DECREASING"
        trend_value = recent_df['AQI'].iloc[-1] - recent_df['AQI'].iloc[-5]
        st.info(f"**24h Trend:** {trend} ({abs(trend_value):.1f} points)")
    
    # Peak hour - handle NaN datetime
    if not recent_df.empty and recent_df['AQI'].notna().any():
        peak_idx = recent_df['AQI'].idxmax()
        peak_dt = recent_df.loc[peak_idx, 'datetime']
        if pd.notna(peak_dt):
            try:
                peak_hour = pd.Timestamp(peak_dt).strftime("%H:%M")
                peak_value = recent_df['AQI'].max()
                st.warning(f"**Peak AQI Hour:** {peak_hour} ({peak_value:.0f})")
            except:
                st.warning(f"**Peak AQI:** {recent_df['AQI'].max():.0f}")

with col2:
    # Best hour - handle NaN datetime
    if not recent_df.empty and recent_df['AQI'].notna().any():
        best_idx = recent_df['AQI'].idxmin()
        best_dt = recent_df.loc[best_idx, 'datetime']
        if pd.notna(best_dt):
            try:
                best_hour = pd.Timestamp(best_dt).strftime("%H:%M")
                best_value = recent_df['AQI'].min()
                st.success(f"**Best AQI Hour:** {best_hour} ({best_value:.0f})")
            except:
                st.success(f"**Best AQI:** {recent_df['AQI'].min():.0f}")
    
    # Next 24h forecast
    if future_df is not None and not future_df.empty and future_df['predicted_AQI'].notna().any():
        max_future = future_df['predicted_AQI'].max()
        st.metric("**Forecast Peak (24h):**", f"{max_future:.0f}")

st.markdown("---")

# Provide combined CSV download (actual + forecast)
if not recent_df.empty and future_df is not None and not future_df.empty:
    try:
        combined_actual = recent_df.copy().rename(columns={'AQI':'AQI_actual'})
        combined_forecast = future_df.copy().rename(columns={'predicted_AQI':'AQI_forecast'})
        combined = pd.merge_asof(
            combined_actual.sort_values('datetime'),
            combined_forecast.sort_values('datetime'),
            on='datetime', direction='nearest', tolerance=pd.Timedelta('1h')
        )
        csv_bytes = combined.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download combined AQI CSV", data=csv_bytes, file_name='aqi_combined.csv', mime='text/csv')
    except:
        pass

# HOURLY DISTRIBUTION (TRUE LAST 24 HOURS)
st.subheader("⏰ Hourly AQI Distribution")

if not recent_df.empty and not recent_df['datetime'].isna().all() and not recent_df['AQI'].isna().all():
    hourly_stats = recent_df.copy()
    hourly_stats["hour_label"] = hourly_stats["datetime"].dt.strftime("%H:%M")

    if not hourly_stats.empty:
        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Bar(
            x=hourly_stats["hour_label"],
            y=hourly_stats["AQI"],
            marker=dict(
                color=hourly_stats["AQI"],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="AQI")
            ),
            text=hourly_stats["AQI"].round(1),
            textposition='outside'
        ))
        fig_hourly.update_layout(
            height=350,
            xaxis_title="Last 24 Hours",
            yaxis_title="AQI",
            template='plotly_white',
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
    else:
        st.warning("⚠️ Unable to compute hourly distribution")
else:
    st.warning("⚠️ Insufficient data for hourly distribution")

st.markdown("---")

# AQI RISK ZONES
st.subheader("🎯 Risk Assessment")

if not recent_df.empty and not recent_df['AQI'].isna().all():
    risk_zones = {
        "Good": len(recent_df[recent_df['AQI'] <= 50]),
        "Satisfactory": len(recent_df[(recent_df['AQI'] > 50) & (recent_df['AQI'] <= 100)]),
        "Moderate": len(recent_df[(recent_df['AQI'] > 100) & (recent_df['AQI'] <= 200)]),
        "Poor": len(recent_df[(recent_df['AQI'] > 200) & (recent_df['AQI'] <= 300)]),
        "Very Poor": len(recent_df[(recent_df['AQI'] > 300) & (recent_df['AQI'] <= 400)]),
        "Severe": len(recent_df[recent_df['AQI'] > 400])
    }

    fig_risk = go.Figure(data=[
        go.Pie(
            labels=list(risk_zones.keys()),
            values=list(risk_zones.values()),
            marker=dict(colors=['#52C77F', '#B8E6C4', '#FFE66D', '#FF9F43', '#FF6B6B', '#8B0000']),
            textposition='auto'
        )
    ])
    fig_risk.update_layout(
        height=400,
        title="Hours in Each AQI Category (Last 24h)",
        template='plotly_white',
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_risk, use_container_width=True)
else:
    st.warning("⚠️ Insufficient data for risk assessment")

st.markdown("---")

# 7-DAY TREND (TRUE LAST 7 AVAILABLE DAYS FROM DATASET)
st.subheader("📅 7-Day Trend")

hist = history.copy()

if not hist.empty:
    hist["date"] = hist["datetime"].dt.date

    last_available_date = hist["date"].max()
    start_date = last_available_date - timedelta(days=6)

    last7 = hist[hist["date"] >= start_date]

    if not last7.empty:
        daily = last7.groupby("date")["AQI"].agg(["mean","max","min"]).reset_index()

        daily["date"] = daily["date"].apply(lambda d: pd.to_datetime(d).strftime("%b %d"))

        fig_7day = go.Figure()

        fig_7day.add_trace(go.Scatter(
            x=daily["date"], y=daily["max"], fill='tonexty',
            line=dict(color='rgba(255,0,0,0)'), name="Max"
        ))

        fig_7day.add_trace(go.Scatter(
            x=daily["date"], y=daily["min"], fill=None,
            line=dict(color='rgba(255,0,0,0)'), fillcolor='rgba(255,0,0,0.2)', name="Min"
        ))

        fig_7day.add_trace(go.Scatter(
            x=daily["date"], y=daily["mean"],
            mode='lines+markers', line=dict(width=3), name="Average"
        ))

        fig_7day.update_layout(
            height=350,
            hovermode='x unified',
            xaxis_title='Date',
            yaxis_title='AQI',
            template='plotly_white',
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_7day, use_container_width=True)
    else:
        st.info("Not enough historical data to show 7-day trend.")
else:
    st.info("Not enough historical data to show 7-day trend.")

st.markdown("---")

# RECOMMENDATION ENGINE
st.subheader("🎯 Smart Recommendations")

recommendations = []

if current_aqi > 200:
    recommendations.append("🏠 Work from home if possible")
    recommendations.append("🚗 Use public transport or carpool")

if current_aqi > 150:
    recommendations.append("🏃 Avoid jogging/running outdoors")
    recommendations.append("🪟 Keep windows closed")

if max_aqi > 250:
    recommendations.append("💨 Consider air purifier usage")
    recommendations.append("🧴 Stock up on N95 masks")

if not recent_df.empty and min_aqi < 75:
    try:
        best_hour_str = recent_df.loc[recent_df['AQI'].idxmin(), 'datetime'].strftime("%H:00")
        recommendations.append(f"🏃 Best time to exercise is around {best_hour_str}")
    except:
        recommendations.append("🚴 Exercise indoors or early morning")
else:
    recommendations.append("🚴 Exercise indoors or early morning")

if len(recommendations) == 0:
    recommendations = ["✅ Air quality is acceptable", "🚶 Outdoor activities can continue normally"]

col1, col2 = st.columns(2)
for i, rec in enumerate(recommendations):
    if i % 2 == 0:
        with col1:
            st.info(rec)
    else:
        with col2:
            st.info(rec)

st.markdown("---")

# HEALTH ADVICE SECTION
st.subheader("⚕️ Health Recommendations")

advice = get_health_advice(current_aqi)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<h4>👨‍👩‍👧‍👦 General Public</h4>", unsafe_allow_html=True)
    for tip in advice["General Public"]:
        st.write(f"• {tip}")
    
    st.markdown("<h4>👴👵 Elderly</h4>", unsafe_allow_html=True)
    for tip in advice["Elderly"]:
        st.write(f"• {tip}")

with col2:
    st.markdown("<h4>👶 Children</h4>", unsafe_allow_html=True)
    for tip in advice["Children"]:
        st.write(f"• {tip}")
    
    st.markdown("<h4>🏥 Asthma / Heart Disease</h4>", unsafe_allow_html=True)
    for tip in advice["Asthma/Heart"]:
        st.write(f"• {tip}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #999; font-size: 12px;'>Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "</p>", unsafe_allow_html=True)
