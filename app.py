import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import timedelta
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Air Quality Prediction", layout="wide")
st.title("🌍 Air Quality Prediction & Early Warning System")
st.markdown("👤 **Author:** Nurikamal Bolatbay")
st.markdown("---")

# ============================================================
# ONLY REAL DATA — NO SYNTHETIC FALLBACK
# ============================================================

def load_data():
    filename = "Global_City_Air_Quality_Hourly.csv"
    
    if not os.path.exists(filename):
        st.error(f"❌ File '{filename}' not found in the current directory.")
        st.info("Make sure the file is uploaded to the repository.")
        st.stop()
    
    # Try to read with different encodings and separators
    errors = []
    for sep in [',', ';', '\t']:
        for enc in ['utf-8', 'latin1', 'cp1252']:
            try:
                df = pd.read_csv(
                    filename,
                    sep=sep,
                    encoding=enc,
                    on_bad_lines='skip',
                    low_memory=False
                )
                if df.shape[1] >= 3:
                    st.success(f"✅ File loaded: {len(df)} rows, {df.shape[1]} columns")
                    return df
            except Exception as e:
                errors.append(f"{sep} / {enc}: {str(e)[:100]}")
    
    st.error("❌ Could not read the CSV file with any common settings.")
    st.write("Details of attempts:")
    for e in errors[:5]:
        st.write(f"- {e}")
    st.stop()

df_full = load_data()

# ============================================================
# AUTO-DETECT COLUMNS
# ============================================================

def find_column(df, possible):
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    for name in possible:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    for c in df.columns:
        c_lower = str(c).strip().lower()
        for name in possible:
            if name.lower() in c_lower:
                return c
    return None

time_col = find_column(df_full, ["time", "datetime", "date", "timestamp"])
pm_col = find_column(df_full, ["pm2.5", "pm25", "pm2_5"])
city_col = find_column(df_full, ["city", "location", "site"])

if time_col is None or pm_col is None:
    st.error("❌ Could not find time or PM2.5 columns.")
    st.write("Available columns:", df_full.columns.tolist())
    st.stop()

df_full.rename(columns={time_col: 'time', pm_col: 'pm25'}, inplace=True)
if city_col:
    df_full.rename(columns={city_col: 'city'}, inplace=True)
else:
    df_full['city'] = 'Unknown'

df_full['time'] = pd.to_datetime(df_full['time'], errors='coerce')
df_full['pm25'] = pd.to_numeric(df_full['pm25'], errors='coerce')
df_full = df_full.dropna(subset=['time', 'pm25'])
df_full = df_full[df_full['pm25'] >= 0]

st.sidebar.header("📍 Select City")
cities = sorted(df_full['city'].unique())
selected_city = st.sidebar.selectbox("City:", cities)

# ============================================================
# PREPARE DATA
# ============================================================

df_city = df_full[df_full['city'] == selected_city].copy()
df_city = df_city.sort_values('time')

if len(df_city) < 50:
    st.error(f"❌ Only {len(df_city)} records for {selected_city}. Need at least 50.")
    st.stop()

df_city['hour'] = df_city['time'].dt.hour
df_city['dayofweek'] = df_city['time'].dt.dayofweek
df_city['month'] = df_city['time'].dt.month
df_city['day'] = df_city['time'].dt.day

for lag in [1, 3, 6, 12, 24]:
    df_city[f'pm25_lag_{lag}'] = df_city['pm25'].shift(lag)

df_city = df_city.dropna()

if len(df_city) < 50:
    st.error(f"❌ After creating lags, only {len(df_city)} records left.")
    st.stop()

features = ['hour', 'dayofweek', 'month', 'day',
            'pm25_lag_1', 'pm25_lag_3', 'pm25_lag_6',
            'pm25_lag_12', 'pm25_lag_24']

X = df_city[features]
y = df_city['pm25']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2:.4f}")
col2.metric("MAE", f"{mae:.2f} µg/m³")
col3.metric("Records", len(df_city))

st.markdown("---")

# ============================================================
# FORECAST & WARNING (same as before)
# ============================================================

def forecast_future(model, df, features, hours_ahead=3):
    last_row = df.iloc[-1:].copy()
    forecasts = []
    for h in range(1, hours_ahead + 1):
        X_pred = last_row[features].copy()
        pred = model.predict(X_pred)[0]
        forecasts.append(pred)
        if h < hours_ahead:
            for lag in [24, 12, 6, 3, 1]:
                if f'pm25_lag_{lag}' in last_row.columns:
                    if lag == 1:
                        last_row[f'pm25_lag_{lag}'] = pred
                    else:
                        prev_lag = f'pm25_lag_{lag-1}'
                        if prev_lag in last_row.columns:
                            last_row[f'pm25_lag_{lag}'] = last_row[prev_lag].values[0]
    return forecasts

st.header("⚠️ Early Warning System")
WARNING_THRESHOLD = 50
st.caption(f"Threshold: {WARNING_THRESHOLD} µg/m³ (WHO)")

last_actual = df_city['pm25'].iloc[-1]
last_time = df_city['time'].iloc[-1]
forecast_values = forecast_future(model, df_city, features, hours_ahead=3)

w1, w2, w3, w4 = st.columns(4)
with w1:
    st.metric(f"🕐 Current ({last_time.strftime('%H:%M')})", f"{last_actual:.1f} µg/m³")

warning_triggered = False
for h, val in enumerate(forecast_values, 1):
    is_warning = val >= WARNING_THRESHOLD
    with [w2, w3, w4][h-1]:
        # Removed time display; only show +1h, +2h, +3h
        st.metric(
            f"{'🔴' if is_warning else '🟢'} +{h}h",
            f"{val:.1f} µg/m³",
            delta=f"{val - last_actual:+.1f}",
            delta_color="inverse" if is_warning else "normal"
        )
    if is_warning:
        warning_triggered = True

if warning_triggered:
    st.error(
        "🚨 WARNING! PM2.5 levels are predicted to exceed 50 µg/m³ in the next few hours. "
        "Please take precautions: Wear a mask (N95 recommended), stay indoors, close all windows, "
        "use an air purifier, avoid outdoor physical activities, keep doors sealed, and monitor vulnerable "
        "groups (children, elderly, and those with respiratory conditions) closely."
    )
else:
    st.success("✅ Air quality is predicted to remain safe. No immediate action required.")

# ============================================================
# PLOTS
# ============================================================

st.markdown("---")
st.header("📈 Forecast (Next 3 Hours)")
fig, ax = plt.subplots(figsize=(12, 5))
hist = df_city['pm25'].iloc[-24:].values
ax.plot(range(24), hist, 'b-', label='History', alpha=0.7)
ax.axvline(x=23, color='gray', linestyle='--', label='Now')
ax.plot(range(24, 24+len(forecast_values)), forecast_values, 'r--o', linewidth=2.5, markersize=8, label='Forecast')
ax.axhline(y=WARNING_THRESHOLD, color='red', linestyle=':', label='Warning')
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.markdown("---")
st.header("📊 Model Performance")
fig2, ax2 = plt.subplots(figsize=(12, 5))
n = min(150, len(y_test))
ax2.plot(range(n), y_test.iloc[:n].values, 'b-', label='Actual')
ax2.plot(range(n), y_pred[:n], 'r-', label='Predicted')
ax2.legend()
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

st.markdown("---")
st.header("🔍 Feature Importance")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.barh(range(len(features)), importances[indices], color='steelblue')
ax3.set_yticks(range(len(features)))
ax3.set_yticklabels([features[i] for i in indices])
ax3.set_xlabel('Importance')
ax3.grid(True, alpha=0.3)
st.pyplot(fig3)

st.markdown("---")
st.caption("👤 Nurikamal Bolatbay © 2026")
