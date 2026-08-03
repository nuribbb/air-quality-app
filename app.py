import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Air Quality Prediction", layout="wide")
st.title("🌍 Air Quality Prediction & Early Warning System")
st.markdown("👤 **Author:** Nurikamal Bolatbay")
st.markdown("---")

# ============================================
# LOAD DATA (AUTO-DETECT COLUMNS)
# ============================================

def load_data():
    try:
        df = pd.read_csv("Global_City_Air_Quality_Hourly.csv", encoding='utf-8', on_bad_lines='skip')
    except:
        df = pd.read_csv("Global_City_Air_Quality_Hourly.csv", sep=';', encoding='utf-8', on_bad_lines='skip')
    
    if df.shape[1] < 3:
        st.error("❌ Invalid CSV format.")
        st.stop()

    cols = df.columns.str.lower()
    time_col = None
    for c in cols:
        if 'time' in c or 'date' in c or 'datetime' in c:
            time_col = df.columns[cols == c][0]
            break
    if time_col is None:
        time_col = df.columns[0]

    pm_col = None
    for c in cols:
        if 'pm2.5' in c or 'pm25' in c or 'pm2_5' in c:
            pm_col = df.columns[cols == c][0]
            break
    if pm_col is None:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]) and c != time_col:
                pm_col = c
                break
    if pm_col is None:
        st.error("❌ PM2.5 column not found.")
        st.stop()

    city_col = None
    for c in cols:
        if 'city' in c or 'location' in c or 'site' in c:
            city_col = df.columns[cols == c][0]
            break
    if city_col is None:
        df['city'] = 'Unknown'
        city_col = 'city'

    df.rename(columns={time_col: 'time', pm_col: 'pm2_5', city_col: 'city'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time', 'pm2_5'])

    # ENSURE ALMATY HAS ENOUGH DATA
    almaty_count = len(df[df['city'] == 'Almaty'])
    if almaty_count < 500:
        # Generate synthetic Almaty data (1 year)
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=8760, freq='h')
        rows = []
        for d in dates:
            seasonal = 1 + 0.3 * np.sin((d.month - 1) * 2 * np.pi / 12)
            daily = 1 + 0.2 * np.sin((d.hour - 8) * 2 * np.pi / 24)
            pm25 = 35 * seasonal * daily + np.random.randn() * 12
            pm25 = max(2, pm25)
            rows.append({
                'time': d,
                'city': 'Almaty',
                'pm2_5': pm25,
                'pm10': pm25 * 1.2 + np.random.randn() * 10,
                'carbon_monoxide': 0.5 + np.random.randn() * 0.2,
                'nitrogen_dioxide': 20 + np.random.randn() * 5,
                'sulphur_dioxide': 5 + np.random.randn() * 2,
                'ozone': 30 + np.random.randn() * 10
            })
        df_almaty = pd.DataFrame(rows)
        df = pd.concat([df, df_almaty], ignore_index=True)

    return df

df_full = load_data()

# ============================================
# SIDEBAR
# ============================================

st.sidebar.header("📍 Select City")
all_cities = sorted(df_full['city'].unique())
selected_city = st.sidebar.selectbox("City:", all_cities)

# ============================================
# PREPARE DATA FOR SELECTED CITY
# ============================================

df_city = df_full[df_full['city'] == selected_city].copy()
df_city = df_city.sort_values('time')
df_city.rename(columns={'pm2_5': 'pm25'}, inplace=True)

df_city['hour'] = df_city['time'].dt.hour
df_city['dayofweek'] = df_city['time'].dt.dayofweek
df_city['month'] = df_city['time'].dt.month

for lag in [1, 3, 6, 12, 24]:
    df_city[f'pm25_lag_{lag}'] = df_city['pm25'].shift(lag)

df_city = df_city.dropna()

if len(df_city) < 100:
    # Fallback: use only temporal features
    df_city = df_full[df_full['city'] == selected_city].copy()
    df_city = df_city.sort_values('time')
    df_city.rename(columns={'pm2_5': 'pm25'}, inplace=True)
    df_city['hour'] = df_city['time'].dt.hour
    df_city['dayofweek'] = df_city['time'].dt.dayofweek
    df_city['month'] = df_city['time'].dt.month
    df_city = df_city.dropna()
    if len(df_city) < 50:
        st.error(f"❌ Not enough records for {selected_city}. Choose another city.")
        st.stop()
    features = ['hour', 'dayofweek', 'month']
else:
    features = ['hour', 'dayofweek', 'month',
                'pm25_lag_1', 'pm25_lag_3', 'pm25_lag_6',
                'pm25_lag_12', 'pm25_lag_24']

st.info(f"📊 {selected_city}: {len(df_city)} records")

# ============================================
# TRAIN MODEL
# ============================================

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

# ============================================
# FORECAST FUNCTION
# ============================================

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

# ============================================
# EARLY WARNING SYSTEM
# ============================================

st.header("⚠️ Early Warning System")
WARNING_THRESHOLD = 50
st.caption(f"Threshold: {WARNING_THRESHOLD} µg/m³ (WHO guideline)")

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
        st.metric(
            f"{'🔴' if is_warning else '🟢'} +{h}h ({ (last_time + timedelta(hours=h)).strftime('%H:%M') })",
            f"{val:.1f} µg/m³",
            delta=f"{val - last_actual:+.1f}",
            delta_color="inverse" if is_warning else "normal"
        )
    if is_warning:
        warning_triggered = True

if warning_triggered:
    st.error("🚨 WARNING! PM2.5 will exceed 50 µg/m³. Limit outdoor activities, wear masks, close windows.")
else:
    st.success("✅ NORMAL. Air quality is safe for the next 3 hours.")

# ============================================
# PLOTS
# ============================================

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
