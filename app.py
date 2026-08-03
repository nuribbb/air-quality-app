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

warnings.filterwarnings("ignore")


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Air Quality Prediction & Early Warning System",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Air Quality Prediction & Early Warning System")
st.markdown("**Author:** Nurikamal Bolatbay")
st.markdown("---")


# ============================================================
# FILE PATHS
# ============================================================

ALMATY_FILE = "air_quality_data.csv"
GLOBAL_FILE = "Global_City_Air_Quality_Hourly.csv"


# ============================================================
# SAFE FILE READER
# ============================================================

def read_file(filename):

    if not os.path.exists(filename):
        return None

    # Try normal CSV
    separators = [",", ";", "\t"]

    for sep in separators:
        try:
            df = pd.read_csv(
                filename,
                sep=sep,
                encoding="utf-8",
                on_bad_lines="skip"
            )

            if df.shape[1] >= 2:
                return df

        except Exception:
            pass

    # Try latin-1 as backup
    for sep in separators:
        try:
            df = pd.read_csv(
                filename,
                sep=sep,
                encoding="latin1",
                on_bad_lines="skip"
            )

            if df.shape[1] >= 2:
                return df

        except Exception:
            pass

    return None


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(df, possible_names):

    columns_lower = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    # Exact match
    for name in possible_names:
        if name.lower() in columns_lower:
            return columns_lower[name.lower()]

    # Partial match
    for c in df.columns:

        c_lower = str(c).strip().lower()

        for name in possible_names:
            if name.lower() in c_lower:
                return c

    return None


# ============================================================
# STANDARDIZE DATASET
# ============================================================

def standardize_dataset(df, forced_city=None):

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Clean column names
    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    # -------------------------
    # TIME COLUMN
    # -------------------------

    time_col = find_column(
        df,
        [
            "time",
            "datetime",
            "date",
            "timestamp",
            "last_updated"
        ]
    )

    if time_col is None:

        # Try first column
        possible = pd.to_datetime(
            df.iloc[:, 0],
            errors="coerce"
        )

        if possible.notna().sum() > len(df) * 0.5:
            time_col = df.columns[0]

    if time_col is None:
        return pd.DataFrame()

    df["time"] = pd.to_datetime(
        df[time_col],
        errors="coerce",
        utc=True
    )

    # Remove timezone
    try:
        df["time"] = df["time"].dt.tz_localize(None)
    except Exception:
        pass

    # -------------------------
    # PM2.5 COLUMN
    # -------------------------

    pm_col = find_column(
        df,
        [
            "pm2.5",
            "pm25",
            "pm2_5",
            "pm_2_5",
            "pm 2.5",
            "pm2"
        ]
    )

    if pm_col is None:
        return pd.DataFrame()

    df["pm25"] = pd.to_numeric(
        df[pm_col],
        errors="coerce"
    )

    # -------------------------
    # CITY COLUMN
    # -------------------------

    city_col = find_column(
        df,
        [
            "city",
            "location",
            "place",
            "site"
        ]
    )

    if forced_city is not None:

        df["city"] = forced_city

    elif city_col is not None:

        df["city"] = (
            df[city_col]
            .astype(str)
            .str.strip()
        )

    else:

        df["city"] = "Unknown"

    # -------------------------
    # CLEAN
    # -------------------------

    df = df[
        ["time", "city", "pm25"]
    ].copy()

    df = df.dropna(
        subset=["time", "pm25"]
    )

    # Remove impossible PM2.5 values
    df = df[
        (df["pm25"] >= 0) &
        (df["pm25"] <= 1000)
    ]

    df = df.sort_values("time")

    return df


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_all_data():

    datasets = []

    # -------------------------
    # ALMATY
    # -------------------------

    almaty = read_file(ALMATY_FILE)

    if almaty is not None:

        almaty = standardize_dataset(
            almaty,
            forced_city="Almaty"
        )

        if not almaty.empty:
            datasets.append(almaty)

    # -------------------------
    # GLOBAL DATA
    # -------------------------

    global_data = read_file(GLOBAL_FILE)

    if global_data is not None:

        global_data = standardize_dataset(
            global_data
        )

        if not global_data.empty:
            datasets.append(global_data)

    # -------------------------
    # COMBINE
    # -------------------------

    if not datasets:
        return pd.DataFrame()

    final_df = pd.concat(
        datasets,
        ignore_index=True
    )

    final_df = final_df.drop_duplicates(
        subset=["time", "city", "pm25"]
    )

    return final_df


df_full = load_all_data()


# ============================================================
# CHECK DATA
# ============================================================

if df_full.empty:

    st.error(
        "❌ Не удалось загрузить данные."
    )

    st.info(
        "Проверь, что рядом с app.py находятся "
        "air_quality_data.csv и/или "
        "Global_City_Air_Quality_Hourly.csv."
    )

    st.stop()


# ============================================================
# DATASET INFORMATION
# ============================================================

st.sidebar.header("📁 Dataset")

st.sidebar.success(
    f"Loaded records: {len(df_full):,}"
)

st.sidebar.write(
    f"Cities: {df_full['city'].nunique()}"
)


# ============================================================
# CITY SELECTION
# ============================================================

st.sidebar.header("📍 Select City")

cities = sorted(
    df_full["city"]
    .dropna()
    .unique()
)

if "Almaty" in cities:

    cities.remove("Almaty")
    cities.insert(0, "Almaty")


selected_city = st.sidebar.selectbox(
    "City:",
    cities
)


# ============================================================
# CITY DATA
# ============================================================

df_city = df_full[
    df_full["city"] == selected_city
].copy()

df_city = df_city.sort_values("time")


# ============================================================
# CHECK RECORD COUNT
# ============================================================

if len(df_city) < 50:

    st.error(
        f"❌ Недостаточно данных для {selected_city}."
    )

    st.write(
        f"Available records: {len(df_city)}"
    )

    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

df_city["hour"] = (
    df_city["time"].dt.hour
)

df_city["dayofweek"] = (
    df_city["time"].dt.dayofweek
)

df_city["month"] = (
    df_city["time"].dt.month
)

df_city["day"] = (
    df_city["time"].dt.day
)


# ============================================================
# LAG FEATURES
# ============================================================

lag_values = [
    1,
    3,
    6,
    12,
    24
]

for lag in lag_values:

    df_city[
        f"pm25_lag_{lag}"
    ] = df_city["pm25"].shift(lag)


# Remove missing lag rows
df_model = df_city.dropna().copy()


# ============================================================
# CHECK MODEL DATA
# ============================================================

if len(df_model) < 50:

    st.error(
        f"❌ После создания временных признаков "
        f"осталось только {len(df_model)} записей."
    )

    st.stop()


# ============================================================
# FEATURES
# ============================================================

features = [
    "hour",
    "dayofweek",
    "month",
    "day",
    "pm25_lag_1",
    "pm25_lag_3",
    "pm25_lag_6",
    "pm25_lag_12",
    "pm25_lag_24"
]


X = df_model[features]
y = df_model["pm25"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


with st.spinner(
    f"Training model for {selected_city}..."
):

    model.fit(
        X_train,
        y_train
    )


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# METRICS
# ============================================================

r2 = r2_score(
    y_test,
    y_pred
)

mae = mean_absolute_error(
    y_test,
    y_pred
)


# ============================================================
# MAIN METRICS
# ============================================================

st.subheader(
    f"📊 {selected_city}"
)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "R² Score",
        f"{r2:.4f}"
    )

with c2:
    st.metric(
        "MAE",
        f"{mae:.2f} µg/m³"
    )

with c3:
    st.metric(
        "Records",
        f"{len(df_city):,}"
    )


st.caption(
    f"Data period: "
    f"{df_city['time'].min().strftime('%Y-%m-%d')} "
    f"→ "
    f"{df_city['time'].max().strftime('%Y-%m-%d')}"
)


# ============================================================
# FORECAST FUNCTION
# ============================================================

def forecast_future(
    model,
    df,
    features,
    hours_ahead=3
):

    work = df.copy()

    forecasts = []

    last_time = work["time"].iloc[-1]

    # Last known PM2.5 values
    pm_values = list(
        work["pm25"].iloc[-24:].values
    )

    for step in range(
        1,
        hours_ahead + 1
    ):

        future_time = (
            last_time
            + timedelta(hours=step)
        )

        row = {
            "hour": future_time.hour,
            "dayofweek": future_time.weekday(),
            "month": future_time.month,
            "day": future_time.day
        }

        # Lag 1
        row["pm25_lag_1"] = (
            pm_values[-1]
        )

        # Lag 3
        row["pm25_lag_3"] = (
            pm_values[-3]
            if len(pm_values) >= 3
            else pm_values[-1]
        )

        # Lag 6
        row["pm25_lag_6"] = (
            pm_values[-6]
            if len(pm_values) >= 6
            else pm_values[-1]
        )

        # Lag 12
        row["pm25_lag_12"] = (
            pm_values[-12]
            if len(pm_values) >= 12
            else pm_values[-1]
        )

        # Lag 24
        row["pm25_lag_24"] = (
            pm_values[-24]
            if len(pm_values) >= 24
            else pm_values[-1]
        )

        X_future = pd.DataFrame(
            [row]
        )[features]

        prediction = model.predict(
            X_future
        )[0]

        # PM2.5 cannot be negative
        prediction = max(
            0,
            float(prediction)
        )

        forecasts.append(
            prediction
        )

        pm_values.append(
            prediction
        )

    return forecasts


# ============================================================
# FORECAST
# ============================================================

forecast_values = forecast_future(
    model,
    df_city,
    features,
    hours_ahead=3
)


# ============================================================
# EARLY WARNING
# ============================================================

st.markdown("---")

st.header(
    "⚠️ Early Warning System"
)

WARNING_THRESHOLD = 50

st.caption(
    f"Warning threshold: "
    f"{WARNING_THRESHOLD} µg/m³"
)


last_actual = float(
    df_city["pm25"].iloc[-1]
)

last_time = (
    df_city["time"].iloc[-1]
)


w1, w2, w3, w4 = st.columns(4)


with w1:

    st.metric(
        "Current",
        f"{last_actual:.1f} µg/m³"
    )


warning_triggered = False


for i, value in enumerate(
    forecast_values,
    start=1
):

    future_time = (
        last_time
        + timedelta(hours=i)
    )

    is_warning = (
        value >= WARNING_THRESHOLD
    )

    if is_warning:
        warning_triggered = True

    with [
        w2,
        w3,
        w4
    ][i - 1]:

        st.metric(
            f"{'🔴' if is_warning else '🟢'} "
            f"+{i}h "
            f"({future_time.strftime('%H:%M')})",
            f"{value:.1f} µg/m³",
            delta=f"{value - last_actual:+.1f}"
        )


if warning_triggered:

    st.error(
        "🚨 WARNING: predicted PM2.5 "
        "exceeds the selected threshold."
    )

else:

    st.success(
        "✅ No warning: predicted PM2.5 "
        "remains below the threshold."
    )


# ============================================================
# FORECAST GRAPH
# ============================================================

st.markdown("---")

st.header(
    "📈 PM2.5 Forecast — Next 3 Hours"
)

fig, ax = plt.subplots(
    figsize=(12, 5)
)

history = (
    df_city["pm25"]
    .iloc[-24:]
    .values
)

history_x = np.arange(
    len(history)
)

forecast_x = np.arange(
    len(history),
    len(history) + len(forecast_values)
)

ax.plot(
    history_x,
    history,
    label="Historical PM2.5",
    linewidth=2
)

ax.plot(
    forecast_x,
    forecast_values,
    "--o",
    label="Forecast",
    linewidth=2
)

ax.axhline(
    WARNING_THRESHOLD,
    linestyle=":",
    linewidth=2,
    label="Warning threshold"
)

ax.axvline(
    len(history) - 1,
    linestyle="--",
    linewidth=1,
    label="Current time"
)

ax.set_xlabel(
    "Hourly observations"
)

ax.set_ylabel(
    "PM2.5 (µg/m³)"
)

ax.set_title(
    f"{selected_city}: PM2.5 Forecast"
)

ax.legend()

ax.grid(
    alpha=0.3
)

st.pyplot(
    fig,
    clear_figure=True
)

plt.close(fig)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("---")

st.header(
    "📊 Model Performance"
)

fig2, ax2 = plt.subplots(
    figsize=(12, 5)
)

n = min(
    150,
    len(y_test)
)

ax2.plot(
    range(n),
    y_test.iloc[:n].values,
    label="Actual",
    linewidth=2
)

ax2.plot(
    range(n),
    y_pred[:n],
    label="Predicted",
    linewidth=2
)

ax2.set_xlabel(
    "Test observations"
)

ax2.set_ylabel(
    "PM2.5 (µg/m³)"
)

ax2.set_title(
    f"{selected_city}: Actual vs Predicted"
)

ax2.legend()

ax2.grid(
    alpha=0.3
)

st.pyplot(
    fig2,
    clear_figure=True
)

plt.close(fig2)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.markdown("---")

st.header(
    "🔍 Feature Importance"
)

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=True
    )
)

fig3, ax3 = plt.subplots(
    figsize=(10, 6)
)

ax3.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

ax3.set_xlabel(
    "Importance"
)

ax3.set_ylabel(
    "Feature"
)

ax3.set_title(
    f"{selected_city}: Random Forest Feature Importance"
)

ax3.grid(
    axis="x",
    alpha=0.3
)

st.pyplot(
    fig3,
    clear_figure=True
)

plt.close(fig3)


# ============================================================
# DATA PREVIEW
# ============================================================

st.markdown("---")

with st.expander(
    "🔎 View latest measurements"
):

    st.dataframe(
        df_city[
            ["time", "city", "pm25"]
        ].tail(20),
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Air Quality Prediction & Early Warning System "
    "© 2026 Nurikamal Bolatbay"
)
