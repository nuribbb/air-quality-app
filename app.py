import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from datetime import timedelta
import os
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# PAGE
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
# FILE NAMES
# ============================================================

ALMATY_FILE = "Almatydata.xlsx"
GLOBAL_FILE = "Global_City_Air_Quality_Hourly.csv"


# ============================================================
# FILE READER
# ============================================================

def read_file(filename):

    if not os.path.exists(filename):
        return None

    try:

        # Excel
        if filename.lower().endswith((".xlsx", ".xls")):

            # Read first sheet
            df = pd.read_excel(
                filename,
                sheet_name=0
            )

            return df

        # CSV
        if filename.lower().endswith(".csv"):

            encodings = [
                "utf-8",
                "utf-8-sig",
                "latin1"
            ]

            separators = [
                ",",
                ";",
                "\t"
            ]

            for encoding in encodings:

                for sep in separators:

                    try:

                        df = pd.read_csv(
                            filename,
                            sep=sep,
                            encoding=encoding,
                            on_bad_lines="skip"
                        )

                        if df.shape[1] >= 2:
                            return df

                    except Exception:
                        continue

    except Exception:
        return None

    return None


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(df, possible_names):

    if df is None or df.empty:
        return None

    # Clean names
    columns = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    # Exact match
    for name in possible_names:

        name_lower = str(name).strip().lower()

        if name_lower in columns:
            return columns[name_lower]

    # Partial match
    for c in df.columns:

        c_lower = str(c).strip().lower()

        for name in possible_names:

            name_lower = str(name).strip().lower()

            if name_lower in c_lower:
                return c

    return None


# ============================================================
# STANDARDIZE DATA
# ============================================================

def standardize_dataset(
    df,
    forced_city=None
):

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    # Remove completely empty columns
    df = df.dropna(
        axis=1,
        how="all"
    )

    # --------------------------------------------------------
    # TIME / DATETIME
    # --------------------------------------------------------

    time_col = find_column(
        df,
        [
            "datetime",
            "date_time",
            "date time",
            "timestamp",
            "time",
            "date",
            "last_updated"
        ]
    )

    # If no obvious time column, test every column
    if time_col is None:

        best_column = None
        best_count = 0

        for c in df.columns:

            try:

                parsed = pd.to_datetime(
                    df[c],
                    errors="coerce"
                )

                count = parsed.notna().sum()

                if count > best_count:

                    best_count = count
                    best_column = c

            except Exception:
                continue

        if (
            best_column is not None
            and best_count > len(df) * 0.5
        ):
            time_col = best_column

    if time_col is None:
        return pd.DataFrame()

    # Convert time
    df["time"] = pd.to_datetime(
        df[time_col],
        errors="coerce"
    )

    # Remove timezone safely
    try:

        if hasattr(
            df["time"].dt,
            "tz"
        ):

            if df["time"].dt.tz is not None:

                df["time"] = (
                    df["time"]
                    .dt
                    .tz_localize(None)
                )

    except Exception:
        pass

    # --------------------------------------------------------
    # PM2.5
    # --------------------------------------------------------

    pm_col = find_column(
        df,
        [
            "pm2.5",
            "pm25",
            "pm2_5",
            "pm_2_5",
            "pm 2.5",
            "pm2",
            "pm_25",
            "pm25_value"
        ]
    )

    if pm_col is None:

        # Search columns containing pm
        for c in df.columns:

            name = str(c).lower()

            if (
                "pm2.5" in name
                or "pm25" in name
                or "pm2_5" in name
            ):

                pm_col = c
                break

    if pm_col is None:
        return pd.DataFrame()

    df["pm25"] = pd.to_numeric(
        df[pm_col],
        errors="coerce"
    )

    # --------------------------------------------------------
    # CITY
    # --------------------------------------------------------

    if forced_city is not None:

        df["city"] = forced_city

    else:

        city_col = find_column(
            df,
            [
                "city",
                "location",
                "place",
                "site",
                "city_name"
            ]
        )

        if city_col is not None:

            df["city"] = (
                df[city_col]
                .astype(str)
                .str.strip()
            )

        else:

            df["city"] = "Unknown"

    # --------------------------------------------------------
    # FINAL DATA
    # --------------------------------------------------------

    df = df[
        [
            "time",
            "city",
            "pm25"
        ]
    ].copy()

    # Remove invalid rows
    df = df.dropna(
        subset=[
            "time",
            "pm25"
        ]
    )

    # Remove impossible PM2.5
    df = df[
        (df["pm25"] >= 0)
        &
        (df["pm25"] <= 1000)
    ]

    # Sort
    df = (
        df
        .sort_values("time")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# LOAD ALL DATA
# ============================================================

@st.cache_data
def load_all_data():

    datasets = []

    # ========================================================
    # ALMATY EXCEL
    # ========================================================

    if os.path.exists(ALMATY_FILE):

        almaty_raw = read_file(
            ALMATY_FILE
        )

        if almaty_raw is not None:

            almaty = standardize_dataset(
                almaty_raw,
                forced_city="Almaty"
            )

            if not almaty.empty:

                datasets.append(
                    almaty
                )

    # ========================================================
    # GLOBAL CSV
    # ========================================================

    if os.path.exists(GLOBAL_FILE):

        global_raw = read_file(
            GLOBAL_FILE
        )

        if global_raw is not None:

            global_data = standardize_dataset(
                global_raw
            )

            if not global_data.empty:

                datasets.append(
                    global_data
                )

    # ========================================================
    # COMBINE
    # ========================================================

    if not datasets:

        return pd.DataFrame()

    final_df = pd.concat(
        datasets,
        ignore_index=True
    )

    final_df = final_df.drop_duplicates(
        subset=[
            "time",
            "city",
            "pm25"
        ]
    )

    final_df = (
        final_df
        .sort_values(
            [
                "city",
                "time"
            ]
        )
        .reset_index(drop=True)
    )

    return final_df


# ============================================================
# LOAD
# ============================================================

df_full = load_all_data()


# ============================================================
# CHECK
# ============================================================

if df_full.empty:

    st.error(
        "❌ No valid air-quality data was loaded."
    )

    st.write(
        "The application searched for:"
    )

    st.code(
        f"""
{ALMATY_FILE}
{GLOBAL_FILE}
"""
    )

    st.info(
        "Make sure the files are in the same folder as app.py."
    )

    st.stop()


# ============================================================
# SIDEBAR DATA INFO
# ============================================================

st.sidebar.header("📁 Dataset")

st.sidebar.success(
    f"Loaded records: {len(df_full):,}"
)

st.sidebar.write(
    f"Cities: {df_full['city'].nunique()}"
)

st.sidebar.write(
    f"Almaty records: "
    f"{len(df_full[df_full['city'] == 'Almaty']):,}"
)


# ============================================================
# CITY LIST
# ============================================================

cities = sorted(
    df_full["city"]
    .dropna()
    .unique()
    .tolist()
)

# Make sure Almaty appears first
if "Almaty" in cities:

    cities.remove("Almaty")
    cities.insert(0, "Almaty")


# ============================================================
# CITY SELECTION
# ============================================================

st.sidebar.header("📍 Select City")

selected_city = st.sidebar.selectbox(
    "City:",
    cities
)


# ============================================================
# SELECT CITY DATA
# ============================================================

df_city = df_full[
    df_full["city"] == selected_city
].copy()

df_city = (
    df_city
    .sort_values("time")
    .reset_index(drop=True)
)


# ============================================================
# CITY INFORMATION
# ============================================================

st.subheader(
    f"📊 {selected_city}"
)

if selected_city == "Almaty":

    st.success(
        "🇰🇿 Almaty data loaded directly from Almatydata.xlsx"
    )


st.write(
    f"Records available: **{len(df_city):,}**"
)

if not df_city.empty:

    st.write(
        f"Period: "
        f"**{df_city['time'].min().strftime('%Y-%m-%d %H:%M')}** "
        f"→ "
        f"**{df_city['time'].max().strftime('%Y-%m-%d %H:%M')}**"
    )


# ============================================================
# CHECK DATA
# ============================================================

if len(df_city) < 50:

    st.error(
        f"❌ Not enough data for {selected_city}."
    )

    st.write(
        f"Available records: {len(df_city)}"
    )

    st.dataframe(
        df_city.tail(20),
        use_container_width=True
    )

    st.stop()


# ============================================================
# TIME FEATURES
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
    ] = (
        df_city["pm25"]
        .shift(lag)
    )


# ============================================================
# MODEL DATA
# ============================================================

df_model = (
    df_city
    .dropna()
    .copy()
)


if len(df_model) < 50:

    st.error(
        "❌ Not enough records after creating lag features."
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
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================
#
# Better for time-series data:
# older observations -> training
# newer observations -> testing
#
# No synthetic data is created.
# ============================================================

split_index = int(
    len(df_model) * 0.8
)

X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]

y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)


# ============================================================
# TRAIN
# ============================================================

with st.spinner(
    f"Training Random Forest for {selected_city}..."
):

    model.fit(
        X_train,
        y_train
    )


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# METRICS DISPLAY
# ============================================================

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

    work = (
        work
        .sort_values("time")
        .reset_index(drop=True)
    )

    forecasts = []

    last_time = (
        work["time"].iloc[-1]
    )

    # We need at least 24 values
    pm_values = list(
        work["pm25"]
        .iloc[-24:]
        .astype(float)
        .values
    )

    # If less than 24, repeat first value
    if len(pm_values) < 24:

        if len(pm_values) == 0:
            return []

        first_value = pm_values[0]

        while len(pm_values) < 24:

            pm_values.insert(
                0,
                first_value
            )

    # ========================================================
    # FUTURE STEPS
    # ========================================================

    for step in range(
        1,
        hours_ahead + 1
    ):

        future_time = (
            last_time
            + timedelta(hours=step)
        )

        row = {

            "hour":
                future_time.hour,

            "dayofweek":
                future_time.weekday(),

            "month":
                future_time.month,

            "day":
                future_time.day,

            "pm25_lag_1":
                pm_values[-1],

            "pm25_lag_3":
                pm_values[-3],

            "pm25_lag_6":
                pm_values[-6],

            "pm25_lag_12":
                pm_values[-12],

            "pm25_lag_24":
                pm_values[-24]
        }

        X_future = pd.DataFrame(
            [row]
        )

        X_future = X_future[
            features
        ]

        prediction = model.predict(
            X_future
        )[0]

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


# Current
with w1:

    st.metric(
        "Current",
        f"{last_actual:.1f} µg/m³"
    )


warning_triggered = False


# Forecast
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

    columns = [
        w2,
        w3,
        w4
    ]

    with columns[i - 1]:

        st.metric(
            f"{'🔴' if is_warning else '🟢'} "
            f"+{i}h "
            f"({future_time.strftime('%H:%M')})",
            f"{value:.1f} µg/m³",
            delta=(
                f"{value - last_actual:+.1f}"
            )
        )


if warning_triggered:

    st.error(
        "🚨 WARNING: predicted PM2.5 "
        "exceeds the warning threshold."
    )

else:

    st.success(
        "✅ No warning: predicted PM2.5 "
        "remains below the warning threshold."
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
    len(history)
    + len(forecast_values)
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

    "Feature":
        features,

    "Importance":
        model.feature_importances_

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
    f"{selected_city}: "
    "Random Forest Feature Importance"
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
# LATEST MEASUREMENTS
# ============================================================

st.markdown("---")

with st.expander(
    "🔎 View latest measurements"
):

    st.dataframe(
        df_city[
            [
                "time",
                "city",
                "pm25"
            ]
        ].tail(20),
        use_container_width=True
    )


# ============================================================
# DEBUG INFORMATION
# ============================================================

with st.expander(
    "🛠️ Data diagnostics"
):

    st.write(
        "Files found:"
    )

    st.write(
        f"Almatydata.xlsx: "
        f"{os.path.exists(ALMATY_FILE)}"
    )

    st.write(
        f"Global_City_Air_Quality_Hourly.csv: "
        f"{os.path.exists(GLOBAL_FILE)}"
    )

    st.write(
        "Available cities:"
    )

    st.write(
        cities
    )

    st.write(
        "Records per city:"
    )

    st.dataframe(
        df_full
        .groupby("city")
        .size()
        .reset_index(
            name="records"
        )
        .sort_values(
            "records",
            ascending=False
        ),
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
