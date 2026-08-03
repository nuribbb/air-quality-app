import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Air Quality Prediction & Early Warning System",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Air Quality Prediction & Early Warning System")
st.caption("Real-world air quality data • No synthetic data • No Almaty data")


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Real global dataset only
DATA_FILE = BASE_DIR / "Global_City_Air_Quality_Hourly.csv"

# Cities to use from GLOBAL dataset
# Almaty is deliberately excluded
PREFERRED_CITIES = [
    "Tokyo",
    "Los Angeles",
    "Jakarta"
]


# ============================================================
# HELPERS
# ============================================================

def normalize_column_name(column):
    """
    Converts different column naming styles into one standard format.
    Examples:
        PM2.5 -> pm25
        pm2_5 -> pm25
        PM10 -> pm10
        DateTime -> datetime
        relative_humidity -> relativehumidity
    """
    name = str(column).strip().lower()

    replacements = {
        " ": "",
        "_": "",
        "-": "",
        ".": "",
        "/": ""
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    return name


def find_column(df, possible_names):
    """
    Finds a column despite different capitalization,
    spaces, dots or underscores.
    """
    normalized = {
        normalize_column_name(col): col
        for col in df.columns
    }

    for name in possible_names:
        key = normalize_column_name(name)

        if key in normalized:
            return normalized[key]

    return None


def detect_columns(df):
    """
    Automatically detects important columns.
    """

    datetime_col = find_column(
        df,
        [
            "datetime",
            "date",
            "time",
            "timestamp",
            "date_time",
            "datetimeutc"
        ]
    )

    city_col = find_column(
        df,
        [
            "city",
            "location",
            "cityname",
            "locationname"
        ]
    )

    pm25_col = find_column(
        df,
        [
            "pm25",
            "pm2.5",
            "pm2_5",
            "pm25ugm3",
            "pm2.5ugm3"
        ]
    )

    pm10_col = find_column(
        df,
        [
            "pm10",
            "pm10ugm3"
        ]
    )

    humidity_col = find_column(
        df,
        [
            "relativehumidity",
            "humidity",
            "relative_humidity"
        ]
    )

    temperature_col = find_column(
        df,
        [
            "temperature",
            "temp",
            "temperaturec",
            "temperaturecelsius"
        ]
    )

    return {
        "datetime": datetime_col,
        "city": city_col,
        "pm25": pm25_col,
        "pm10": pm10_col,
        "humidity": humidity_col,
        "temperature": temperature_col
    }


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_global_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"File not found:\n{DATA_FILE}\n\n"
            "Put Global_City_Air_Quality_Hourly.csv "
            "in the same folder as app.py."
        )

    # Read real CSV
    df = pd.read_csv(
        DATA_FILE,
        low_memory=False
    )

    if df.empty:
        raise ValueError("The CSV file is empty.")

    detected = detect_columns(df)

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing = []

    if detected["datetime"] is None:
        missing.append("datetime/time/date")

    if detected["pm25"] is None:
        missing.append("PM2.5")

    if detected["city"] is None:
        missing.append("city/location")

    if missing:
        raise ValueError(
            "Could not detect required columns: "
            + ", ".join(missing)
            + "\n\nColumns found in your file:\n"
            + ", ".join(map(str, df.columns))
        )

    # --------------------------------------------------------
    # Rename columns to standard names
    # --------------------------------------------------------

    rename_dict = {
        detected["datetime"]: "datetime",
        detected["city"]: "city",
        detected["pm25"]: "pm25"
    }

    if detected["pm10"] is not None:
        rename_dict[detected["pm10"]] = "pm10"

    if detected["humidity"] is not None:
        rename_dict[detected["humidity"]] = "humidity"

    if detected["temperature"] is not None:
        rename_dict[detected["temperature"]] = "temperature"

    df = df.rename(columns=rename_dict)

    # --------------------------------------------------------
    # Convert datetime
    # --------------------------------------------------------

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "pm25",
        "pm10",
        "humidity",
        "temperature"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Remove invalid PM2.5 records
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "datetime",
            "city",
            "pm25"
        ]
    )

    # Remove impossible PM2.5 values
    df = df[
        (df["pm25"] >= 0) &
        (df["pm25"] < 1000)
    ]

    # --------------------------------------------------------
    # IMPORTANT:
    # Remove Almaty completely
    # --------------------------------------------------------

    df["city"] = df["city"].astype(str).str.strip()

    df = df[
        ~df["city"]
        .str.lower()
        .str.contains(
            "almaty|алматы",
            na=False
        )
    ]

    # Sort
    df = df.sort_values(
        ["city", "datetime"]
    ).reset_index(drop=True)

    return df


# ============================================================
# CREATE FEATURES
# ============================================================

@st.cache_data(show_spinner=False)
def prepare_data(df):

    data = df.copy()

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    data["hour"] = data["datetime"].dt.hour
    data["day"] = data["datetime"].dt.day
    data["dayofweek"] = data["datetime"].dt.dayofweek
    data["month"] = data["datetime"].dt.month

    # --------------------------------------------------------
    # PM2.5 lag features
    # --------------------------------------------------------

    data["pm25_lag_1"] = (
        data.groupby("city")["pm25"]
        .shift(1)
    )

    data["pm25_lag_3"] = (
        data.groupby("city")["pm25"]
        .shift(3)
    )

    data["pm25_lag_6"] = (
        data.groupby("city")["pm25"]
        .shift(6)
    )

    data["pm25_lag_12"] = (
        data.groupby("city")["pm25"]
        .shift(12)
    )

    data["pm25_lag_24"] = (
        data.groupby("city")["pm25"]
        .shift(24)
    )

    # --------------------------------------------------------
    # Optional environmental variables
    # --------------------------------------------------------

    feature_columns = [
        "hour",
        "day",
        "dayofweek",
        "month",
        "pm25_lag_1",
        "pm25_lag_3",
        "pm25_lag_6",
        "pm25_lag_12",
        "pm25_lag_24"
    ]

    if "pm10" in data.columns:
        feature_columns.append("pm10")

    if "humidity" in data.columns:
        feature_columns.append("humidity")

    if "temperature" in data.columns:
        feature_columns.append("temperature")

    # Remove rows without enough history
    data = data.dropna(
        subset=feature_columns + ["pm25"]
    ).copy()

    return data, feature_columns


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def train_model(data, feature_columns):

    X = data[feature_columns].copy()
    y = data["pm25"].copy()

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    split_index = int(
        len(data) * 0.8
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return (
        model,
        mae,
        rmse,
        r2,
        y_test,
        predictions
    )


# ============================================================
# MAIN APP
# ============================================================

try:

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    with st.spinner("Loading real air-quality data..."):
        df = load_global_data()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    with st.spinner("Preparing data..."):
        prepared_df, feature_columns = prepare_data(df)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    with st.spinner("Training Random Forest model..."):
        (
            model,
            mae,
            rmse,
            r2,
            y_test,
            predictions
        ) = train_model(
            prepared_df,
            feature_columns
        )

except Exception as e:

    st.error("❌ Application could not load the data.")

    st.code(
        str(e),
        language="text"
    )

    st.info(
        "Make sure Global_City_Air_Quality_Hourly.csv "
        "is located in the same folder as app.py."
    )

    st.stop()


# ============================================================
# DATA OVERVIEW
# ============================================================

st.success(
    f"✅ Real dataset loaded successfully: "
    f"{len(df):,} records"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Records",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Cities",
        f"{df['city'].nunique():,}"
    )

with col3:
    st.metric(
        "MAE",
        f"{mae:.2f}"
    )

with col4:
    st.metric(
        "R²",
        f"{r2:.3f}"
    )


# ============================================================
# CITY SELECTION
# ============================================================

st.subheader("🌆 City Analysis")

available_cities = sorted(
    df["city"]
    .dropna()
    .unique()
    .tolist()
)

# Make sure Almaty isn't accidentally displayed
available_cities = [
    city
    for city in available_cities
    if "almaty" not in str(city).lower()
    and "алматы" not in str(city).lower()
]

default_city = None

for city in PREFERRED_CITIES:
    if city in available_cities:
        default_city = city
        break

if default_city is None and available_cities:
    default_city = available_cities[0]

if not available_cities:

    st.error(
        "No cities were found in the global dataset."
    )

    st.stop()


selected_city = st.selectbox(
    "Select city",
    available_cities,
    index=(
        available_cities.index(default_city)
        if default_city in available_cities
        else 0
    )
)


# ============================================================
# CITY DATA
# ============================================================

city_data = prepared_df[
    prepared_df["city"] == selected_city
].copy()

if city_data.empty:

    st.warning(
        "There are not enough records for this city "
        "after creating lag features."
    )

    st.stop()


# ============================================================
# CITY STATISTICS
# ============================================================

st.subheader(
    f"📊 {selected_city} — Air Quality"
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "PM2.5 Average",
        f"{city_data['pm25'].mean():.2f} µg/m³"
    )

with c2:

    st.metric(
        "PM2.5 Minimum",
        f"{city_data['pm25'].min():.2f} µg/m³"
    )

with c3:

    st.metric(
        "PM2.5 Maximum",
        f"{city_data['pm25'].max():.2f} µg/m³"
    )

with c4:

    st.metric(
        "Observations",
        f"{len(city_data):,}"
    )


# ============================================================
# PM2.5 HISTORY
# ============================================================

st.subheader(
    "📈 PM2.5 Historical Data"
)

history = city_data[
    ["datetime", "pm25"]
].copy()

history = history.sort_values(
    "datetime"
)

st.line_chart(
    history.set_index("datetime")["pm25"]
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader(
    "🤖 Machine Learning Model Performance"
)

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "MAE",
        f"{mae:.3f}"
    )

with m2:
    st.metric(
        "RMSE",
        f"{rmse:.3f}"
    )

with m3:
    st.metric(
        "R²",
        f"{r2:.3f}"
    )


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

st.subheader(
    "🎯 Actual vs Predicted PM2.5"
)

comparison = pd.DataFrame(
    {
        "Actual": y_test.values,
        "Predicted": predictions
    }
)

st.line_chart(
    comparison
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader(
    "🔎 Feature Importance"
)

importance = pd.DataFrame(
    {
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    }
)

importance = importance.sort_values(
    "Importance",
    ascending=False
)

st.bar_chart(
    importance.set_index("Feature")
)


# ============================================================
# EARLY WARNING
# ============================================================

st.subheader(
    "⚠️ Early Warning System"
)

latest_pm25 = city_data[
    "pm25"
].iloc[-1]


def classify_air_quality(value):

    if value <= 12:
        return (
            "🟢 Good",
            "Air quality is good."
        )

    elif value <= 35.4:
        return (
            "🟡 Moderate",
            "Air quality is acceptable, but sensitive groups should be cautious."
        )

    elif value <= 55.4:
        return (
            "🟠 Unhealthy for sensitive groups",
            "Sensitive groups may experience health effects."
        )

    elif value <= 150.4:
        return (
            "🔴 Unhealthy",
            "Everyone may begin to experience health effects."
        )

    elif value <= 250.4:
        return (
            "🟣 Very unhealthy",
            "Health alert: increased risk of health effects."
        )

    else:
        return (
            "⚫ Hazardous",
            "Health warning of emergency conditions."
        )


status, message = classify_air_quality(
    latest_pm25
)

st.metric(
    "Latest PM2.5",
    f"{latest_pm25:.2f} µg/m³"
)

st.warning(
    f"{status}\n\n{message}"
)


# ============================================================
# DATASET INFORMATION
# ============================================================

with st.expander(
    "📋 Dataset information"
):

    st.write(
        "The application uses real observations from "
        "Global_City_Air_Quality_Hourly.csv."
    )

    st.write(
        "Synthetic data is NOT generated."
    )

    st.write(
        "Almaty data is NOT used in this version."
    )

    st.write(
        f"Number of cities: {df['city'].nunique():,}"
    )

    st.write(
        f"Number of observations: {len(df):,}"
    )

    st.write(
        f"Period: {df['datetime'].min()} — "
        f"{df['datetime'].max()}"
    )

    st.write(
        "Cities included:"
    )

    st.write(
        ", ".join(
            sorted(
                df["city"].unique()
            )
        )
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Air Quality Prediction & Early Warning System | "
    "Random Forest Regression | Real-world data only"
)
