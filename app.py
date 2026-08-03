import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Air Quality Prediction & Early Warning System",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌍 Air Quality Prediction & Early Warning System")
st.caption("Author: Nurikamal Bolatbay")


# ============================================================
# FILE
# ============================================================

DATA_FILE = "Global_City_Air_Quality_Hourly.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not os.path.exists(DATA_FILE):
        return None, f"File '{DATA_FILE}' was not found."

    try:

        df = pd.read_csv(DATA_FILE)

        # Remove spaces from column names
        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
            .str.lower()
        )

        return df, None

    except Exception as e:

        return None, str(e)


df, error = load_data()


if error:

    st.error("❌ Data could not be loaded.")
    st.code(error)

    st.info(
        "Please make sure Global_City_Air_Quality_Hourly.csv "
        "is located in the same folder as app.py."
    )

    st.stop()


# ============================================================
# BASIC DATA CLEANING
# ============================================================

# Find datetime column automatically
datetime_candidates = [
    "datetime",
    "date",
    "timestamp",
    "time"
]

datetime_col = None

for col in datetime_candidates:

    if col in df.columns:
        datetime_col = col
        break


if datetime_col is None:

    st.error(
        "❌ No datetime column was found in the dataset."
    )

    st.write("Available columns:")
    st.write(list(df.columns))

    st.stop()


df[datetime_col] = pd.to_datetime(
    df[datetime_col],
    errors="coerce"
)


# Remove invalid dates
df = df.dropna(subset=[datetime_col])


# ============================================================
# FIND PM2.5 COLUMN
# ============================================================

pm25_candidates = [
    "pm2.5",
    "pm25",
    "pm_2_5",
    "pm2_5"
]

pm25_col = None

for col in pm25_candidates:

    if col in df.columns:
        pm25_col = col
        break


if pm25_col is None:

    st.error("❌ PM2.5 column was not found.")

    st.write("Available columns:")
    st.write(list(df.columns))

    st.stop()


# Convert PM2.5 to numeric
df[pm25_col] = pd.to_numeric(
    df[pm25_col],
    errors="coerce"
)


# Remove invalid PM2.5 values
df = df.dropna(subset=[pm25_col])


# ============================================================
# REMOVE ALMATY COMPLETELY
# ============================================================

if "location" in df.columns:

    df = df[
        ~df["location"]
        .astype(str)
        .str.lower()
        .str.contains("almaty", na=False)
    ]


# ============================================================
# SORT DATA
# ============================================================

df = df.sort_values(datetime_col).reset_index(drop=True)


# ============================================================
# CHECK DATA
# ============================================================

if len(df) < 100:

    st.error(
        "❌ Not enough real observations for modelling."
    )

    st.write(
        f"Number of available observations: {len(df)}"
    )

    st.stop()


# ============================================================
# CITY SELECTION
# ============================================================

if "location" in df.columns:

    cities = sorted(
        df["location"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(cities) == 0:

        st.error("❌ No cities were found in the dataset.")
        st.stop()

    selected_city = st.selectbox(
        "🌆 Select a megacity",
        cities
    )

    city_df = df[
        df["location"].astype(str) == selected_city
    ].copy()

else:

    selected_city = "All available data"
    city_df = df.copy()


# ============================================================
# CITY DATA CHECK
# ============================================================

if len(city_df) < 50:

    st.warning(
        "⚠️ This city does not contain enough observations "
        "for reliable prediction."
    )

    st.stop()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

city_df = city_df.sort_values(datetime_col).copy()


# Time features
city_df["hour"] = city_df[datetime_col].dt.hour

city_df["day"] = city_df[datetime_col].dt.day

city_df["dayofweek"] = city_df[datetime_col].dt.dayofweek

city_df["month"] = city_df[datetime_col].dt.month


# Historical PM2.5 features
city_df["pm25_lag_1"] = city_df[pm25_col].shift(1)

city_df["pm25_lag_3"] = city_df[pm25_col].shift(3)

city_df["pm25_lag_6"] = city_df[pm25_col].shift(6)

city_df["pm25_lag_12"] = city_df[pm25_col].shift(12)

city_df["pm25_lag_24"] = city_df[pm25_col].shift(24)


# ============================================================
# OPTIONAL WEATHER FEATURES
# ============================================================

possible_weather_features = [
    "temperature",
    "temperature_c",
    "relativehumidity",
    "relative_humidity",
    "humidity",
    "wind_speed",
    "windspeed"
]


weather_features = []

for col in possible_weather_features:

    if col in city_df.columns:

        city_df[col] = pd.to_numeric(
            city_df[col],
            errors="coerce"
        )

        weather_features.append(col)


# ============================================================
# MODEL FEATURES
# ============================================================

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


# Add available real weather data
for col in weather_features:

    if col not in feature_columns:

        feature_columns.append(col)


# Keep only existing columns
feature_columns = [
    col for col in feature_columns
    if col in city_df.columns
]


# ============================================================
# PREPARE MODEL DATA
# ============================================================

model_df = city_df[
    feature_columns + [pm25_col]
].copy()


# Convert everything to numeric
for col in feature_columns:

    model_df[col] = pd.to_numeric(
        model_df[col],
        errors="coerce"
    )


model_df[pm25_col] = pd.to_numeric(
    model_df[pm25_col],
    errors="coerce"
)


# Remove missing values
model_df = model_df.dropna()


# ============================================================
# MODEL CHECK
# ============================================================

if len(model_df) < 100:

    st.error(
        "❌ Not enough complete historical observations "
        "for this city."
    )

    st.write(
        f"Complete observations available: {len(model_df)}"
    )

    st.stop()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

# Chronological split is used because this is time-series data.

split_index = int(
    len(model_df) * 0.8
)


train_df = model_df.iloc[:split_index]

test_df = model_df.iloc[split_index:]


X_train = train_df[feature_columns]

y_train = train_df[pm25_col]


X_test = test_df[feature_columns]

y_test = test_df[pm25_col]


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTIONS
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


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)


r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# DASHBOARD
# ============================================================

st.subheader(
    f"📊 Air Quality Analysis — {selected_city}"
)


# ============================================================
# METRIC CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Records",
        f"{len(city_df):,}"
    )


with col2:

    st.metric(
        "MAE",
        f"{mae:.2f}"
    )


with col3:

    st.metric(
        "RMSE",
        f"{rmse:.2f}"
    )


with col4:

    st.metric(
        "R²",
        f"{r2:.3f}"
    )


# ============================================================
# HISTORICAL PM2.5
# ============================================================

st.subheader(
    "📈 Historical PM2.5 Concentration"
)


fig1, ax1 = plt.subplots(
    figsize=(12, 5)
)


ax1.plot(
    city_df[datetime_col],
    city_df[pm25_col],
    linewidth=1
)


ax1.set_xlabel("Date")

ax1.set_ylabel("PM2.5")

ax1.set_title(
    f"Historical PM2.5 — {selected_city}"
)


ax1.grid(
    alpha=0.3
)


fig1.tight_layout()


st.pyplot(fig1)


plt.close(fig1)


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

st.subheader(
    "🎯 Actual vs Predicted PM2.5"
)


comparison_df = pd.DataFrame({

    "Date":
        test_df.index,

    "Actual PM2.5":
        y_test.values,

    "Predicted PM2.5":
        y_pred

})


fig2, ax2 = plt.subplots(
    figsize=(12, 5)
)


ax2.plot(
    comparison_df["Actual PM2.5"],
    label="Actual"
)


ax2.plot(
    comparison_df["Predicted PM2.5"],
    label="Predicted"
)


ax2.set_xlabel("Test observation")

ax2.set_ylabel("PM2.5")

ax2.set_title(
    f"Actual vs Predicted PM2.5 — {selected_city}"
)


ax2.legend()

ax2.grid(
    alpha=0.3
)


fig2.tight_layout()


st.pyplot(fig2)


plt.close(fig2)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader(
    "🔎 Feature Importance"
)


importance_df = pd.DataFrame({

    "Feature":
        feature_columns,

    "Importance":
        model.feature_importances_

})


importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)


fig3, ax3 = plt.subplots(
    figsize=(10, 5)
)


ax3.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)


ax3.set_xlabel(
    "Importance"
)


ax3.set_title(
    "Random Forest Feature Importance"
)


ax3.invert_yaxis()


fig3.tight_layout()


st.pyplot(fig3)


plt.close(fig3)


# ============================================================
# EARLY WARNING
# ============================================================

st.subheader(
    "⚠️ Early Warning"
)


latest_pm25 = float(
    city_df[pm25_col].iloc[-1]
)


# Simple threshold for dashboard warning
# This is a screening threshold, not a medical recommendation.

if latest_pm25 >= 35:

    st.error(
        f"High PM2.5 level detected: "
        f"{latest_pm25:.1f}"
    )

elif latest_pm25 >= 15:

    st.warning(
        f"Elevated PM2.5 level: "
        f"{latest_pm25:.1f}"
    )

else:

    st.success(
        f"Current PM2.5 level is relatively low: "
        f"{latest_pm25:.1f}"
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

st.subheader(
    "📁 Dataset Information"
)


info_col1, info_col2 = st.columns(2)


with info_col1:

    st.write(
        "**Source file:** "
        "Global_City_Air_Quality_Hourly.csv"
    )

    st.write(
        f"**Selected city:** {selected_city}"
    )

    st.write(
        f"**Observations:** {len(city_df):,}"
    )


with info_col2:

    start_date = city_df[
        datetime_col
    ].min()


    end_date = city_df[
        datetime_col
    ].max()


    st.write(
        f"**Start:** {start_date}"
    )

    st.write(
        f"**End:** {end_date}"
    )


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander(
    "🔍 View real dataset"
):

    st.dataframe(
        city_df.tail(100),
        use_container_width=True
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "🤖 Model information"
):

    st.write(
        """
        The system uses Random Forest Regression to predict
        PM2.5 concentration.

        No synthetic data are generated.

        The model uses only real observations from the
        Global_City_Air_Quality_Hourly.csv dataset.

        Historical PM2.5 lag features are used to capture
        temporal dependencies.

        The data are split chronologically:
        80% for training and 20% for testing.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Air Quality Prediction & Early Warning System | "
    "Developed by Nurikamal Bolatbay"
)
