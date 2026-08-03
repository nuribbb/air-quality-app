import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
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
st.write("**Author: Nurikamal Bolatbay**")

st.markdown("---")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, possible_names):
    """
    Find a column using several possible names.
    """
    columns_lower = {str(c).lower().strip(): c for c in df.columns}

    for name in possible_names:
        name_lower = name.lower().strip()

        if name_lower in columns_lower:
            return columns_lower[name_lower]

    # Partial search
    for col in df.columns:
        col_lower = str(col).lower().strip()

        for name in possible_names:
            if name.lower() in col_lower:
                return col

    return None


def load_file(uploaded_file):
    """
    Load CSV or Excel file.
    """

    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()

    try:

        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            df = pd.read_excel(uploaded_file)

        else:
            st.error("Please upload a CSV or Excel file.")
            return None

        return df

    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return None


def prepare_data(df):
    """
    Prepare real air-quality data.
    No synthetic data is generated.
    """

    data = df.copy()

    # --------------------------------------------------------
    # Find important columns
    # --------------------------------------------------------

    time_col = find_column(
        data,
        [
            "datetime",
            "date",
            "time",
            "timestamp",
            "date_time"
        ]
    )

    city_col = find_column(
        data,
        [
            "city",
            "location",
            "name",
            "location_name"
        ]
    )

    pm25_col = find_column(
        data,
        [
            "pm25",
            "pm2.5",
            "pm_25",
            "pm2_5",
            "PM2.5"
        ]
    )

    temperature_col = find_column(
        data,
        [
            "temperature",
            "temp",
            "temperature_c"
        ]
    )

    humidity_col = find_column(
        data,
        [
            "humidity",
            "relative_humidity",
            "relative humidity"
        ]
    )

    # --------------------------------------------------------
    # Check PM2.5
    # --------------------------------------------------------

    if pm25_col is None:
        st.error(
            "I could not find the PM2.5 column.\n\n"
            "Expected something like: pm25, PM2.5, pm2.5, pm_25."
        )

        st.write("Columns found in your file:")
        st.write(list(data.columns))

        return None, None

    # --------------------------------------------------------
    # Datetime
    # --------------------------------------------------------

    if time_col is not None:

        data[time_col] = pd.to_datetime(
            data[time_col],
            errors="coerce"
        )

        data = data.dropna(subset=[time_col])

        data = data.sort_values(time_col)

    else:

        st.warning(
            "No datetime column was detected. "
            "Time-based features cannot be created."
        )

        data["datetime_generated"] = pd.date_range(
            start="2020-01-01",
            periods=len(data),
            freq="h"
        )

        time_col = "datetime_generated"

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    data[pm25_col] = pd.to_numeric(
        data[pm25_col],
        errors="coerce"
    )

    if temperature_col is not None:
        data[temperature_col] = pd.to_numeric(
            data[temperature_col],
            errors="coerce"
        )

    if humidity_col is not None:
        data[humidity_col] = pd.to_numeric(
            data[humidity_col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove invalid PM2.5
    # --------------------------------------------------------

    data = data.dropna(subset=[pm25_col])

    # PM2.5 cannot be negative
    data = data[data[pm25_col] >= 0]

    # --------------------------------------------------------
    # IMPORTANT:
    # Remove Almaty
    # --------------------------------------------------------

    if city_col is not None:

        city_text = data[city_col].astype(str)

        almaty_mask = city_text.str.contains(
            "almaty",
            case=False,
            na=False
        )

        removed = int(almaty_mask.sum())

        data = data[~almaty_mask].copy()

    else:

        removed = 0

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    data["hour"] = data[time_col].dt.hour
    data["day"] = data[time_col].dt.day
    data["dayofweek"] = data[time_col].dt.dayofweek
    data["month"] = data[time_col].dt.month

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    if city_col is not None:

        data = data.sort_values(
            [city_col, time_col]
        )

        data["pm25_lag_1"] = (
            data.groupby(city_col)[pm25_col]
            .shift(1)
        )

        data["pm25_lag_3"] = (
            data.groupby(city_col)[pm25_col]
            .shift(3)
        )

        data["pm25_lag_6"] = (
            data.groupby(city_col)[pm25_col]
            .shift(6)
        )

        data["pm25_lag_12"] = (
            data.groupby(city_col)[pm25_col]
            .shift(12)
        )

        data["pm25_lag_24"] = (
            data.groupby(city_col)[pm25_col]
            .shift(24)
        )

    else:

        data["pm25_lag_1"] = data[pm25_col].shift(1)
        data["pm25_lag_3"] = data[pm25_col].shift(3)
        data["pm25_lag_6"] = data[pm25_col].shift(6)
        data["pm25_lag_12"] = data[pm25_col].shift(12)
        data["pm25_lag_24"] = data[pm25_col].shift(24)

    # --------------------------------------------------------
    # Drop rows without lag values
    # --------------------------------------------------------

    data = data.dropna(
        subset=[
            "pm25_lag_1",
            "pm25_lag_3",
            "pm25_lag_6",
            "pm25_lag_12",
            "pm25_lag_24"
        ]
    )

    return data, {
        "time": time_col,
        "city": city_col,
        "pm25": pm25_col,
        "temperature": temperature_col,
        "humidity": humidity_col,
        "almaty_removed": removed
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📂 Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload real air-quality data",
    type=["csv", "xlsx", "xls"]
)


# ============================================================
# LOAD DATA
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Upload your real CSV or Excel dataset in the sidebar."
    )

    st.markdown(
        """
        ### Expected data

        The dataset should contain at least:

        - **datetime / date / timestamp**
        - **PM2.5**
        - city/location column is recommended

        Optional:

        - temperature
        - humidity
        - PM10
        - PM1
        - other environmental variables

        **No synthetic data is generated by this application.**
        """
    )

    st.stop()


raw_df = load_file(uploaded_file)

if raw_df is None:
    st.stop()


# ============================================================
# PREPARE DATA
# ============================================================

data, columns = prepare_data(raw_df)

if data is None or len(data) == 0:

    st.error(
        "After cleaning and removing Almaty, "
        "there are no usable records."
    )

    st.stop()


# ============================================================
# DATA SUMMARY
# ============================================================

st.success(
    f"Real dataset loaded successfully: "
    f"{len(data):,} usable records."
)


if columns["almaty_removed"] > 0:

    st.info(
        f"Almaty was excluded from the analysis: "
        f"{columns['almaty_removed']:,} records removed."
    )

else:

    st.info(
        "No Almaty records were found in the dataset."
    )


# ============================================================
# CITIES
# ============================================================

if columns["city"] is not None:

    cities = sorted(
        data[columns["city"]]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

else:

    cities = ["All available data"]


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("⚙️ Model settings")

if columns["city"] is not None:

    selected_city = st.sidebar.selectbox(
        "Select city",
        ["All cities"] + cities
    )

else:

    selected_city = "All available data"


n_estimators = st.sidebar.slider(
    "Number of trees",
    min_value=50,
    max_value=300,
    value=100,
    step=50
)


# ============================================================
# FILTER CITY
# ============================================================

model_data = data.copy()

if (
    columns["city"] is not None
    and selected_city != "All cities"
):

    model_data = model_data[
        model_data[columns["city"]].astype(str)
        == selected_city
    ].copy()


if len(model_data) < 100:

    st.error(
        "Not enough real observations for this selection. "
        "Choose another city or use All cities."
    )

    st.stop()


# ============================================================
# FEATURES
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


# Optional temperature
if columns["temperature"] is not None:

    feature_columns.append(
        columns["temperature"]
    )


# Optional humidity
if columns["humidity"] is not None:

    feature_columns.append(
        columns["humidity"]
    )


# ------------------------------------------------------------
# Make sure all features are numeric
# ------------------------------------------------------------

for col in feature_columns:

    model_data[col] = pd.to_numeric(
        model_data[col],
        errors="coerce"
    )


model_data[columns["pm25"]] = pd.to_numeric(
    model_data[columns["pm25"]],
    errors="coerce"
)


model_data = model_data.dropna(
    subset=feature_columns + [columns["pm25"]]
)


# ============================================================
# CHRONOLOGICAL TRAIN/TEST SPLIT
# ============================================================

model_data = model_data.sort_values(
    columns["time"]
).reset_index(drop=True)


split_index = int(
    len(model_data) * 0.80
)


train = model_data.iloc[:split_index]
test = model_data.iloc[split_index:]


X_train = train[feature_columns]
y_train = train[columns["pm25"]]

X_test = test[feature_columns]
y_test = test[columns["pm25"]]


# ============================================================
# MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=n_estimators,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)


with st.spinner("Training Random Forest on real data..."):

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )


# ============================================================
# METRICS
# ============================================================

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


# ============================================================
# HEADER METRICS
# ============================================================

st.markdown("---")

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "MAE",
        f"{mae:.2f} µg/m³"
    )

with col2:
    st.metric(
        "RMSE",
        f"{rmse:.2f} µg/m³"
    )

with col3:
    st.metric(
        "R²",
        f"{r2:.3f}"
    )

with col4:
    st.metric(
        "Test records",
        f"{len(test):,}"
    )


# ============================================================
# DATASET INFO
# ============================================================

st.markdown("---")

st.subheader("📁 Dataset")

info1, info2, info3 = st.columns(3)

with info1:

    st.metric(
        "Original records",
        f"{len(raw_df):,}"
    )

with info2:

    st.metric(
        "Usable records",
        f"{len(model_data):,}"
    )

with info3:

    st.metric(
        "Cities",
        str(len(cities))
    )


if columns["city"] is not None:

    st.write("### Cities used in the analysis")

    city_counts = (
        model_data[columns["city"]]
        .value_counts()
        .reset_index()
    )

    city_counts.columns = [
        "City",
        "Records"
    ]

    st.dataframe(
        city_counts,
        use_container_width=True
    )


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

st.markdown("---")

st.subheader(
    "📈 Actual vs Predicted PM2.5"
)

fig, ax = plt.subplots(
    figsize=(12, 5)
)

ax.plot(
    test[columns["time"]],
    y_test.values,
    label="Actual PM2.5",
    linewidth=1.5
)

ax.plot(
    test[columns["time"]],
    predictions,
    label="Predicted PM2.5",
    linewidth=1.5
)

ax.set_xlabel("Time")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title("Actual vs Predicted PM2.5")

ax.legend()

ax.grid(
    alpha=0.3
)

plt.xticks(rotation=45)

plt.tight_layout()

st.pyplot(fig)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.markdown("---")

st.subheader(
    "🔍 Feature Importance"
)

importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)


fig2, ax2 = plt.subplots(
    figsize=(10, 5)
)

ax2.barh(
    importance["Feature"],
    importance["Importance"]
)

ax2.invert_yaxis()

ax2.set_xlabel("Importance")
ax2.set_title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

st.pyplot(fig2)


# ============================================================
# EARLY WARNING
# ============================================================

st.markdown("---")

st.subheader(
    "🚨 Air Quality Early Warning"
)


# Use predicted values
max_prediction = float(
    np.max(predictions)
)

mean_prediction = float(
    np.mean(predictions)
)


warning_col1, warning_col2 = st.columns(2)


with warning_col1:

    st.metric(
        "Average predicted PM2.5",
        f"{mean_prediction:.2f} µg/m³"
    )


with warning_col2:

    st.metric(
        "Maximum predicted PM2.5",
        f"{max_prediction:.2f} µg/m³"
    )


# Simple warning threshold
if max_prediction >= 150:

    st.error(
        "🔴 HIGH AIR POLLUTION RISK: "
        "Predicted PM2.5 reaches very high levels."
    )

elif max_prediction >= 55:

    st.warning(
        "🟠 ELEVATED AIR POLLUTION: "
        "Predicted PM2.5 reaches elevated levels."
    )

elif max_prediction >= 35:

    st.warning(
        "🟡 MODERATE AIR POLLUTION."
    )

else:

    st.success(
        "🟢 Predicted PM2.5 remains relatively low."
    )


# ============================================================
# FUTURE RESEARCH
# ============================================================

st.markdown("---")

st.subheader("🔬 Future Research")

st.write(
    """
    Future research can extend this system by incorporating additional
    megacities and longer historical datasets, testing more advanced
    machine-learning approaches, and comparing models across different
    climatic and geographical conditions. This would help determine
    whether the relationships identified by the model generalize across
    cities rather than being specific to one location.
    """
)


# ============================================================
# METHODOLOGY
# ============================================================

st.markdown("---")

st.subheader("🧪 Methodology")

st.write(
    """
    The system uses real historical air-quality observations.
    No synthetic observations are generated.

    The data are cleaned, chronological features and PM2.5 lag variables
    are created, and the dataset is divided chronologically into training
    and testing subsets. A Random Forest Regression model is then trained
    to predict PM2.5 concentration.

    Model performance is evaluated using MAE, RMSE and R².
    Almaty records are excluded from the analysis.
    """
)


# ============================================================
# RAW DATA
# ============================================================

with st.expander("🔎 View processed data"):

    st.dataframe(
        model_data.head(500),
        use_container_width=True
    )


st.caption(
    "Air Quality Prediction & Early Warning System — "
    "real data only, Almaty excluded."
)
