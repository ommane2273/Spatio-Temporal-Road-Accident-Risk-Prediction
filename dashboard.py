import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import folium
from streamlit_folium import folium_static
import plotly.express as px

st.set_page_config(layout="wide")

# ----------------------------
# Load Dataset
# ----------------------------
data = pd.read_csv("accident_prediction_india.csv")

# Convert time to period
def get_period(time):
    hour = int(time.split(":")[0])
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

data["Time Period"] = data["Time of Day"].apply(get_period)

# ----------------------------
# Label Encoding
# ----------------------------
le_weather = LabelEncoder()
le_road = LabelEncoder()
le_time = LabelEncoder()
le_severity = LabelEncoder()

data['Weather Conditions'] = le_weather.fit_transform(data['Weather Conditions'])
data['Road Type'] = le_road.fit_transform(data['Road Type'])
data['Time Period'] = le_time.fit_transform(data['Time Period'])
data['Accident Severity'] = le_severity.fit_transform(data['Accident Severity'])

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("model.pkl")

# ----------------------------
# Title
# ----------------------------
st.title("Spatio-Temporal Road Accident Risk Prediction")

# ----------------------------
# Prediction Section
# ----------------------------
st.header("Accident Risk Prediction")

weather = st.selectbox("Weather Condition", list(le_weather.classes_))
road = st.selectbox("Road Type", list(le_road.classes_))
time = st.selectbox("Time Period", ["Morning","Afternoon","Evening","Night"])

if st.button("Predict Risk"):

    weather_val = le_weather.transform([weather])[0]
    road_val = le_road.transform([road])[0]
    time_val = le_time.transform([time])[0]

    prob = model.predict_proba([[weather_val, road_val, time_val]])
    risk_score = max(prob[0])

    if risk_score < 0.4:
        risk = "Low Risk"
    elif risk_score < 0.7:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    # Color-coded result
    if risk == "Low Risk":
        st.success(f"Risk Level: {risk}")
    elif risk == "Medium Risk":
        st.warning(f"Risk Level: {risk}")
    else:
        st.error(f"Risk Level: {risk}")

    st.write("Risk Score:", round(risk_score,2))

# ----------------------------
# Risk Map
# ----------------------------
st.header("Accident Risk Map")

m = folium.Map(location=[20.59,78.96], zoom_start=5)

risk_points = [
    {"city":"Mumbai","lat":19.0760,"lon":72.8777,"risk":0.82,"level":"High"},
    {"city":"Delhi","lat":28.6139,"lon":77.2090,"risk":0.74,"level":"High"},
    {"city":"Pune","lat":18.5204,"lon":73.8567,"risk":0.55,"level":"Medium"},
    {"city":"Bangalore","lat":12.9716,"lon":77.5946,"risk":0.48,"level":"Medium"},
    {"city":"Hyderabad","lat":17.3850,"lon":78.4867,"risk":0.32,"level":"Low"},
]

for point in risk_points:

    if point["level"] == "High":
        color = "red"
    elif point["level"] == "Medium":
        color = "orange"
    else:
        color = "green"

    popup = f"""
    City: {point['city']} <br>
    Risk Score: {point['risk']} <br>
    Risk Level: {point['level']}
    """

    folium.CircleMarker(
        location=[point["lat"], point["lon"]],
        radius=10,
        color=color,
        fill=True,
        fill_color=color,
        popup=popup
    ).add_to(m)

folium_static(m)

# ----------------------------
# Dataset Preview
# ----------------------------
st.header("Dataset Preview")

data.index = data.index + 1
st.dataframe(data.head(20))

# ----------------------------
# Accident Statistics Charts
# ----------------------------
st.header("Accident Statistics")

col1, col2 = st.columns(2)

with col1:
    weather_chart = px.histogram(
        data,
        x="Weather Conditions",
        title="Accidents by Weather",
        color="Weather Conditions"
    )
    st.plotly_chart(weather_chart)

with col2:
    road_chart = px.histogram(
        data,
        x="Road Type",
        title="Accidents by Road Type",
        color="Road Type"
    )
    st.plotly_chart(road_chart)

state_chart = px.histogram(
    data,
    x="State Name",
    title="Accidents by State",
    color="State Name"
)

st.plotly_chart(state_chart)
