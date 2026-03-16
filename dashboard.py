import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# ----------------------------
# Load Dataset
# ----------------------------
data = pd.read_csv("accident_prediction_india.csv")

# ----------------------------
# Convert Time to Period
# ----------------------------
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
# Create Encoders
# ----------------------------
le_weather = LabelEncoder()
le_road = LabelEncoder()
le_time = LabelEncoder()

le_weather.fit(data["Weather Conditions"])
le_road.fit(data["Road Type"])
le_time.fit(data["Time Period"])

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
time = st.selectbox("Time Period", list(le_time.classes_))

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

    # ----------------------------
    # Risk Messages + Suggestions
    # ----------------------------
    if risk == "Low Risk":
        st.success(f"Risk Level: {risk}")
        st.info("Road conditions appear safe. Continue driving responsibly and maintain normal speed limits.")

    elif risk == "Medium Risk":
        st.warning(f"Risk Level: {risk}")
        st.info("Moderate accident risk detected. Reduce speed slightly, maintain safe distance from other vehicles, and stay alert.")

    else:
        st.error(f"Risk Level: {risk}")
        st.warning("High accident risk detected. Avoid overspeeding, increase driver attention, and drive defensively.")

    st.write("Risk Score:", round(risk_score, 2))

    # ----------------------------
    # Risk Gauge
    # ----------------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "Accident Risk Score"},
        gauge={
            'axis': {'range': [0, 1]},
            'steps': [
                {'range': [0, 0.4], 'color': "green"},
                {'range': [0.4, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig)

# ----------------------------
# State Coordinates
# ----------------------------
state_coords = {
"Uttar Pradesh":[26.8467,80.9462],
"Maharashtra":[19.7515,75.7139],
"Rajasthan":[27.0238,74.2179],
"Kerala":[10.8505,76.2711],
"Karnataka":[15.3173,75.7139],
"Tamil Nadu":[11.1271,78.6569],
"Gujarat":[22.2587,71.1924],
"Bihar":[25.0961,85.3131],
"Madhya Pradesh":[22.9734,78.6569],
"West Bengal":[22.9868,87.8550],
"Punjab":[31.1471,75.3412],
"Haryana":[29.0588,76.0856],
"Jharkhand":[23.6102,85.2799],
"Odisha":[20.9517,85.0985],
"Chhattisgarh":[21.2787,81.8661],
"Telangana":[18.1124,79.0193],
"Andhra Pradesh":[15.9129,79.7400],
"Himachal Pradesh":[31.1048,77.1734],
"Jammu and Kashmir":[33.7782,76.5762],
"Assam":[26.2006,92.9376],
"Sikkim":[27.5330,88.5122],
"Meghalaya":[25.4670,91.3662]
}

coords = []

for state in data["State Name"]:
    if state in state_coords:
        coords.append(state_coords[state])

cluster_df = pd.DataFrame(coords, columns=["lat","lon"])

# ----------------------------
# DBSCAN Clustering
# ----------------------------
if len(cluster_df) > 10:
    db = DBSCAN(eps=1.5, min_samples=5).fit(cluster_df)
    cluster_df["cluster"] = db.labels_
else:
    cluster_df["cluster"] = -1

# ----------------------------
# Map
# ----------------------------
st.header("Accident Hotspot Detection")

m = folium.Map(location=[20.59,78.96], zoom_start=5)

for _, row in cluster_df.iterrows():

    color = "red" if row["cluster"] != -1 else "gray"

    folium.CircleMarker(
        location=[row["lat"],row["lon"]],
        radius=6,
        color=color,
        fill=True,
        fill_color=color
    ).add_to(m)

folium_static(m)

# ----------------------------
# Dataset Preview
# ----------------------------
st.header("Dataset Preview")

preview = data.copy()
preview.index = preview.index + 1

st.dataframe(preview.head(20))

# ----------------------------
# Accident Statistics
# ----------------------------
st.header("Accident Statistics")

col1,col2 = st.columns(2)

with col1:
    weather_chart = px.histogram(
        data,
        x="Weather Conditions",
        color="Weather Conditions",
        title="Accidents by Weather"
    )
    st.plotly_chart(weather_chart)

with col2:
    road_chart = px.histogram(
        data,
        x="Road Type",
        color="Road Type",
        title="Accidents by Road Type"
    )
    st.plotly_chart(road_chart)

state_chart = px.histogram(
    data,
    x="State Name",
    color="State Name",
    title="Accidents by State"
)

st.plotly_chart(state_chart)
