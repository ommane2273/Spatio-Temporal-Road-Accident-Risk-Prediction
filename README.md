# 🚦 Spatio-Temporal Road Accident Risk Prediction

An interactive Machine Learning dashboard that predicts road accident risk based on weather conditions, road type, and time of day. The project also visualizes accident trends and identifies accident hotspots using clustering techniques.

---

## 📌 Project Overview

Road accidents are one of the leading causes of injuries and fatalities worldwide. This project leverages Machine Learning and data visualization to analyze accident patterns and predict accident risk based on different environmental and road conditions.

The application provides:
- 🚗 Accident Risk Prediction
- 📊 Interactive Analytics Dashboard
- 🗺️ Accident Hotspot Detection
- 💡 Road Safety Recommendations

---

## ✨ Features

- Predict accident risk in real-time
- Risk Score (0–1)
- Low / Moderate / High Risk Classification
- Interactive Risk Gauge
- Weather-wise Accident Analysis
- Road Type Analysis
- State-wise Accident Statistics
- Interactive India Accident Map
- DBSCAN-based Hotspot Detection
- Safety Recommendations based on predicted risk
- Responsive Streamlit Dashboard

---

## 🛠 Tech Stack

### Programming Language
- Python

### Framework
- Streamlit

### Machine Learning
- Scikit-learn
  - Random Forest Classifier
  - DBSCAN Clustering

### Libraries
- Pandas
- Plotly
- Folium
- Streamlit-Folium
- Joblib

### Deployment
- GitHub
- Streamlit Community Cloud

---

## 📂 Project Structure

```
Spatio-Temporal-Road-Accident-Risk-Prediction/
│
├── dashboard.py
├── model.pkl
├── accident_prediction_india.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

1. User selects:
   - Weather Condition
   - Road Type
   - Time Period

2. Input data is preprocessed using Label Encoding.

3. The trained Random Forest model predicts accident risk.

4. A Risk Score is generated.

5. The application classifies the result into:
   - 🟢 Low Risk
   - 🟡 Moderate Risk
   - 🔴 High Risk

6. Charts and maps are generated for better visualization.

7. DBSCAN identifies accident hotspots.

---

## 📊 Machine Learning

### Algorithm Used
- Random Forest Classifier

### Why Random Forest?
- Handles categorical data effectively
- High prediction accuracy
- Reduces overfitting
- Provides probability-based predictions

---

## 🗺️ Hotspot Detection

The project uses **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** to identify accident-prone regions from spatial accident data.

---

## 📈 Visualizations

- Risk Gauge
- Weather Analysis
- Road Type Analysis
- State-wise Accident Distribution
- Interactive India Map
- Accident Hotspots

---

## 🎯 Applications

- Smart Traffic Management
- Driver Safety Assistance
- Government Road Safety Planning
- Accident Hotspot Identification
- Data Analytics & Research
- Smart City Initiatives

---

## 🚀 Future Enhancements

- Live Weather API Integration
- Real-time Traffic Data
- GPS-based Risk Prediction
- Route Safety Analysis
- Deep Learning Models
- Mobile Application
- AI-powered Safety Recommendations

---

## 👨‍💻 Developed By

**Om Mane**

B.Tech Artificial Intelligence & Data Science

---

## 📜 License

This project is developed for educational and research purposes.
