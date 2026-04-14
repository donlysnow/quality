import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("air_quality_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Air Quality App", layout="centered")

st.title("🌍 Air Quality CO Prediction System")
st.write("Enter environmental sensor values to predict Carbon Monoxide (CO) level")

# 9 INPUTS (MATCH TRAINING)
PT08_S1 = st.number_input("PT08.S1 (CO Sensor)", value=1000.0)
NMHC = st.number_input("NMHC(GT)", value=100.0)
C6H6 = st.number_input("C6H6(GT)", value=10.0)
PT08_S2 = st.number_input("PT08.S2 (NMHC Sensor)", value=900.0)
NOx = st.number_input("NOx(GT)", value=100.0)
PT08_S3 = st.number_input("PT08.S3 (NOx Sensor)", value=1000.0)
NO2 = st.number_input("NO2(GT)", value=100.0)
PT08_S4 = st.number_input("PT08.S4 (NO2 Sensor)", value=1500.0)
PT08_S5 = st.number_input("PT08.S5 (O3 Sensor)", value=1000.0)

# Prediction
if st.button("Predict CO Level"):
    features = np.array([[PT08_S1, NMHC, C6H6, PT08_S2,
                          NOx, PT08_S3, NO2, PT08_S4, PT08_S5]])
    
    # Scale input
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)
    result = prediction[0]

    st.success(f"Predicted CO(GT): {result:.2f}")

    # Air quality interpretation
    if result < 2:
        st.info("Air Quality: Good ✅ (Safe levels)")
    elif result < 5:
        st.warning("Air Quality: Moderate ⚠️ (Caution advised)")
    else:
        st.error("Air Quality: Poor 🚨 (Harmful levels)")