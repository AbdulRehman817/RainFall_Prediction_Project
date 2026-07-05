import streamlit as st
import pandas as pd
import joblib

# Load model, scaler and columns
model = joblib.load("KNN_rainfall.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("🌧️ Rainfall Prediction")
st.markdown("### Enter the weather details")

pressure = st.number_input("Pressure", value=1013.0)
temperature = st.number_input("Temperature", value=25.0)
dewpoint = st.number_input("Dew Point", value=20.0)
humidity = st.slider("Humidity (%)", 0, 100, 80)
cloud = st.slider("Cloud Cover (%)", 0, 100, 50)
sunshine = st.number_input("Sunshine (Hours)", value=5.0)
winddirection = st.slider("Wind Direction (°)", 0, 360, 180)
windspeed = st.number_input("Wind Speed (km/h)", value=15.0)

if st.button("Predict Rainfall"):

    raw_input = {
        "pressure": pressure,
        "temparature": temperature,
        "dewpoint": dewpoint,
        "humidity": humidity,
        "cloud": cloud,
        "sunshine": sunshine,
        "winddirection": winddirection,
        "windspeed": windspeed
    }

    input_df = pd.DataFrame([raw_input])

    # Ensure all columns exist
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns in correct order
    input_df = input_df[expected_columns]

    # Scale data
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("🌧️ Rainfall is Expected")
    else:
        st.success("☀️ No Rainfall Expected")