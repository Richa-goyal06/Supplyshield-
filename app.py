
import streamlit as st
import pandas as pd
from joblib import load

# Load saved files
model = load("model.pkl")
scaler = load("scaler.pkl")
columns = load("columns.pkl")

# Title
st.title("Supply Chain Disruption Predictor")

st.write("Enter shipment details to predict disruption risk.")

# -----------------------------
# User Inputs
# -----------------------------

distance = st.number_input(
    "Distance (km)",
    min_value=0.0
)

weight = st.number_input(
    "Weight (MT)",
    min_value=0.0
)

fuel_price = st.number_input(
    "Fuel Price Index"
)

geo_risk = st.slider(
    "Geopolitical Risk Score",
    0.0,
    1.0
)

carrier = st.slider(
    "Carrier Reliability Score",
    0.0,
    1.0
)

lead_time = st.number_input(
    "Lead Time (Days)",
    min_value=0
)

transport = st.selectbox(
    "Transport Mode",
    ["Air", "Sea", "Road", "Rail"]
)

weather = st.selectbox(
    "Weather Condition",
    ["Clear", "Rain", "Storm"]
)

# -----------------------------
# Prediction Button
# -----------------------------

if st.button("Predict"):

    # Create dataframe
    input_data = pd.DataFrame({
        'Distance_km': [distance],
        'Weight_MT': [weight],
        'Fuel_Price_Index': [fuel_price],
        'Geopolitical_Risk_Score': [geo_risk],
        'Carrier_Reliability_Score': [carrier],
        'Lead_Time_Days': [lead_time],
        'Transport_Mode': [transport],
        'Weather_Condition': [weather]
    })

    # Feature Engineering
    input_data['Risk_x_Fuel'] = (
        input_data['Geopolitical_Risk_Score']
        * input_data['Fuel_Price_Index']
    )

    input_data['Carrier_Unreliability'] = (
        1 - input_data['Carrier_Reliability_Score']
    )

    input_data['Weight_per_km'] = (
        input_data['Weight_MT']
        / (input_data['Distance_km'] + 1)
    )

    # One-hot encoding
    input_data = pd.get_dummies(input_data)

    # Add missing columns
    missing_cols = set(columns) - set(input_data.columns)

    for col in missing_cols:
        input_data[col] = 0

    # Correct column order
    input_data = input_data[columns]

    # Scale numerical columns
    scale_cols = [
        'Distance_km',
        'Weight_MT',
        'Fuel_Price_Index',
        'Geopolitical_Risk_Score',
        'Lead_Time_Days'
    ]

    input_data[scale_cols] = scaler.transform(
        input_data[scale_cols]
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    # Output
    st.subheader("Prediction Result")

    st.write(
        f"Disruption Probability: {probability*100:.2f}%"
    )

    if prediction == 1:
        st.error("High Risk of Disruption")
    else:
        st.success("Low Risk of Disruption")

