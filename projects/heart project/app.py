from typing import Tuple
import streamlit as st
import pandas as pd
import joblib
import os

def locate(location:str) -> str:
    LOCATION = os.path.join(os.getcwd(),location)
    return LOCATION

model = joblib.load(locate("KNN_heart.pkl"))
scaler = joblib.load(locate("scaler.pkl"))
expected_column = joblib.load(locate("columns.pkl"))

st.title("Hlo")
st.markdown("Provide the following details")

age = st.slider("Age",18,100,40)

sex = st.selectbox("SEX",["M","F"])
chest_pain = st.selectbox("Chest pain type",["ATA", "NAP","TA","ASY"])
resting_bp = st.number_input("Resting BP (mm Hg)",80,200,120)
ol = st.number_input("Cholestrol (mg/dL)",100,600,200)
fasting_bp = st.selectbox("Fasting Blood sugar > 120 mg/dL",[0,1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVM"])
max_hr = st.slider("Max Heart Rate", 60, 220, 70)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["N", "Y"])  # N = No, Y = Yes
old_peak = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    raw_input = {
        "Age" : age,
        "RestingBP" : resting_bp,
        "Cholesterol" : ol,
        "MaxHR" : max_hr,
        "FastingBS" : fasting_bp,
        "RestingECG" : resting_ecg,
        "Sex" : sex,
        "ChestPainType" : chest_pain,
        "ExerciseAngina" : exercise_angina,
        "ST_Slope" : old_peak
    }

    df = pd.DataFrame([raw_input])
    # One-hot encode categorical variables to match training-time encoding
    df = pd.get_dummies(df, drop_first=True)

    # Add any missing columns that the model expects (set to 0)
    for col in expected_column:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training
    df = df[expected_column]

    # Ensure numeric dtype for scaler
    df = df.astype(float)

    scaled_input = scaler.transform(df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("High risk")
    else:
        st.success("Low risk")


