import streamlit as st
import joblib
import numpy as np

model = joblib.load("model/diabetes_model.pkl")

st.title("🩺 Diabetes Prediction System")

st.write("Enter the patient's details below.")

preg = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=1)

if st.button("Predict"):

    features = np.array([
        [
            preg,
            glucose,
            bp,
            skin,
            insulin,
            bmi,
            dpf,
            age
        ]
    ])

    prediction = model.predict(features)

    probability = model.predict_proba(features)

    if prediction[0] == 1:
        st.error("High Risk of Diabetes")
    else:
        st.success("Low Risk of Diabetes")

    st.write(
        f"Confidence: {max(probability[0])*100:.2f}%"
    )