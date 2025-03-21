import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open("heart_disease_model.pkl", "rb"))

st.title("💓 Heart Disease Prediction App")

st.markdown("""
Enter your health information below to predict your risk of heart disease.
""")

# User inputs
age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Encode categorical values manually 
sex = 1 if sex == "Male" else 0
chest_pain = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}[chest_pain]
fasting_bs = 1 if fasting_bs == "Yes" else 0
resting_ecg = {"Normal": 0, "ST": 1, "LVH": 2}[resting_ecg]
exercise_angina = 1 if exercise_angina == "Yes" else 0
st_slope = {"Up": 0, "Flat": 1, "Down": 2}[st_slope]


user_data = np.array([[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
                       resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])

# Predict button
if st.button("Predict"):
    prediction = model.predict(user_data)
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
