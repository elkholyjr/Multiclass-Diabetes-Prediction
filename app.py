import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("🩺 Multiclass Diabetes Prediction System")

st.markdown("""
This app predicts whether a person is:
- **0 → Non-Diabetic**
- **1 → Diabetic**
- **2 → Predicted Diabetic**
""")

# Sidebar Input
st.header("📝 Patient Medical Information")

# Input fields
age = st.number_input("Age", min_value=2, value=50)
urea = st.number_input("Urea (mg/dL)", min_value=1, value=4.7)
cr = st.number_input("Creatinine (mg/dL)", min_value=0.1, value=61.0)
hba1c = st.number_input("HbA1c (%)", min_value=0.1, value=6.1)
tg = st.number_input("Triglycerides (TG)", min_value=0.1, value=1.8)
hdl = st.number_input("HDL Cholesterol", min_value=0.1, value=1.1)
bmi = st.number_input("BMI", min_value=1, value=25.0)

# Compute TG/HDL ratio
tg_to_hdl = tg / hdl if hdl != 0 else 0

# Predict button
if st.button("🔍 Predict"):
    input_data = np.array([[age, urea, cr, hba1c, tg, tg_to_hdl, bmi]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    label_map = {
        0: "🟢 Non-Diabetic",
        1: "🔴 Diabetic",
        2: "🟡 Predicted Diabetic"
    }

    st.subheader("🧠 Prediction Result")
    st.success(f"**Predicted Class: {prediction} → {label_map[prediction]}**")