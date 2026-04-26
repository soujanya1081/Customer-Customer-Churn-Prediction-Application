import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- LOAD MODEL AND SCALER ---
model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('churn_scaler.pkl', 'rb'))

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📡")
st.title("Customer Churn Prediction 📉")
st.write("Predict if a customer will leave based on their profile and service usage.")

# --- USER INPUTS ---
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox('Gender', ('Female', 'Male'))
    senior = st.selectbox('Is Senior Citizen?', ('No', 'Yes'))
    tenure = st.slider('Tenure (Months)', 0, 72, 12)
    contract = st.selectbox('Contract Type', ('Month-to-month', 'One year', 'Two year'))
    internet = st.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))

with col2:
    monthly = st.number_input('Monthly Charges ($)', 0.0, 200.0, 65.0)
    total = st.number_input('Total Charges ($)', 0.0, 10000.0, 500.0)
    payment = st.selectbox('Payment Method', 
                          ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    tech_support = st.selectbox('Has Tech Support?', ('No', 'Yes', 'No internet service'))
    security = st.selectbox('Has Online Security?', ('No', 'Yes', 'No internet service'))

# --- MAPPING LOGIC (Must match Training Encodings) ---
data = {
    'gender': 0 if gender == 'Female' else 1,
    'SeniorCitizen': 1 if senior == 'Yes' else 0,
    'tenure': tenure,
    'MonthlyCharges': monthly,
    'TotalCharges': total,
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2}[contract],
    'PaymentMethod': {'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1, 
                      'Electronic check': 2, 'Mailed check': 3}[payment],
    'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2}[internet],
    'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2}[tech_support],
    'OnlineSecurity': {'No': 0, 'No internet service': 1, 'Yes': 2}[security]
}

# Create DataFrame in the exact order of features used during training
input_df = pd.DataFrame([data])

# --- PREDICTION ---
if st.button('Analyze Churn Risk'):
    # Apply the same scaling used in training
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ **High Risk:** Likely to Churn ({probability:.1%} probability)")
    else:
        st.success(f"✅ **Low Risk:** Likely to Stay ({1-probability:.1%} probability)")