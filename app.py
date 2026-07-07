
import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

model = pickle.load(open('xgb_model.pkl', 'rb'))

df = pd.read_csv('creditcard_small.csv')
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])

fraud_sample = df[df['Class'] == 1].drop('Class', axis=1).iloc[0].tolist()

st.title("Credit Card Fraud Detection")
st.write("Enter transaction details to predict fraud probability")

if 'input_data' not in st.session_state:
    st.session_state.input_data = [0.0] * 30

if st.button("Load Sample Fraud Case"):
    st.session_state.input_data = fraud_sample

time = st.number_input("Time", value=float(st.session_state.input_data[0]), format="%.6f", step=0.000001)
v_features = []
cols = st.columns(4)
for i in range(1, 29):
    with cols[(i-1) % 4]:
        v_features.append(st.number_input(f"V{i}", value=float(st.session_state.input_data[i]), format="%.6f", step=0.000001))
amount = st.number_input("Amount", value=float(st.session_state.input_data[29]), format="%.6f", step=0.000001)

if st.button("Predict"):
    features = [time] + v_features + [amount]
    prediction = model.predict_proba([features])[0][1]
    if prediction > 0.5:
        st.error(f"Fraud Detected! Probability: {prediction:.2%}")
    else:
        st.success(f"Legitimate Transaction! Fraud Probability: {prediction:.2%}")
