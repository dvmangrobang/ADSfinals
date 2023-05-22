import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('bank_customer_churn_prediction.h5')
    return model

model = load_model()

st.title('Bank Customer Churn Prediction App')
st.write('This app predicts customer churn using a deep learning model.')

st.write("Please enter the following information:")

credit_score = st.slider("Credit Score", 300, 850, step=1)
age = st.slider("Age", 18, 100, step=1)
tenure = st.slider("Tenure (Number of years)", 0, 10, step=1)
balance = st.number_input("Account Balance")
num_of_products = st.slider("Number of Products", 1, 4, step=1)
has_credit_card = st.selectbox("Has Credit Card", ["No", "Yes"])
is_active_member = st.selectbox("Is Active Member", ["No", "Yes"])
estimated_salary = st.number_input("Estimated Salary")

gender = st.radio("Gender", ["Female", "Male"])
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

# Prepare input data
data = {
    "credit_score": [credit_score],
    "gender": [0 if gender == "Female" else 1],
    "age": [age],
    "tenure": [tenure],
    "balance": [balance],
    "products_number": [num_of_products],
    "credit_card": [1 if has_credit_card == "Yes" else 0],
    "active_member": [1 if is_active_member == "Yes" else 0],
    "estimated_salary": [estimated_salary],
    "country_France": [1 if geography == "France" else 0],
    "country_Germany": [1 if geography == "Germany" else 0],
    "country_Spain": [1 if geography == "Spain" else 0]
}

df = pd.DataFrame(data)

# Scale the input data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Make predictions
prediction = model.predict(scaled_data)[-1]  # Get the prediction for the last row (user input)
prediction = (prediction > 0.5)

st.write("Prediction:")
if prediction:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is likely to stay.")
