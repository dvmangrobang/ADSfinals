import streamlit as st
import pandas as pd
import numpy as np
import pickle

@st.cache(allow_output_mutation=True)
def load_model():
    with open('bank_customer_churn_prediction_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

def scale_data(data):
    # Define the scaling ranges for each feature
    scaling_ranges = {
        "credit_score": (300, 850),
        "age": (18, 100),
        "tenure": (0, 10),
        "balance": (0, 250000),
        "products_number": (1, 4),
        "credit_card": (0, 1),
        "active_member": (0, 1),
        "estimated_salary": (0, 500000),
        "country_France": (0, 1),
        "country_Germany": (0, 1),
        "country_Spain": (0, 1)
    }

    scaled_data = {}
    for feature, (min_val, max_val) in scaling_ranges.items():
        scaled_val = (data[feature] - min_val) / (max_val - min_val)
        scaled_data[feature] = scaled_val

    return pd.DataFrame([scaled_data])

st.title('Bank Customer Churn Prediction App')
st.write('This app predicts customer churn using a pre-trained model.')

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
    "credit_score": credit_score,
    "gender": 0 if gender == "Female" else 1,
    "age": age,
    "tenure": tenure,
    "balance": balance,
    "products_number": num_of_products,
    "credit_card": 1 if has_credit_card == "Yes" else 0,
    "active_member": 1 if is_active_member == "Yes" else 0,
    "estimated_salary": estimated_salary,
    "country_France": 1 if geography == "France" else 0,
    "country_Germany": 1 if geography == "Germany" else 0,
    "country_Spain": 1 if geography == "Spain" else 0
}

df = scale_data(data)

# Make predictions
prediction = model.predict(df)[-1]  # Get the prediction for the last row (user input)
prediction = (prediction > 0.5)

st.write("Prediction:")
if prediction:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is likely to stay.")
