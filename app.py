import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
import joblib

# Load the trained model
model = joblib.load('model.h5')

# Function to preprocess the input data
def preprocess_input(data):
    # Apply the same preprocessing steps as in the training phase
    le = LabelEncoder()
    data['gender'] = le.fit_transform(data['gender'])
    data = pd.get_dummies(data, columns=['country'], prefix=['country'])
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

# Function to make predictions
def predict_churn(data):
    data_preprocessed = preprocess_input(data)
    predictions = model.predict(data_preprocessed)
    return predictions

# Streamlit app
def main():
    # Set the app title
    st.title('Bank Customer Churn Prediction')

    # Add input fields for the relevant variables
    st.header('Customer Information')
    credit_score = st.number_input('Credit Score')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age')
    tenure = st.number_input('Tenure')
    balance = st.number_input('Balance')
    products_number = st.number_input('Number of Products')
    credit_card = st.selectbox('Has Credit Card', ['No', 'Yes'])
    active_member = st.selectbox('Is Active Member', ['No', 'Yes'])
    estimated_salary = st.number_input('Estimated Salary')
    country = st.selectbox('Country', ['France', 'Germany', 'Spain'])

    # Prepare the input data as a DataFrame
    data = pd.DataFrame({
        'credit_score': [credit_score],
        'gender': [gender],
        'age': [age],
        'tenure': [tenure],
        'balance': [balance],
        'products_number': [products_number],
        'credit_card': [credit_card],
        'active_member': [active_member],
        'estimated_salary': [estimated_salary],
        'country': [country]
    })

    # Make predictions
    if st.button('Predict Churn'):
        churn_prediction = predict_churn(data)
        if churn_prediction[0] == 0:
            st.success('The customer is predicted to not churn.')
        else:
            st.warning('The customer is predicted to churn.')

# Run the app
if __name__ == '__main__':
    main()
