pip install streamlit pandas tensorflow scikit-learn imbalanced-learn matplotlib

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

<<<<<<< HEAD
st.write("Prediction:")
if prediction:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is likely to stay.")
=======
    # Count the values in y_resampled
    counter = Counter(y_resampled)
    st.write("Customer Churn after resampling:")
    st.write(counter)

    # Create a donut chart after resampling
    labels = counter.keys()
    counts = counter.values()
    percentages = [count / sum(counts) * 100 for count in counts]

    fig, ax = plt.subplots()
    ax.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4))
    ax.set_title('Customer Churn')
    circle = plt.Circle((0, 0), 0.3, color='white')
    ax.add_artist(circle)
    ax.axis('equal')
    st.pyplot(fig)

    # Transform data using MinMaxScaler, range 0 to 1
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.25, random_state=250)

    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=16, activation='relu'),
        tf.keras.layers.Dense(units=8, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32)

    # Evaluate the model
    results = model.evaluate(X_test, y_test, verbose=1)
    st.write('Test loss, Test accuracy:', results)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5) # Convert probabilities to binary predictions

    st.write("Predictions:")
    st.write(y_pred)


if __name__ == '__main__':
    main()
>>>>>>> 1e379da26159bc0f82466e033319ec5e77156712
