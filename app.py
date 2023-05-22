import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt

def main():
    st.title('Bank Customer Churn Prediction App')
    st.write('This app predicts customer churn using a deep learning model.')

    # Read dataset
    df = pd.read_csv("Bank Customer Churn Prediction.csv")
    del df["customer_id"] # Delete customer_id column
    st.write("Dataset:")
    st.dataframe(df)

    # Customer churn count
    customer_churn = df["churn"].value_counts()
    st.write("Customer Churn:")
    st.write(customer_churn)

    # Create a donut chart
    labels = customer_churn.index
    counts = customer_churn.values
    percentages = counts / counts.sum() * 100

    fig, ax = plt.subplots()
    ax.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4))
    ax.set_title('Customer Churn')
    circle = plt.Circle((0, 0), 0.3, color='white')
    ax.add_artist(circle)
    ax.axis('equal')
    st.pyplot(fig)

    # Assign female=0, male=1 using LabelEncoder
    le = LabelEncoder()
    df["gender"] = le.fit_transform(df["gender"])

    # Drop churn column
    data = df.drop("churn", axis=1)

    # Convert categorical data to 1 or 0 for countries
    data = pd.get_dummies(data, columns=["country"])

    X = data.values # Features (credit_score to country_Spain column)
    y = df["churn"].values # Target (churn column)

    # Resample data using SMOTE
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)

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
---
how to save code in a file named app.py