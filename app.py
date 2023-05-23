import streamlit as st
import tensorflow as tf
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('final_model.h5')
    return model

model = load_model()

st.write("""
# Iris Flower Detection
""")

sepal_length = st.number_input("Enter the Sepal Length (cm)")
sepal_width = st.number_input("Enter the Sepal Width (cm)")
petal_length = st.number_input("Enter the Petal Length (cm)")
petal_width = st.number_input("Enter the Petal Width (cm)")

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

def predict_iris_class(input_data, model):
    prediction = model.predict(input_data)
    return prediction

if st.button("Predict"):
    prediction = predict_iris_class(input_data, model)
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    result = class_names[np.argmax(prediction)]
    st.success("Predicted Iris Flower Class: " + result)
