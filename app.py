import streamlit as st
import pickle
import numpy as np

# Load the trained Gradient Boosting model
filename = 'gradient_boosting_model1.pkl'
with open(filename, 'rb') as file:
    gb_model = pickle.load(file)

# Streamlit app
st.title("Diet Plan Recommendation System")

# User inputs
st.header("Input Parameters")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 0, 100, 25)
height = st.number_input("Height (cm)", min_value=0, max_value=300, value=170)
weight = st.number_input("Weight (kg)", min_value=0, max_value=300, value=70)

# Convert categorical data to numerical
gender = 1 if gender == "Male" else 0

# Prepare input data for prediction
input_data = np.array([[gender, age, height, weight]])

label_data=["Bodybuilding","Cycling","HIIT","Jogging","Pilates","Powerlifting","Resistance Training","Running","Strength Training","Swimming","Weight Lifting","Yoga"]

# Make prediction
if st.button("Predict"):
    #prediction=[]
    prediction = gb_model.predict(input_data)
    st.write(f"Recommended Diet and Exercise: {label_data[int(prediction[0])]}")


# Sample test to check the model
st.header("Sample Test")
if st.button("Run Sample Test"):
    sample_data = np.array([[1, 25, 170, 70]])  # Sample data for a 25-year-old male with 170 cm height and 70 kg weight
    sample_prediction = gb_model.predict(sample_data)
    st.write(f"Sample Test Prediction (1, 25, 170, 70): {sample_prediction[0]}")
