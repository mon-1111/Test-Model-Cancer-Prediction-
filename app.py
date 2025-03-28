import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("cancer_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🔬 Breast Cancer Prediction App")
st.markdown("Enter the values for the 30 input features below:")

# Define all 30 input features
features = [
    "Radius_mean", "Texture_mean", "Perimeter_mean", "Area_mean", "Smoothness_mean",
    "Compactness_mean", "Concavity_mean", "Concave_points_mean", "Symmetry_mean", "Fractal_dimension_mean",
    "Radius_se", "Texture_se", "Perimeter_se", "Area_se", "Smoothness_se",
    "Compactness_se", "Concavity_se", "Concave_points_se", "Symmetry_se", "Fractal_dimension_se",
    "Radius_worst", "Texture_worst", "Perimeter_worst", "Area_worst", "Smoothness_worst",
    "Compactness_worst", "Concavity_worst", "Concave_points_worst", "Symmetry_worst", "Fractal_dimension_worst"
]

# Create a dictionary to store input values
user_inputs = {}

# Layout: Display 2 columns for cleaner UI
col1, col2 = st.columns(2)
for i, feature in enumerate(features):
    with (col1 if i % 2 == 0 else col2):
        user_inputs[feature] = st.number_input(f"{feature}", value=0.0)

# Collect inputs into a single array for prediction
input_array = np.array([list(user_inputs.values())]).astype(np.float32)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_array)[0]
    st.success(f"Prediction: {'Malignant' if prediction == 1 else 'Benign'}")