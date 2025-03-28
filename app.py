import streamlit as st
import numpy as np
import pickle

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# Load the model
with open("cancer_model.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("🩺 Breast Cancer Risk Prediction")
st.markdown("""
Welcome to the **Breast Cancer Prediction App**.
This tool uses a logistic regression model trained on the Wisconsin Breast Cancer dataset to predict whether a tumor is **benign** or **malignant** based on cell features.
""")

# Sidebar Inputs
st.sidebar.header("📊 Input Features")

features = [
    "Radius_mean", "Texture_mean", "Perimeter_mean", "Area_mean", "Smoothness_mean",
    "Compactness_mean", "Concavity_mean", "Concave_points_mean", "Symmetry_mean", "Fractal_dimension_mean",
    "Radius_se", "Texture_se", "Perimeter_se", "Area_se", "Smoothness_se",
    "Compactness_se", "Concavity_se", "Concave_points_se", "Symmetry_se", "Fractal_dimension_se",
    "Radius_worst", "Texture_worst", "Perimeter_worst", "Area_worst", "Smoothness_worst",
    "Compactness_worst", "Concavity_worst", "Concave_points_worst", "Symmetry_worst", "Fractal_dimension_worst"
]

user_inputs = {}
for feature in features:
    user_inputs[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

# Prediction button
st.markdown("### 🧠 Prediction Result")
if st.button("Predict"):
    input_array = np.array([list(user_inputs.values())]).astype(np.float32)
    prediction = model.predict(input_array)[0]
    result = "🟢 Benign" if prediction == 0 else "🔴 Malignant"
    st.subheader(f"Prediction: {result}")
    if prediction == 1:
        st.warning("⚠️ The model predicts a *malignant* tumor. Please consult a healthcare provider.")
    else:
        st.success("✅ The model predicts a *benign* tumor.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit. Model trained on the Breast Cancer Wisconsin Diagnostic dataset.")
