import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# --- CONFIG & STYLING ---
st.set_page_config(page_title="CO2 Emission Predictor", layout="wide")

@st.cache_resource
def load_model():
    # Ensure this matches your saved filename
    return joblib.load("xgb_model_co2.pkl")

model_pipeline = load_model()

# --- SIDEBAR INPUTS ---
st.sidebar.header("🚗 Vehicle Specifications")

def user_input_features():
    make = st.sidebar.selectbox("Manufacturer", ['Ford', 'Chevrolet', 'Toyota', 'Honda', 'BMW', 'Other'])
    model_name = st.sidebar.text_input("Model Name", "Corolla")
    year = st.sidebar.slider("Model Year", 1984, 2025, 2020)
    displ = st.sidebar.number_input("Engine Displacement (Liters)", 0.6, 8.4, 2.5)
    cylinders = st.sidebar.slider("Cylinders", 2, 16, 4)
    fuelType = st.sidebar.selectbox("Fuel Type", ['Regular', 'Premium', 'Diesel', 'Electricity'])
    drive = st.sidebar.selectbox("Drive Train", ['Front-Wheel Drive', 'Rear-Wheel Drive', '4-Wheel Drive', 'All-Wheel Drive'])
    trany = st.sidebar.selectbox("Transmission", ['Automatic', 'Manual'])
    vclass = st.sidebar.selectbox("Vehicle Class", ['Compact Cars', 'Subcompact Cars', 'Midsize Cars', 'Large Cars', 'SUV', 'Minivan', 'Pickup Trucks'])

    data = {
        'make': make,
        'model': model_name,
        'year': year,
        'cylinders': cylinders,
        'displ': displ,
        'fuelType': fuelType,
        'drive': drive,
        'trany': trany,
        'VClass': vclass
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- MAIN PANEL ---
st.title("🌿 AI Carbon Emission Diagnostic")
st.write("This app uses a **Trained XGBoost Regressor** to predict vehicle CO2 emissions and explains the reasoning behind the prediction.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Vehicle Summary")
    st.write(input_df)
    
    if st.button("Predict CO2 Emission"):
        prediction = model_pipeline.predict(input_df)
        st.metric(label="Predicted CO2 (g/mile)", value=f"{round(float(prediction[0]), 2)}")
        
        # --- EXPLAINABILITY (SHAP) ---
        with col2:
            st.subheader("🔍 Prediction Breakdown (SHAP)")
            # Extract model and preprocessor from pipeline
            model = model_pipeline.named_steps['model']
            preprocessor = model_pipeline.named_steps['preprocess']
            
            # Transform the input
            transformed_input = preprocessor.transform(input_df)
            
            # Use TreeExplainer for XGBoost
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(transformed_input)
            
            # Plotting
            plt.figure(figsize=(10, 4))
            shap.summary_plot(shap_values, transformed_input, feature_names=preprocessor.get_feature_names_out(), plot_type="bar")
            st.pyplot(plt.gcf())
            st.info("The chart shows which features most heavily influenced the emission prediction.")

# --- FOOTER ---
st.markdown("---")
st.caption("Data Source: EPA Fuel Economy Dataset | Built with XGBoost & SHAP")
