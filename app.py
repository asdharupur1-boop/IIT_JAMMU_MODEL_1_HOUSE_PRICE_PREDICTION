# app.py - House Price Prediction App
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import traceback

# Set page configuration
st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 Bengaluru House Price Prediction")
st.markdown("""
    Predict the price of a house in Bengaluru based on various features.
    Fill in the details below to get an estimated price.
""")

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Debug: Show directory structure
if st.sidebar.checkbox("Show Debug Info", value=False):
    st.sidebar.write("### Debug Information")
    st.sidebar.write(f"Base Directory: {BASE_DIR}")
    st.sidebar.write(f"Model Directory: {MODEL_DIR}")
    st.sidebar.write("Files in models/ folder:")
    if os.path.exists(MODEL_DIR):
        for file in os.listdir(MODEL_DIR):
            st.sidebar.write(f"  - {file}")
    else:
        st.sidebar.error(f"models/ folder not found at: {MODEL_DIR}")

# Load the model and preprocessing objects
@st.cache_resource
def load_models():
    try:
        # Define file paths
        model_path = os.path.join(MODEL_DIR, 'house_price_model.pkl')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        encoders_path = os.path.join(MODEL_DIR, 'label_encoders.pkl')
        features_path = os.path.join(MODEL_DIR, 'feature_names.pkl')
        
        # Create models directory if it doesn't exist
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            st.warning(f"Created models/ directory at: {MODEL_DIR}")
            st.info("Please train and save the model files in the models/ directory.")
            return None, None, None, None
        
        # Check if all files exist
        missing_files = []
        if not os.path.exists(model_path):
            missing_files.append("house_price_model.pkl")
        if not os.path.exists(scaler_path):
            missing_files.append("scaler.pkl")
        if not os.path.exists(encoders_path):
            missing_files.append("label_encoders.pkl")
        if not os.path.exists(features_path):
            missing_files.append("feature_names.pkl")
        
        if missing_files:
            st.error(f"❌ Missing model files: {', '.join(missing_files)}")
            st.info("Please ensure all model files are present in the 'models/' directory.")
            return None, None, None, None
        
        # Load all files
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoders = joblib.load(encoders_path)
        feature_names = joblib.load(features_path)
        
        st.success("✅ Model loaded successfully!")
        return model, scaler, label_encoders, feature_names
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.code(traceback.format_exc())
        return None, None, None, None

model, scaler, label_encoders, feature_names = load_models()

# Show helpful message if model is not loaded
if model is None:
    st.warning("""
    ### ⚠️ Model Not Loaded
    
    Please make sure you have trained and saved the model files in the `models/` directory.
    
    To train the model:
    1. Run the Jupyter notebook `notebooks/house_price_prediction.ipynb`
    2. The notebook will save the model files to the `models/` folder
    3. Refresh this page
    """)
    
    # Option to show training instructions
    with st.expander("📖 How to train the model"):
        st.markdown("""
        1. Install required packages:
        ```bash
        pip install pandas numpy scikit-learn matplotlib seaborn joblib
