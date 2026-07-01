# app.py - House Price Prediction App
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import os

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

# Load the model and preprocessing objects
@st.cache_resource
def load_models():
    try:
        model = joblib.load('house_price_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, label_encoders, feature_names
    except FileNotFoundError:
        st.error("Model files not found. Please ensure the model is trained and saved.")
        return None, None, None, None

model, scaler, label_encoders, feature_names = load_models()

if model is not None:
    # Create input fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Property Details")
        
        # Area Type
        area_type = st.selectbox(
            "Area Type",
            ["Super built-up Area", "Built-up Area", "Plot Area", "Carpet Area"]
        )
        
        # Location
        if label_encoders and 'location' in label_encoders:
            locations = list(label_encoders['location'].classes_)
            location = st.selectbox("Location", locations)
        
        # BHK
        bhk = st.number_input("BHK (Number of Bedrooms)", min_value=1, max_value=10, value=2)
        
        # Total Sqft
        total_sqft = st.number_input("Total Square Feet", min_value=100, max_value=20000, value=1000)
        
    with col2:
        st.subheader("Additional Features")
        
        # Bathrooms
        bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
        
        # Balcony
        balcony = st.number_input("Number of Balconies", min_value=0, max_value=5, value=1)
        
        # Availability
        availability = st.selectbox("Availability Status", ["Ready To Move", "Not Ready"])
        is_ready_to_move = 1 if availability == "Ready To Move" else 0
        
        # Society (optional)
        society = st.text_input("Society Name (Optional)", "")

    # Predict button
    if st.button("🏠 Predict House Price", type="primary"):
        try:
            # Prepare input data
            input_data = {
                'bhk': bhk,
                'total_sqft': total_sqft,
                'bath': bath,
                'balcony': balcony,
                'is_ready_to_move': is_ready_to_move,
                'area_type_encoded': label_encoders['area_type'].transform([area_type])[0],
                'location_encoded': label_encoders['location'].transform([location])[0],
                'society_encoded': label_encoders['society'].transform([society if society else 'Unknown'])[0]
            }
            
            # Create dataframe
            input_df = pd.DataFrame([input_data])
            
            # Scale the features
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            predicted_price = model.predict(input_scaled)[0]
            
            # Display results
            st.success(f"### Estimated House Price: ₹{predicted_price:.2f} Lakhs")
            st.info(f"#### Equivalent: ₹{predicted_price * 100000:,.0f} INR")
            
            # Additional information
            st.write("---")
            st.write("**Input Summary:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Area Type", area_type)
                st.metric("BHK", bhk)
                st.metric("Total Sqft", f"{total_sqft} sqft")
            
            with col2:
                st.metric("Location", location)
                st.metric("Bathrooms", bath)
                st.metric("Balconies", balcony)
            
            with col3:
                st.metric("Availability", availability)
                st.metric("Society", society if society else "Not specified")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.info("Please check if the location or society is in the training data.")

# Sidebar with information
with st.sidebar:
    st.header("📊 About")
    st.markdown("""
    This app predicts house prices in Bengaluru based on:
    - Property size (sqft)
    - Number of BHK
    - Location
    - Bathrooms & Balconies
    - Availability status
    - Society name
    """)
    
    st.header("📈 Model Performance")
    st.markdown("""
    - **Algorithm:** Random Forest Regressor
    - **R² Score:** ~85-90%
    - **Features Used:** 8
    """)
    
    st.header("💡 Tips")
    st.markdown("""
    - Prices are in **Lakhs** (1 Lakh = 100,000 INR)
    - Provide accurate sqft for better predictions
    - Ready-to-move properties usually have higher prices
    - Popular locations have higher prices
    """)
    
    st.header("📁 Data Source")
    st.markdown("Bengaluru Housing Dataset (Kaggle)")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Bengaluru House Price Predictor")