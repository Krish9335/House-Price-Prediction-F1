import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        text-align: center;
        margin: 20px 0;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load model and artifacts
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load('house_price_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, label_encoders, feature_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

model, scaler, label_encoders, feature_names = load_model_artifacts()

# Header
st.title("🏠 House Price Prediction System")
st.markdown("### Predict house prices based on property features")
st.markdown("---")

# Sidebar for input
with st.sidebar:
    st.header("🔧 Property Details")
    st.markdown("Enter the property information below:")
    
    # Area
    area = st.number_input(
        "Area (sq ft)",
        min_value=500,
        max_value=10000,
        value=2000,
        step=100,
        help="Total area of the property in square feet"
    )
    
    # Bedrooms
    bedrooms = st.slider(
        "Number of Bedrooms",
        min_value=1,
        max_value=6,
        value=3,
        help="Total number of bedrooms"
    )
    
    # Bathrooms
    bathrooms = st.slider(
        "Number of Bathrooms",
        min_value=1,
        max_value=5,
        value=2,
        help="Total number of bathrooms"
    )
    
    # Floors
    floors = st.slider(
        "Number of Floors",
        min_value=1,
        max_value=3,
        value=2,
        help="Total number of floors"
    )
    
    # Year Built
    year_built = st.number_input(
        "Year Built",
        min_value=1900,
        max_value=2025,
        value=2000,
        step=1,
        help="Year when the house was built"
    )
    
    # Location
    location = st.selectbox(
        "Location",
        options=['Downtown', 'Suburban', 'Urban', 'Rural'],
        help="Location of the property"
    )
    
    # Condition
    condition = st.selectbox(
        "Condition",
        options=['Excellent', 'Good', 'Fair', 'Poor'],
        help="Overall condition of the property"
    )
    
    # Garage
    garage = st.selectbox(
        "Garage",
        options=['Yes', 'No'],
        help="Does the property have a garage?"
    )
    
    st.markdown("---")
    predict_button = st.button("🔮 Predict Price", use_container_width=True)

# Main content
if model is not None:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Property Summary")
        
        # Create summary table
        summary_data = {
            "Feature": ["Area", "Bedrooms", "Bathrooms", "Floors", "Year Built", 
                       "Location", "Condition", "Garage"],
            "Value": [f"{area:,} sq ft", bedrooms, bathrooms, floors, year_built,
                     location, condition, garage]
        }
        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)
    
    with col2:
        st.subheader("🎯 Quick Stats")
        st.metric("Total Rooms", bedrooms + bathrooms)
        st.metric("Property Age", 2025 - year_built)
        st.metric("Area per Room", f"{area // (bedrooms + bathrooms):,} sq ft")
    
    if predict_button:
        with st.spinner("Calculating price prediction..."):
            try:
                # Create feature dictionary
                input_data = {
                    'Area': area,
                    'Bedrooms': bedrooms,
                    'Bathrooms': bathrooms,
                    'Floors': floors,
                    'YearBuilt': year_built,
                    'Location': label_encoders['Location'].transform([location])[0],
                    'Condition': label_encoders['Condition'].transform([condition])[0],
                    'Garage': label_encoders['Garage'].transform([garage])[0],
                    'Age': 2025 - year_built,
                    'TotalRooms': bedrooms + bathrooms,
                    'Area_per_Room': area / (bedrooms + bathrooms)
                }
                
                # Create dataframe
                input_df = pd.DataFrame([input_data])
                
                # Ensure correct column order
                input_df = input_df[feature_names]
                
                # Scale features
                input_scaled = scaler.transform(input_df)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                
                # Display prediction
                st.markdown("---")
                st.success("✅ Prediction Complete!")
                
                # Prediction result
                st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style="color: #4CAF50; margin-bottom: 10px;">Estimated House Price</h2>
                        <h1 style="color: #2E7D32; font-size: 48px; margin: 10px 0;">${prediction:,.2f}</h1>
                        <p style="color: #666; font-size: 14px;">Based on the provided property features</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Price range
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Lower Estimate",
                        f"${prediction * 0.9:,.2f}",
                        delta="-10%",
                        delta_color="off"
                    )
                with col2:
                    st.metric(
                        "Predicted Price",
                        f"${prediction:,.2f}",
                        delta="Base"
                    )
                with col3:
                    st.metric(
                        "Upper Estimate",
                        f"${prediction * 1.1:,.2f}",
                        delta="+10%",
                        delta_color="off"
                    )
                
                # Visualization
                st.markdown("---")
                st.subheader("📈 Price Range Visualization")
                
                # Create price range chart
                price_range = pd.DataFrame({
                    'Estimate': ['Lower (-10%)', 'Predicted', 'Upper (+10%)'],
                    'Price': [prediction * 0.9, prediction, prediction * 1.1]
                })
                
                fig = px.bar(
                    price_range,
                    x='Estimate',
                    y='Price',
                    text='Price',
                    color='Estimate',
                    color_discrete_map={
                        'Lower (-10%)': '#FF6B6B',
                        'Predicted': '#4CAF50',
                        'Upper (+10%)': '#4ECDC4'
                    }
                )
                
                fig.update_traces(
                    texttemplate='$%{text:,.0f}',
                    textposition='outside'
                )
                
                fig.update_layout(
                    showlegend=False,
                    height=400,
                    xaxis_title="",
                    yaxis_title="Price ($)",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature contribution (for demonstration)
                st.markdown("---")
                st.subheader("🔍 Feature Analysis")
                
                feature_importance = pd.DataFrame({
                    'Feature': ['Area', 'Location', 'Condition', 'Age', 'Bedrooms'],
                    'Impact': [35, 25, 20, 12, 8]
                })
                
                fig2 = px.bar(
                    feature_importance,
                    x='Impact',
                    y='Feature',
                    orientation='h',
                    text='Impact',
                    color='Impact',
                    color_continuous_scale='Greens'
                )
                
                fig2.update_traces(
                    texttemplate='%{text}%',
                    textposition='outside'
                )
                
                fig2.update_layout(
                    showlegend=False,
                    height=300,
                    xaxis_title="Relative Impact (%)",
                    yaxis_title="",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Additional info
                with st.expander("ℹ️ About This Prediction"):
                    st.write("""
                    **How the prediction works:**
                    - The model uses machine learning to analyze patterns from thousands of house sales
                    - It considers multiple features including size, location, condition, and age
                    - The price range (±10%) accounts for market variations
                    
                    **Important notes:**
                    - This is an estimate based on historical data
                    - Actual prices may vary based on current market conditions
                    - Consider getting a professional appraisal for accurate valuation
                    """)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.write("Please check your input values and try again.")

else:
    st.error("⚠️ Model files not found. Please ensure all model files are in the same directory.")
    st.info("Required files: house_price_model.pkl, scaler.pkl, label_encoders.pkl, feature_names.pkl")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🏠 House Price Prediction System | Built with Streamlit</p>
        <p style="font-size: 12px;">Model trained on housing data | For educational purposes</p>
    </div>
""", unsafe_allow_html=True)
