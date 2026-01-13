import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('house_data.csv')
    return df

# Train model
@st.cache_resource
def train_model(df):
    # Drop Id column
    df = df.drop('Id', axis=1)
    
    # Encode categorical features
    le_location = LabelEncoder()
    le_condition = LabelEncoder()
    le_garage = LabelEncoder()
    
    df['Location'] = le_location.fit_transform(df['Location'])
    df['Condition'] = le_condition.fit_transform(df['Condition'])
    df['Garage'] = le_garage.fit_transform(df['Garage'])
    
    # Split features and target
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    
    return model, metrics, le_location, le_condition, le_garage, X.columns

# Main app
st.title("🏠 House Price Predictor")
st.markdown("Predict house prices using machine learning")
st.markdown("---")

# Load data
df = load_data()

# Sidebar
with st.sidebar:
    st.header("📊 Dataset Info")
    st.write(f"Total Houses: {len(df)}")
    st.write(f"Average Price: ${df['Price'].mean():,.0f}")
    st.write(f"Min Price: ${df['Price'].min():,.0f}")
    st.write(f"Max Price: ${df['Price'].max():,.0f}")
    
    st.markdown("---")
    st.markdown("### Features")
    st.write("• Area (sq ft)")
    st.write("• Bedrooms")
    st.write("• Bathrooms")
    st.write("• Floors")
    st.write("• Year Built")
    st.write("• Location")
    st.write("• Condition")
    st.write("• Garage")

# Train button
if st.button("🚀 Train Model", type="primary"):
    with st.spinner("Training..."):
        model, metrics, le_loc, le_cond, le_gar, features = train_model(df)
        st.session_state['model'] = model
        st.session_state['metrics'] = metrics
        st.session_state['le_location'] = le_loc
        st.session_state['le_condition'] = le_cond
        st.session_state['le_garage'] = le_gar
        st.session_state['features'] = features
        st.session_state['trained'] = True
        st.success("Model trained!")

st.markdown("---")

# Show metrics
if 'metrics' in st.session_state:
    st.header("📈 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    m = st.session_state['metrics']
    
    with col1:
        st.metric("R² Score", f"{m['r2']:.3f}")
    with col2:
        st.metric("RMSE", f"${m['rmse']:,.0f}")
    with col3:
        st.metric("MAE", f"${m['mae']:,.0f}")
    with col4:
        st.metric("Accuracy", f"{m['r2']*100:.1f}%")
    
    st.markdown("---")

# Prediction
st.header("🏡 Predict House Price")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", 500, 5000, 2000, 50)
    bedrooms = st.slider("Bedrooms", 1, 5, 3)
    bathrooms = st.slider("Bathrooms", 1, 4, 2)
    floors = st.slider("Floors", 1, 3, 2)

with col2:
    year = st.number_input("Year Built", 1900, 2024, 2000)
    location = st.selectbox("Location", ['Downtown', 'Urban', 'Suburban', 'Rural'])
    condition = st.selectbox("Condition", ['Excellent', 'Good', 'Fair', 'Poor'])
    garage = st.selectbox("Garage", ['Yes', 'No'])

if st.button("💰 Predict Price", type="secondary"):
    if 'trained' not in st.session_state:
        st.warning("Train the model first!")
    else:
        # Encode inputs
        loc_encoded = st.session_state['le_location'].transform([location])[0]
        cond_encoded = st.session_state['le_condition'].transform([condition])[0]
        gar_encoded = st.session_state['le_garage'].transform([garage])[0]
        
        # Create input
        input_data = pd.DataFrame({
            'Area': [area],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'Floors': [floors],
            'YearBuilt': [year],
            'Location': [loc_encoded],
            'Condition': [cond_encoded],
            'Garage': [gar_encoded]
        })
        
        # Predict
        prediction = st.session_state['model'].predict(input_data)[0]
        
        st.markdown("---")
        st.success(f"## Predicted Price: ${prediction:,.0f}")
        
        avg = df['Price'].mean()
        if prediction > avg:
            st.info(f"📈 ${prediction - avg:,.0f} above average")
        else:
            st.info(f"📉 ${avg - prediction:,.0f} below average")

# Charts
st.markdown("---")
st.header("📊 Data Exploration")

tab1, tab2, tab3 = st.tabs(["Price Distribution", "Feature Analysis", "Data Sample"])

with tab1:
    fig = px.histogram(df, x='Price', nbins=50, title='House Price Distribution')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    feature = st.selectbox("Select Feature", 
                          ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt'])
    fig = px.scatter(df, x=feature, y='Price', title=f'{feature} vs Price')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.dataframe(df.head(50), use_container_width=True)

# Info
st.markdown("---")
with st.expander("ℹ️ How It Works"):
    st.markdown("""
    **Steps:**
    1. Load 2000 house records
    2. Encode text data (Location, Condition, Garage)
    3. Split data: 80% train, 20% test
    4. Train Linear Regression model
    5. Predict prices for new houses
    
    **Metrics:**
    - R² Score: How well model fits (1.0 = perfect)
    - RMSE: Average prediction error
    - MAE: Mean absolute error
    """)
