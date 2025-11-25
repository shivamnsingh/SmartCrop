import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import os

# Configure page
st.set_page_config(
    page_title="Crop Prediction System",
    page_icon="🌾",
    layout="centered"
)

# Custom CSS for scrolling and styling
st.markdown("""
    <style>
    .main {
        max-height: 100vh;
        overflow-y: auto;
    }
    .stApp {
        max-height: 100vh;
    }
    .input-section {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .result-section {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Crop Emojis Dictionary
# -----------------------------
def get_crop_emoji(crop_name):
    """Get emoji for crop"""
    crop_emojis = {
        'blackgram': '🫘',
        'chickpea': '🫘',
        'cotton': '🌸',
        'jute': '🌿',
        'kidneybeans': '🫘',
        'lentil': '🫘',
        'maize': '🌽',
        'mothbeans': '🫘',
        'mungbean': '🫛',
        'muskmelon': '🍈',
        'pigeonpeas': '🫛',
        'rice': '🌾',
        'watermelon': '🍉',
    }
    
    crop_key = crop_name.lower().strip()
    return crop_emojis.get(crop_key, '🌱')

# -----------------------------
# Model Training/Loading Function
# -----------------------------
@st.cache_resource
def load_or_train_model():
    """Load existing model or train new one"""
    try:
        # Try to load existing model
        if os.path.exists("crop_model.pkl") and os.path.exists("label_encoder.pkl"):
            model = joblib.load("crop_model.pkl")
            le = joblib.load("label_encoder.pkl")
            return model, le, None
        else:
            # Train new model if files don't exist
            if os.path.exists("Crop_recommendation.csv"):
                data = pd.read_csv("Crop_recommendation.csv")
                X = data[['temperature', 'humidity', 'ph', 'water availability', 'season']]
                y = data['label']
                
                # Standardize season column
                X['season'] = X['season'].str.strip().str.lower()
                
                # Encode target
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                
                # Columns
                cat_cols = ['season']
                num_cols = ['temperature', 'humidity', 'ph', 'water availability']
                
                # Preprocessor
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
                        ('num', 'passthrough', num_cols)
                    ]
                )
                
                # Pipeline
                model = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', RandomForestClassifier(random_state=42))
                ])
                
                # Train model
                model.fit(X, y_encoded)
                
                # Save model
                joblib.dump(model, "crop_model.pkl")
                joblib.dump(le, "label_encoder.pkl")
                
                accuracy = model.score(X, y_encoded)
                return model, le, accuracy
            else:
                return None, None, None
    except Exception as e:
        st.error(f"Error loading/training model: {str(e)}")
        return None, None, None

# -----------------------------
# Main App
# -----------------------------
def main():
    # Header
    st.title("🌾 Crop Prediction System")
    st.markdown("### Predict the best crop based on environmental conditions")
    
    # Load model
    with st.spinner("Loading model..."):
        model, le, accuracy = load_or_train_model()
    
    if model is None:
        st.error("⚠️ Unable to load or train model. Please ensure 'Crop_recommendation.csv' exists.")
        st.info("Upload your dataset or check the file path.")
        return
    
    if accuracy is not None:
        st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")
    
    st.markdown("---")
    
    # Input Section
    st.markdown("### 📊 Input Parameters")
    
    # Temperature
    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-10.0,
        max_value=50.0,
        value=25.0,
        step=0.1
    )
    
    # Humidity
    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=65.0,
        step=0.1
    )
    
    # pH
    ph = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=6.5,
        step=0.1
    )
    
    # Water Availability
    water_availability = st.number_input(
        "Water Availability (mm)",
        min_value=0.0,
        max_value=300.0,
        value=100.0,
        step=1.0
    )
    
    # Season
    season = st.selectbox(
        "Season",
        options=["Rainy", "Spring", "Summer", "Winter"]
    )
    
    # Predict button
    predict_btn = st.button("🔮 Predict Crop", use_container_width=True)
    
    # Results Section
    if predict_btn:
            # Validate inputs
            if temperature == 0 and humidity == 0 and ph == 0 and water_availability == 0:
                st.warning("⚠️ Please enter valid values for all parameters.")
            else:
                try:
                    # Prepare input
                    input_data = pd.DataFrame({
                        'temperature': [temperature],
                        'humidity': [humidity],
                        'ph': [ph],
                        'water availability': [water_availability],
                        'season': [season.strip().lower()]
                    })
                    
                    # Make prediction
                    with st.spinner("Predicting..."):
                        prediction = model.predict(input_data)
                        predicted_crop = le.inverse_transform(prediction)[0]
                        
                        # Get prediction probabilities if available
                        try:
                            probabilities = model.predict_proba(input_data)[0]
                            top_3_idx = probabilities.argsort()[-3:][::-1]
                            
                            st.success(f"### 🎯 Recommended Crop: **{predicted_crop.title()}**")
                            st.markdown("---")
                            st.markdown("#### 🏆 Top 3 Predictions:")
                            
                            # Display top 3 with emojis
                            for i, idx in enumerate(top_3_idx):
                                crop_name = le.inverse_transform([idx])[0]
                                prob = probabilities[idx]
                                crop_emoji = get_crop_emoji(crop_name)
                                
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    # Display crop emoji (large)
                                    st.markdown(f"<h1 style='text-align: center; font-size: 80px;'>{crop_emoji}</h1>", unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"**#{i+1} {crop_name.title()}**")
                                    st.progress(prob)
                                    st.write(f"Confidence: {prob:.2%}")
                                
                                if i < 2:  # Add separator except after last item
                                    st.markdown("---")
                                
                        except:
                            st.success(f"### 🎯 Recommended Crop: **{predicted_crop.title()}**")
                        
                        # Display input summary
                        st.markdown("---")
                        st.markdown("#### Input Summary:")
                        st.write(f"🌡️ Temperature: {temperature}°C")
                        st.write(f"💧 Humidity: {humidity}%")
                        st.write(f"🧪 Soil pH: {ph}")
                        st.write(f"💦 Water: {water_availability}mm")
                        st.write(f"🍂 Season: {season}")
                        
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit")

if __name__ == "__main__":
    main()