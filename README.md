🌾 Crop Recommendation & Plant Height Prediction System

📌 Overview

This Machine Learning (ML)-powered system is designed to assist farmers, agronomists, and researchers by providing data-driven recommendations and predictions. It combines Classification and Regression models to optimize agricultural decisions.

🚀 Key Features

1. 🔍 Crop Recommendation

Predicts the most suitable crop to cultivate based on a variety of environmental and seasonal factors.

Inputs: Temperature (°C), Humidity (%), Soil pH, Water Availability (mm), and Season.

Output: The optimal crop (e.g., Maize, Rice, Wheat).

2. 🌱 Plant Height & Growth Prediction

Utilizes a trained regression model to forecast plant development over time.

Predictions: Plant Height (cm) and New Growth (cm).

3. 🖥️ Web Interface

A simple, interactive web application built with Flask / Streamlit for users to input environmental values and receive instant predictions.

🛠️ Tech Stack

The project is built using a robust Python-based data science and web development stack:

Language: Python 3.10+

ML & Data: Scikit-Learn, Pandas, NumPy

Visualization: Matplotlib, Seaborn

Web Framework: Flask / Streamlit

Version Control: Git & GitHub

🧠 How It Works

A. Crop Recommendation Model

This typically uses a Classification Model (e.g., Random Forest, KNN) trained on a comprehensive dataset linking environmental factors to successful crop types.

B. Plant Height Prediction Model

This employs a Regression Model to predict continuous numerical values (height and growth). It uses features like age, current height, and environmental variables.

Model Performance Highlights

The models are evaluated using common regression metrics to ensure reliability:

Metric

Height Prediction

New Growth Prediction

MAE (Mean Absolute Error)

$8.431$

$2.796$

RMSE (Root Mean Squared Error)

$10.533$

$3.239$

Note: The MAE and RMSE values indicate the average magnitude of the errors, in cm, of the predictions. Lower values are better.

📂 Project Structure

📁 crop-plant-prediction
│
├── 📁 data                  # Raw and processed datasets
├── 📁 models                # Trained ML models (e.g., .pkl files)
├── 📁 notebooks             # Exploration and model development notebooks
├── app.py                  # Main application file (Flask or Streamlit)
├── requirements.txt        # Python dependency list
├── README.md               # This file
├── crop_recommendation.py  # Script for crop prediction logic
└── height_prediction.py    # Script for plant height prediction logic


⚙️ Installation & Setup

1. Clone the repository

git clone [https://github.com/yourusername/crop-plant-prediction.git](https://github.com/yourusername/crop-plant-prediction.git)
cd crop-plant-prediction


2. Install dependencies

pip install -r requirements.txt


3. Run the application

Depending on which framework is used:

For Streamlit:

streamlit run app.py


For Flask:

python app.py


📊 Example Output

The web interface provides clear, actionable feedback to the user:

🌾 Crop Prediction System

Input:
Temperature: 25°C
Humidity: 65%
Soil pH: 6.5
Water: 100mm
Season: Rainy

🎯 Recommended Crop: Maize


📈 Future Enhancements

We are continuously working to improve the system's accuracy and utility:

Integrate rainfall prediction models.

Add fertilizer recommendations based on soil nutrient analysis.

Develop a dashboard for viewing historical predictions and analytics.

Explore Deep Learning techniques (e.g., LSTMs) for more sophisticated growth modeling.

🤝 Contributing

We welcome contributions! If you have suggestions for new features, bug fixes, or model improvements, please feel free to:

Open an Issue.

Submit a Pull Request.

📜 License

This project is licensed under the MIT License. See the LICENSE file for details.
