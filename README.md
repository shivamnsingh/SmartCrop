# 🌾 Crop Recommendation & Plant Height Prediction System

[![GitHub license](https://img.shields.io/github/license/yourusername/crop-plant-prediction)](https://github.com/yourusername/crop-plant-prediction/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-orange)](https://scikit-learn.org/stable/)
[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-red)](https://streamlit.io/)

---

## 📌 Overview

This Machine Learning (ML)-powered system is designed to assist farmers, agronomists, and researchers by providing data-driven recommendations and predictions. It combines **Classification** and **Regression** models to optimize agricultural decisions.



---

## 🚀 Key Features

### 1. 🔍 Crop Recommendation
Predicts the **most suitable crop** to cultivate based on a variety of environmental and seasonal factors.
* **Inputs:** Temperature (°C), Humidity (%), Soil pH, Water Availability (mm), and Season.
* **Output:** The optimal crop (e.g., Maize, Rice, Wheat).

### 2. 🌱 Plant Height & Growth Prediction
Utilizes a trained regression model to forecast plant development over time.
* **Predictions:** **Plant Height** (cm) and **New Growth** (cm).

### 3. 🖥️ Web Interface
A simple, interactive **web application** built with **Flask / Streamlit** for users to input environmental values and receive instant predictions.

---

## 🛠️ Tech Stack

The project is built using a robust Python-based data science and web development stack:

* **Language:** **Python 3.10+**
* **ML & Data:** **Scikit-Learn**, **Pandas**, **NumPy**
* **Visualization:** **Matplotlib**, **Seaborn**
* **Web Framework:** **Flask** / **Streamlit**
* **Version Control:** **Git** & **GitHub**

---

## 🧠 How It Works

### A. Crop Recommendation Model
This typically uses a **Classification Model** (e.g., Random Forest, KNN) trained on a comprehensive dataset linking environmental factors to successful crop types.

### B. Plant Height Prediction Model
This employs a **Regression Model** to predict continuous numerical values (height and growth). It uses features like age, current height, and environmental variables.

#### Model Performance Highlights

The models are evaluated using common regression metrics to ensure reliability:

| Metric | Height Prediction | New Growth Prediction |
| :--- | :---: | :---: |
| **MAE** (Mean Absolute Error) | $8.431$ | $2.796$ |
| **RMSE** (Root Mean Squared Error) | $10.533$ | $3.239$ |

> **Note:** The MAE and RMSE values indicate the average magnitude of the errors, in cm, of the predictions. Lower values are better.

---

## 📂 Project Structure

```bash
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
