# 🌱 Crop Recommendation System

A machine learning project that predicts the optimal crop to be grown based on various environmental parameters such as temperature, humidity, pH, and water availability. This system uses a **Random Forest Classifier** trained via a scikit-learn pipeline for robust and accurate predictions.

---

## 🚀 Project Overview

The goal of this project is to provide accurate and quick recommendations for farmers or agricultural planners, maximizing yield by matching the right crop to the prevailing environmental conditions.

### Features Used:
* `temperature`
* `humidity`
* `ph`
* `water availability`
* `season` (Categorical)

### Target Variable:
* `label` (The recommended crop type/class)

---

## ⚙️ Model Architecture

The classification model is implemented using a **scikit-learn Pipeline** to manage both the data preprocessing and the final classifier efficiently. 

[Image of a typical machine learning pipeline diagram]


### Preprocessing
A **ColumnTransformer** handles the heterogeneous data types:
1.  **Categorical Features:** The `season` column is processed using **One-Hot Encoding** (`OneHotEncoder(drop='first')`) to convert it into a numerical format suitable for the model.
2.  **Numerical Features:** All other numerical features are passed through directly (`'passthrough'`).

### Classifier
A **Random Forest Classifier** is used for its high performance and ability to handle non-linear data relationships effectively. 

[Image of a random forest classification process]


---

## ✅ Performance and Results

The model was evaluated on a test set (20% of the data) after training, yielding extremely high accuracy.

### Test Accuracy
| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | **98.93%** |

### Confusion Matrix
The confusion matrix demonstrates the model's excellent ability to correctly classify all **13 crop classes**, with only a handful of minor misclassifications across the entire test set. The highly diagonal nature confirms the high accuracy.
