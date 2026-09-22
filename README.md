# Crop Recommendation Model

A random forest classifier that recommends the best crop to grow given soil and weather conditions (temperature, humidity, pH, water availability, and season).

## Dataset

`Crop_recommendation.csv` — 2200 rows, 13 crop classes.

| Column | Description |
|---|---|
| `temperature` | Temperature (°C) |
| `humidity` | Relative humidity (%) |
| `ph` | Soil pH |
| `water availability` | Water availability |
| `season` | Season (rainy / winter / summer / spring) |
| `label` | Target — crop name |

Note: the dataset is imbalanced on both the target and `season`. Maize has 200 samples vs. 100 for every other crop, and `season` ranges from 600 (rainy) down to 100 (spring). The pipeline handles the label imbalance with a stratified train/test split and `class_weight='balanced'`.

## Setup

```bash
pip install pandas numpy scikit-learn joblib
```

## Pipeline

1. Load the CSV, clean up `season` (strip whitespace, lowercase).
2. Encode `label` with `LabelEncoder`, one-hot encode `season`, pass the numeric columns through unchanged.
3. Train a `RandomForestClassifier` inside a `Pipeline`, with a stratified 80/20 train/test split.
4. Tune hyperparameters with `GridSearchCV` (5-fold CV) over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`.
5. Evaluate on the held-out test set (confusion matrix + classification report).
6. Save the fitted pipeline and label encoder with `joblib`.

## Results

| Metric | Score |
|---|---|
| Baseline test accuracy | 0.989 |
| Best CV accuracy (grid search) | 0.990 |
| Tuned test accuracy | 0.989 |
| Train accuracy | 1.0 |

Train accuracy hitting 1.0 is expected for an untuned-depth random forest and isn't on its own evidence of overfitting — what matters is that test accuracy and 5-fold CV accuracy are consistent with each other (0.989 vs 0.990), which is the standard check for generalization. The high accuracy overall reflects that this dataset's classes are well separated by feature (e.g. rice needs roughly 3x the water of most other crops), not that the model is overfitting.

Weakest classes: `rice`, `jute`, and `blackgram`, each around 0.95 precision/recall — everything else is at or near 1.0.

## Files

- `code.ipynb` — full notebook: EDA, preprocessing, training, tuning, evaluation
- `crop_model.pkl` — trained pipeline (preprocessing + tuned random forest)
- `label_encoder.pkl` — fitted `LabelEncoder` for the crop labels

## Usage

```python
import joblib
import pandas as pd

model = joblib.load("crop_model.pkl")
le = joblib.load("label_encoder.pkl")

sample = pd.DataFrame([{
    "temperature": 24.5,
    "humidity": 82.3,
    "ph": 6.4,
    "water availability": 200,
    "season": "rainy"
}])

pred = model.predict(sample)
print(le.inverse_transform(pred))
```

## Possible next steps

- Compare against a depth-capped random forest or a simple logistic regression baseline to further confirm the high accuracy comes from the data, not model complexity
- Expand the hyperparameter grid if compute allows
- Cross-check `season` and `label` imbalance doesn't skew per-class metrics as the dataset grows
