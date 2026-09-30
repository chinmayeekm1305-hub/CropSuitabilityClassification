# Crop Suitability Classification — Track 1 Capstone, Problem 19

Predicts the most suitable crop (22 classes) from soil nutrients (N, P, K), pH, temperature, humidity and rainfall using foundational ML models, and demonstrates it in a Streamlit app.

## Results (held-out test set, 440 rows)
| Model | CV F1 | Accuracy | Macro F1 |
|---|---|---|---|
| Logistic Regression | 0.978 | 0.984 | 0.984 |
| KNN (k=5, distance) | 0.968 | 0.982 | 0.982 |
| **Decision Tree (selected)** | **0.986** | **0.980** | **0.979** |

The Decision Tree was selected on cross-validated F1 and for interpretability.

## Project structure
```
data/Crop_recommendation.csv          dataset
Crop_Suitability_Classification.ipynb  EDA, modelling, evaluation (run top to bottom)
model/crop_model.joblib                trained Pipeline (preprocessing + model)
model/model_meta.json                  features, classes, ranges, metrics
model/model_comparison.csv             comparison of all 3 models
figures/                               charts used in the paper
app.py                                 Streamlit application
paper/Crop_Suitability_Technical_Paper.docx (+ .pdf)
requirements.txt
```

## How to run
```bash
pip install -r requirements.txt
jupyter notebook Crop_Suitability_Classification.ipynb   # optional: re-train, re-creates model/ and figures/
streamlit run app.py
```
Test input: N=90, P=42, K=43, temperature=20.9, humidity=82, pH=6.5, rainfall=203 -> **Rice**.

## Reproducibility
Fixed random seed (42), stratified 80/20 split, scaling inside a scikit-learn Pipeline (no data leakage).

## Dataset citation
Ingle, A. (2020). *Crop Recommendation Dataset*. Kaggle. https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

## Disclaimer
Educational prototype — not agronomic advice.
