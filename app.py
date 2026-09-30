"""Crop Suitability Classifier — Streamlit prototype (Track 1 Capstone, Problem 19).
Run:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "crop_model.joblib"
META_PATH = BASE / "model" / "model_meta.json"
if not MODEL_PATH.exists():
    MODEL_PATH, META_PATH = BASE / "crop_model.joblib", BASE / "model_meta.json"

# Physically sensible limits used for input validation
LIMITS = {
    "N": (0.0, 200.0, "kg/ha", "Nitrogen content in soil"),
    "P": (0.0, 200.0, "kg/ha", "Phosphorus content in soil"),
    "K": (0.0, 250.0, "kg/ha", "Potassium content in soil"),
    "temperature": (0.0, 50.0, "°C", "Average temperature"),
    "humidity": (0.0, 100.0, "%", "Relative humidity"),
    "ph": (0.0, 14.0, "", "Soil pH"),
    "rainfall": (0.0, 400.0, "mm", "Rainfall"),
}


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())


st.set_page_config(page_title="Crop Suitability Classifier", page_icon="🌱", layout="centered")
st.title("🌱 Crop Suitability Classifier")
st.caption("Enter soil and climate conditions to get the most suitable crop. "
           "Educational prototype — not agronomic advice.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run the notebook first to create model/crop_model.joblib.")
    st.stop()

model, meta = load_artifacts()
features, ranges, means = meta["features"], meta["feature_ranges"], meta["feature_means"]

with st.form("inputs"):
    st.subheader("Soil nutrients")
    c1, c2, c3 = st.columns(3)
    st_cols = {"N": c1, "P": c2, "K": c3}
    values = {}
    for f in ["N", "P", "K"]:
        lo, hi, unit, help_ = LIMITS[f]
        values[f] = st_cols[f].number_input(f"{f} ({unit})", min_value=lo, max_value=hi,
                                            value=float(means[f]), step=1.0, help=help_)
    st.subheader("Climate & soil pH")
    c4, c5 = st.columns(2)
    for col, f in zip([c4, c5, c4, c5], ["temperature", "humidity", "ph", "rainfall"]):
        lo, hi, unit, help_ = LIMITS[f]
        label = f"{f.capitalize() if f != 'ph' else 'pH'}" + (f" ({unit})" if unit else "")
        values[f] = col.number_input(label, min_value=lo, max_value=hi, value=float(means[f]),
                                     step=0.1, help=help_)
    submitted = st.form_submit_button("Predict suitable crop", type="primary")

if submitted:
    X = pd.DataFrame([[values[f] for f in features]], columns=features)

    # Warn when inputs are outside the range the model was trained on
    outside = [f"{f} (trained range {ranges[f][0]:.1f}–{ranges[f][1]:.1f})"
               for f in features if not ranges[f][0] <= values[f] <= ranges[f][1]]
    if outside:
        st.warning("These inputs are outside the training data range, so the prediction is less reliable: "
                   + ", ".join(outside))

    # The saved Pipeline applies the same preprocessing used during training
    pred = model.predict(X)[0]
    proba = pd.Series(model.predict_proba(X)[0], index=model.classes_).sort_values(ascending=False)

    st.success(f"### Recommended crop: **{pred.title()}**")
    st.metric("Model confidence", f"{proba.iloc[0]:.0%}")
    top = proba.head(3)
    if len(top[top > 0]) > 1:
        st.write("**Top alternatives**")
        st.bar_chart(top[top > 0].rename(lambda s: s.title()))

with st.expander("About the model"):
    m = meta["test_metrics"]
    st.write(f"**Model:** {meta['model_name']} (scikit-learn Pipeline)")
    st.write(f"**Test accuracy:** {m['Test Accuracy']:.2%} · **Macro F1:** {m['Test F1']:.2%}")
    st.write(f"**Crops covered ({len(meta['classes'])}):** " + ", ".join(c.title() for c in meta["classes"]))
    st.write("**Data:** Crop Recommendation Dataset (Ingle, Kaggle, 2020).")
