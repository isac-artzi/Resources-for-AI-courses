"""Page 3: Enter a customer's details and see each model's prediction.

This makes the comparison concrete: for one made-up customer, do the three
models agree? Where they disagree is exactly where model choice matters.
"""

import pandas as pd
import streamlit as st

from src.model import FEATURES
from src.ui_data import get_trained_models, load_table

st.title("3. Try a Prediction")
st.write("Adjust the sliders to describe a customer, then see what each model predicts.")

customers = load_table("customers")
fitted, _ = get_trained_models()

# Build one input row from sliders, using the data's real ranges as bounds.
inputs = {}
cols = st.columns(2)
for i, feature in enumerate(FEATURES):
    lo = float(customers[feature].min())
    hi = float(customers[feature].max())
    mid = float(customers[feature].median())
    inputs[feature] = cols[i % 2].slider(feature, lo, hi, mid)

row = pd.DataFrame([inputs])[FEATURES]

st.subheader("What each model predicts")
results = []
for name, model in fitted.items():
    prob = float(model.predict_proba(row)[0, 1])
    results.append({"model": name, "churn_probability": round(prob, 3),
                    "prediction": "churn" if prob >= 0.5 else "stay"})
st.dataframe(pd.DataFrame(results), use_container_width=True)

st.caption(
    "When the models disagree, the customer sits near a decision boundary — the "
    "hardest and most interesting cases to get right."
)
