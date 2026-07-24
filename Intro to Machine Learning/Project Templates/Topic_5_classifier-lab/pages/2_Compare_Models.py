"""Page 2: Train all three models and compare them fairly.

One scoreboard, one confusion matrix per model, and a plain recommendation of
the best performer by ROC-AUC.
"""

import pandas as pd
import streamlit as st

from src.model import evaluate, recommend, split
from src.ui_data import get_trained_models, load_table

st.title("2. Compare Models")
st.write(
    "All three models were trained on the **same** stratified split with the "
    "**same** seed and scaler. The only difference is the classifier, so the "
    "scoreboard is a fair comparison."
)

fitted, comparison = get_trained_models()

st.subheader("Scoreboard (sorted by ROC-AUC)")
st.dataframe(comparison, use_container_width=True)

best = recommend(comparison)
st.success(f"**Recommendation:** deploy **{best}** — it has the highest ROC-AUC.")

st.subheader("Confusion matrix per model")
# Rebuild the same test split to show each model's confusion matrix.
customers = load_table("customers")
_, X_test, _, y_test = split(customers)
choice = st.radio("Model", list(fitted.keys()), horizontal=True)
scores = evaluate(fitted[choice], X_test, y_test)
cm = pd.DataFrame(
    scores["confusion_matrix"],
    index=["actual: stayed", "actual: churned"],
    columns=["predicted: stayed", "predicted: churned"],
)
st.dataframe(cm, use_container_width=True)

st.info(
    "No single number tells the whole story. kNN and the SVM measure distances "
    "(so scaling matters); the tree splits on thresholds and is easy to explain. "
    "Pick the model whose strengths match what your product needs."
)
