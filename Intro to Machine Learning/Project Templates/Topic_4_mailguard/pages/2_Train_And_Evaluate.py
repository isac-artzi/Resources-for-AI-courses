"""Page 2: Train the filter and read its honest scores.

We split the corpus (stratified, fixed seed), train naive Bayes, and report the
standard classification metrics plus a confusion matrix on the held-out test
messages.
"""

import pandas as pd
import streamlit as st

from src.model import evaluate
from src.ui_data import get_trained_filter

st.title("2. Train & Evaluate")
st.write(
    "The corpus is split into training and test sets (keeping the spam/ham "
    "balance). We train on one and score on the other — the model never sees the "
    "test messages during training."
)

pipe, X_test, y_test = get_trained_filter()
scores = evaluate(pipe, X_test, y_test)

st.subheader("Scores on the held-out test set")
cols = st.columns(5)
cols[0].metric("Accuracy", f"{scores['accuracy']:.3f}")
cols[1].metric("Precision", f"{scores['precision']:.3f}")
cols[2].metric("Recall", f"{scores['recall']:.3f}")
cols[3].metric("F1", f"{scores['f1']:.3f}")
cols[4].metric("ROC-AUC", f"{scores['roc_auc']:.3f}")

st.subheader("Confusion matrix")
# Rows = actual, columns = predicted. Labels make it readable for beginners.
cm = pd.DataFrame(
    scores["confusion_matrix"],
    index=["actual: ham", "actual: spam"],
    columns=["predicted: ham", "predicted: spam"],
)
st.dataframe(cm, use_container_width=True)

st.info(
    "**Precision vs recall matters here.** A false positive (real mail marked "
    "spam) is worse than a false negative (spam that slips through), because "
    "people miss important messages. Watch precision on the ham class closely."
)
