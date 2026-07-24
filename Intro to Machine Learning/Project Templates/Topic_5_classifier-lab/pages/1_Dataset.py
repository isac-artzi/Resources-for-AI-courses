"""Page 1: Explore the dataset before modelling.

Look at the features, the class balance, and how each feature relates to churn.
Understanding the data first makes the model results far easier to interpret.
"""

import streamlit as st

from src.model import FEATURES
from src.ui_data import load_table

st.title("1. Dataset")
st.write(
    "Each row is one customer with six numeric features and a `churned` label "
    "(1 = left, 0 = stayed). We predict `churned` from the features."
)

customers = load_table("customers")

st.subheader("Class balance")
st.bar_chart(customers["churned"].value_counts().sort_index())
st.caption("0 = stayed, 1 = churned. A moderate imbalance is normal for churn data.")

st.subheader("Sample rows")
st.dataframe(customers.head(20), use_container_width=True)

st.subheader("How each feature relates to churn")
st.write(
    "The table shows each feature's average for customers who stayed vs churned. "
    "A big gap means the feature carries useful signal."
)
means = customers.groupby("churned")[FEATURES].mean().round(2)
means.index = ["stayed (0)", "churned (1)"]
st.dataframe(means, use_container_width=True)

st.subheader("Distribution of one feature")
feature = st.selectbox("Feature", FEATURES, index=FEATURES.index("satisfaction"))
st.bar_chart(customers.groupby(feature)["churned"].mean())
st.caption("Average churn rate at each value of the chosen feature.")
