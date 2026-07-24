"""Home page of the ClassifierLab product.

This is the file Streamlit runs first (``streamlit run app.py``). It introduces
the product and explains the build pattern. The individual steps live in the
``pages/`` folder and appear automatically in the sidebar.

The whole app follows one data flow, the universal build pattern:

    raw CSV  ->  SQLite (basic CRUD)  ->  pandas + sklearn  ->  Streamlit UI
"""

import streamlit as st

from src.ui_data import load_table

st.set_page_config(page_title="ClassifierLab", page_icon="🔬", layout="wide")

st.title("🔬 ClassifierLab")
st.caption("An introductory classifier-comparison product template.")

st.markdown(
    """
This app trains **three classic classifiers** — k-nearest-neighbours, an RBF
support-vector machine, and a decision tree — on the *same* data with the *same*
preprocessing, then compares them fairly and recommends one.

The task: predict whether a customer **churned** (left the service).

- **Dataset** — explore the customers and how each feature relates to churn.
- **Compare Models** — one scoreboard, one confusion matrix per model, and a
  recommendation.
- **Try a Prediction** — enter a customer's details and see what each model says.

### The build pattern this project follows
"""
)

st.code(
    "data/raw/customers.csv  ->  db/build_sqlite.py  ->  SQLite (CRUD)\n"
    "SQLite table            ->  pandas.read_sql     ->  DataFrame\n"
    "DataFrame               ->  StandardScaler       ->  scaled features\n"
    "scaled features         ->  kNN / SVM / Tree     ->  three fitted models\n"
    "models                  ->  Streamlit            ->  the pages you see here",
    language="text",
)

customers = load_table("customers")

st.subheader("Dataset at a glance")
churn_rate = customers["churned"].mean()
cols = st.columns(3)
cols[0].metric("Customers", len(customers))
cols[1].metric("Features", customers.shape[1] - 1)
cols[2].metric("Churn rate", f"{churn_rate:.1%}")

st.info(
    "A fair comparison changes only ONE thing at a time. Here every model shares "
    "the same split, the same seed, and the same scaler — so any difference in "
    "the scores comes from the classifier itself."
)
