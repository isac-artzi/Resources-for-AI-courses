"""Home page of the Mailguard product.

This is the file Streamlit runs first (``streamlit run app.py``). It introduces
the product and explains the build pattern. The individual steps live in the
``pages/`` folder and appear automatically in the sidebar.

The whole app follows one data flow, the universal build pattern:

    raw CSV  ->  SQLite (basic CRUD)  ->  pandas + sklearn  ->  Streamlit UI
"""

import streamlit as st

from src.ui_data import load_table

st.set_page_config(page_title="Mailguard", page_icon="🛡️", layout="wide")

st.title("🛡️ Mailguard")
st.caption("An introductory naive-Bayes spam-filter product template.")

st.markdown(
    """
This app learns to tell **spam** from legitimate mail (**ham**) using
**naive Bayes** — the classic probabilistic text classifier. It also *explains*
each decision by showing which words tipped the balance.

- **Dataset** — browse the labelled email corpus.
- **Train & Evaluate** — fit the filter and read its accuracy, precision,
  recall, F1, ROC-AUC, and confusion matrix.
- **Classify a Message** — paste any text and see the spam probability plus the
  words that drove the score.

### The build pattern this project follows
"""
)

st.code(
    "data/raw/emails.csv  ->  db/build_sqlite.py  ->  SQLite (CRUD)\n"
    "SQLite table         ->  pandas.read_sql     ->  DataFrame\n"
    "DataFrame            ->  TfidfVectorizer      ->  word-weight vectors\n"
    "vectors              ->  MultinomialNB        ->  fitted spam filter\n"
    "filter               ->  Streamlit            ->  the pages you see here",
    language="text",
)

emails = load_table("emails")

st.subheader("Dataset at a glance")
counts = emails["label"].value_counts()
cols = st.columns(3)
cols[0].metric("Messages", len(emails))
cols[1].metric("Spam", int(counts.get("spam", 0)))
cols[2].metric("Ham (not spam)", int(counts.get("ham", 0)))

st.info(
    "Naive Bayes is 'naive' because it assumes each word is independent evidence. "
    "That assumption is technically wrong for language, yet the model works "
    "remarkably well — a useful lesson about pragmatic modelling."
)
