"""Page 1: Browse the labelled email corpus.

Before training a text classifier, look at the raw text and check the class
balance. This page shows a sample of messages and the spam/ham split.
"""

import streamlit as st

from src.ui_data import load_table

st.title("1. Dataset")
st.write(
    "The filter learns from labelled examples. Each row is one message with a "
    "`label` of **spam** or **ham** (legitimate mail)."
)

emails = load_table("emails")

st.subheader("Class balance")
counts = emails["label"].value_counts()
st.bar_chart(counts)
st.caption("A balanced corpus (equal spam and ham) keeps the metrics easy to read.")

st.subheader("Sample messages")
# Let the reader filter to one class to see what each looks like.
choice = st.radio("Show", ["all", "spam", "ham"], horizontal=True)
view = emails if choice == "all" else emails[emails["label"] == choice]
st.dataframe(view.head(25), use_container_width=True)

st.subheader("Typical message length (words)")
lengths = emails["text"].str.split().str.len()
st.dataframe(lengths.describe().to_frame("word_count").T, use_container_width=True)
