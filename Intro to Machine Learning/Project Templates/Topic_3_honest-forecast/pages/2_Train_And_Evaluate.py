"""Page 2: Split by time, train, and honestly score the model.

This page demonstrates the single most important idea of the topic: for a time
series you split by TIME, not at random. We train on the earliest days and test
on the most recent days the model never saw.
"""

import pandas as pd
import streamlit as st

from src.features import chronological_split
from src.model import evaluate, train_linear
from src.ui_data import load_table

st.title("2. Train & Evaluate")
st.write(
    "We split the series by **time**: train on the past, test on the future. "
    "Shuffling first would leak future information into training and give a "
    "dishonestly high score."
)

# Let the user choose how much of the (most recent) data is held out.
test_frac = st.slider("Fraction held out for testing (most recent days)", 0.1, 0.4, 0.2, 0.05)

prices = load_table("prices")
train, test = chronological_split(prices, test_frac=test_frac)

cols = st.columns(2)
cols[0].metric("Training days (past)", len(train))
cols[1].metric("Testing days (future)", len(test))

# Fit on the training portion only, then score on the untouched test portion.
model = train_linear(train)
scores = evaluate(model, test)

st.subheader("Fitted model")
slope = float(model.coef_[0])
intercept = float(model.intercept_)
st.latex(rf"\text{{close}} \approx {slope:.4f}\,t + {intercept:.2f}")

st.subheader("Honest scores on the held-out future")
cols = st.columns(2)
cols[0].metric("R² (variance explained)", f"{scores['r2']:.3f}")
cols[1].metric("RMSE (typical error, $)", f"{scores['rmse']:.2f}")

st.subheader("Fitted line vs actual")
# Show the model's predictions across the whole series next to the truth.
full = pd.concat([train, test], ignore_index=True)
full["fitted"] = model.predict(full[["t"]])
chart = full.set_index("date")[["close", "fitted"]]
st.line_chart(chart)

st.info(
    "R² near 1 and a small RMSE mean the straight-line trend explains most of "
    "the movement. On real market data these numbers would be far worse -- which "
    "is itself an honest, useful finding."
)
