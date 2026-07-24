"""Page 1: Look at the raw price series before modelling anything.

Always plot your data first. This page shows the daily closing price and its
overall trend, so you understand what a linear model is trying to capture.
"""

import streamlit as st

from src.features import prepare
from src.ui_data import load_table

st.title("1. Price History")
st.write(
    "The first rule of forecasting: **look at the data**. Below is the daily "
    "closing price. Notice the gentle upward trend a straight line can capture, "
    "plus the day-to-day noise it cannot."
)

prices = load_table("prices")
df = prepare(prices)  # parse dates + add the day index used later

st.subheader("Closing price over time")
# A line chart indexed by date reads naturally as a time series.
st.line_chart(df.set_index("date")["close"])

st.subheader("Summary statistics")
st.dataframe(df["close"].describe().to_frame().T, use_container_width=True)

st.subheader("Sample rows")
st.dataframe(df.head(10), use_container_width=True)

st.info(
    "The `t` column is a simple day index (0, 1, 2, ...). It is the single "
    "feature the linear model regresses the price on."
)
