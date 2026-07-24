"""Page 1 — explore the raw house data.

Before engineering anything, look at the raw columns and spot the problems that
feature engineering will fix: a category with no natural number line, and two
badly skewed size columns.
"""

import streamlit as st

from src.ui_data import load_table

st.title("Dataset")

houses = load_table("houses")

st.markdown(
    """
Each row is one house sale. The goal is to predict **`price`** from the other
columns. Two things make the raw data tricky for a simple model:

- **`neighborhood`** is a *category* with no natural order — turning it into a
  number (Ashby = 0, Brook = 1, …) invents a ranking that is not real.
- **`sqft`** and **`lot_size`** are *right-skewed*: most houses are modest but a
  few are huge, which drags a linear model around.
"""
)

st.subheader("The houses")
st.dataframe(houses.head(50), use_container_width=True)

st.subheader("Price by neighborhood")
st.markdown("The premiums are **not** in alphabetical order — that is the trap:")
by_nb = houses.groupby("neighborhood")["price"].median().sort_values(ascending=False)
st.bar_chart(by_nb)

st.subheader("Skew in the size columns")
col1, col2 = st.columns(2)
with col1:
    st.caption("sqft")
    st.bar_chart(houses["sqft"], height=220)
with col2:
    st.caption("lot_size")
    st.bar_chart(houses["lot_size"], height=220)

st.caption(
    "A long right tail like this is exactly what a log transform tames on the "
    "Feature Engineering page."
)
