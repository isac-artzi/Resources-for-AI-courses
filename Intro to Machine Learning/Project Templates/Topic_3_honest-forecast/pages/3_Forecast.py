"""Page 3: Project the trend forward -- with an honest uncertainty band.

A single predicted line pretends we know the future exactly. We do not. This
page draws a band of +/- two residual standard deviations around the forecast
and lets you save the result as a dated derived CSV.
"""

import pandas as pd
import streamlit as st

from src.features import prepare
from src.model import forecast, residual_std, train_linear
from src.persist import save_derived
from src.ui_data import load_table

st.title("3. Forecast")
st.write(
    "We fit on **all** the history, then project the trend forward. The shaded "
    "band is ± 2 standard deviations of the model's past errors -- a rough 95% "
    "interval that keeps the forecast honest about its uncertainty."
)

horizon = st.slider("Forecast horizon (business days ahead)", 5, 60, 20, 5)

prices = load_table("prices")
history = prepare(prices)

# Fit on the full history for the final forward-looking model.
model = train_linear(history)
resid = residual_std(model, history)
future = forecast(model, history, horizon=horizon, resid_std=resid)

st.subheader("Forecast with uncertainty band")
# Combine recent history with the forecast so the transition is visible.
recent = history[["date", "close"]].tail(60).rename(columns={"close": "actual"})
plot = pd.concat(
    [
        recent.set_index("date"),
        future.set_index("date")[["predicted", "lower", "upper"]],
    ]
)
st.line_chart(plot)

st.subheader("Forecast table")
st.dataframe(future, use_container_width=True)
st.caption(f"Band width is ± {2 * resid:.2f} (two residual standard deviations).")

st.subheader("Save the forecast")
if st.button("Save forecast to a dated CSV"):
    path = save_derived(future, "forecast")
    st.success(f"Saved {path.name} ({len(future)} rows).")

st.warning(
    "Honesty check: this straight-line model assumes the past trend simply "
    "continues. Real prices are driven by events it cannot see, so treat the "
    "band -- not the center line -- as the real message."
)
