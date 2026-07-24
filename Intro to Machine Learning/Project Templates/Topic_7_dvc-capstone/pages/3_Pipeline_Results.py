"""Page 3 — the head-to-head comparison (the payoff).

Both feature sets are scored with the *same* 5-fold cross-validation, so the
only difference is the features. This is the number you would put in a release
note to justify the work.
"""

import pandas as pd
import streamlit as st

from src.ui_data import get_metrics

st.title("Pipeline Results")

metrics = get_metrics()

st.markdown(
    """
Baseline vs engineered features, scored with 5-fold cross-validation on the same
data. Two metrics:

- **R²** — fraction of price variation explained (higher is better, max 1.0).
- **RMSE** — typical dollar error (lower is better).
"""
)

table = pd.DataFrame(
    {
        "model": ["Baseline", "Engineered"],
        "R²": [metrics["baseline"]["r2"], metrics["engineered"]["r2"]],
        "RMSE ($)": [metrics["baseline"]["rmse"], metrics["engineered"]["rmse"]],
    }
)
st.dataframe(table, use_container_width=True, hide_index=True)

col1, col2 = st.columns(2)
col1.metric("R² gain", f"+{metrics['r2_gain']:.3f}")
col2.metric("RMSE reduced by", f"${metrics['rmse_drop']:,.0f}")

st.bar_chart(table.set_index("model")["R²"])

st.success(
    f"Feature engineering lifts R² from **{metrics['baseline']['r2']:.3f}** to "
    f"**{metrics['engineered']['r2']:.3f}** — a defensible, measured improvement, "
    "not a guess."
)

st.caption(
    "Run `dvc repro` to reproduce these exact numbers as tracked files "
    "(data/derived/metrics.json), so a teammate can verify them from the raw data."
)
