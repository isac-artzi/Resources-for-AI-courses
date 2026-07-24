"""Page 2 — see the baseline vs engineered feature tables side by side.

This page makes the abstract idea of "feature engineering" concrete: it shows
the actual columns each approach feeds to the model.
"""

import streamlit as st

from src.featurize import baseline_features, engineered_features
from src.ui_data import get_prepared

st.title("Feature Engineering")

prepared = get_prepared()
baseline = baseline_features(prepared)
engineered = engineered_features(prepared)

st.markdown(
    """
Same houses, two ways of describing them to the model.

**Baseline** — the naive version: raw numbers, and `neighborhood` squashed into
a single arbitrary code column.
"""
)
st.dataframe(baseline.head(10), use_container_width=True)
st.caption(f"{baseline.shape[1]} columns.")

st.markdown(
    """
**Engineered** — four classic techniques applied:

1. **One-hot encoding** — one `nb_*` column per neighborhood.
2. **Log transform** — `log_sqft`, `log_lot_size` tame the skew.
3. **Binning** — house age grouped into `age_*` eras.
4. **Ratios** — `bed_bath_ratio`, `sqft_per_bedroom` combine columns.
"""
)
st.dataframe(engineered.head(10), use_container_width=True)
st.caption(f"{engineered.shape[1]} columns.")

st.info(
    "More columns is not automatically better — what matters is whether they "
    "help the model generalise. The Pipeline Results page measures exactly that."
)
