"""Page 4 — mine "buy X -> buy Y" rules from shopping baskets.

The Apriori algorithm finds product combinations bought together often, then
turns them into rules scored by support, confidence, and lift. Two sliders let
you tighten or loosen the thresholds and watch the rules change.
"""

import streamlit as st

from src.rules import readable_rules
from src.ui_data import get_rules

st.title("Association Rules")

st.markdown(
    """
Which products tend to be bought **together**? The **Apriori** algorithm answers
this in two steps: find product sets that appear in enough baskets, then turn
them into `if bought → then bought` rules. Each rule has three scores:

- **support** — how common the whole combination is.
- **confidence** — of baskets with the *if* part, how many also have the *then*.
- **lift** — how much more likely the *then* is than by pure chance. **Lift > 1**
  means a genuine positive association; lift ≈ 1 means "just coincidence".
"""
)

col1, col2 = st.columns(2)
with col1:
    min_support = st.slider("Minimum support", 0.05, 0.5, 0.20, step=0.05)
with col2:
    min_confidence = st.slider("Minimum confidence", 0.3, 0.9, 0.60, step=0.05)

rules = get_rules(min_support, min_confidence)
table = readable_rules(rules)

st.subheader("Rules found")
if table.empty:
    st.warning("No rules clear these thresholds — try lowering support or confidence.")
else:
    st.dataframe(table, use_container_width=True, hide_index=True)
    top = table.iloc[0]
    st.success(
        f"Strongest association: shoppers who buy **{top['if_bought']}** also buy "
        f"**{top['then_bought']}** — lift {top['lift']} "
        f"(confidence {top['confidence']:.0%})."
    )

st.caption(
    "Sort your attention by lift, not confidence: a very common product can look "
    "confident simply because almost everyone buys it, even with no real link."
)
