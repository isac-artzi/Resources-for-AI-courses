"""Page 2 — Top Customers (a business question answered with pandas).

Business question: *Who are our highest-spending customers?*

This page demonstrates the "pandas analysis" half of the build pattern. The
raw tables come from SQLite; we combine them with ``DataFrame.merge`` (the
pandas equivalent of a SQL JOIN) and then persist the result as a derived CSV.
"""

import streamlit as st

from src.analysis import top_customers_by_spend
from src.persist import save_derived
from src.ui_data import load_table

st.title("🏆 Top Customers")
st.write("**Business question:** who are our highest-spending customers?")

# --- Load the raw tables (each already came from SQLite via read_sql) ---------
customers = load_table("customers")
invoices = load_table("invoices")
invoice_items = load_table("invoice_items")

# Let the user choose how many customers to show.
top_n = st.slider("How many customers to show?", min_value=3, max_value=8, value=5)

# --- Run the analysis (all joining/aggregation happens in pandas) -------------
result = top_customers_by_spend(customers, invoices, invoice_items, top_n=top_n)

# Show the reader exactly how this was computed, labeled clearly.
st.markdown("**pandas analysis** (joining + aggregation):")
st.code(
    "items = invoice_items.copy()\n"
    "items['revenue'] = items['unit_price'] * items['quantity']\n"
    "items = items.merge(invoices, on='invoice_id')      # JOIN in pandas\n"
    "items = items.merge(customers, on='customer_id')    # JOIN in pandas\n"
    "result = (items.groupby('customer_id')['revenue']   # GROUP BY in pandas\n"
    "               .sum().sort_values(ascending=False))",
    language="python",
)

# --- Present the result: a table, a chart, and a plain-English takeaway -------
st.subheader("Result")
st.dataframe(result, use_container_width=True)

# A full customer name makes the bar chart readable.
chart_data = result.assign(
    customer=result["first_name"] + " " + result["last_name"]
).set_index("customer")["revenue"]
st.bar_chart(chart_data)

if not result.empty:
    leader = result.iloc[0]
    st.success(
        f"**Interpretation:** {leader['first_name']} {leader['last_name']} "
        f"({leader['country']}) is the top spender at ${leader['revenue']:.2f}. "
        "Use this list to prioritize loyalty outreach."
    )

# --- Persist the derived result as a versioned CSV ----------------------------
if st.button("Save this result to data/derived/"):
    path = save_derived(result, "top_customers")
    st.info(f"Saved to `{path.name}`. Other pages or later assignments can re-load it.")
