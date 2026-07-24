"""Page 4 — Revenue by Employee (business question via pandas merge).

Business question: *Which sales rep is responsible for which revenue line?*

This page chains several merges, including one where the key columns have
DIFFERENT names in the two tables (``support_rep_id`` vs ``employee_id``).
"""

import streamlit as st

from src.analysis import revenue_by_employee
from src.ui_data import load_table

st.title("💼 Revenue by Employee")
st.write("**Business question:** which sales rep is tied to which revenue?")

# Raw tables from SQLite.
employees = load_table("employees")
customers = load_table("customers")
invoices = load_table("invoices")
invoice_items = load_table("invoice_items")

st.markdown("**pandas analysis** (note the differently named merge keys):")
st.code(
    "items = items.merge(invoices, on='invoice_id')\n"
    "items = items.merge(customers, on='customer_id')\n"
    "items = items.merge(\n"
    "    employees,\n"
    "    left_on='support_rep_id',  # column name in customers\n"
    "    right_on='employee_id',    # column name in employees\n"
    ")\n"
    "result = items.groupby('employee_id')['revenue'].sum()",
    language="python",
)

result = revenue_by_employee(employees, customers, invoices, invoice_items)

st.subheader("Result")
st.dataframe(result, use_container_width=True)

chart_data = result.assign(
    employee=result["first_name"] + " " + result["last_name"]
).set_index("employee")["revenue"]
st.bar_chart(chart_data)

if not result.empty:
    top = result.iloc[0]
    st.success(
        f"**Interpretation:** {top['first_name']} {top['last_name']} supports the "
        f"customers who generated the most revenue (${top['revenue']:.2f}). "
        "Remember this reflects assigned customers, not individual sales skill."
    )
