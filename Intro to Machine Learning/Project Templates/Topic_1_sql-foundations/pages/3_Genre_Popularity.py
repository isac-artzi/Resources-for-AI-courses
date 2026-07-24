"""Page 3 — Genre Popularity (business question via pandas merge).

Business question: *Which genres bring in the most revenue?*
"""

import streamlit as st

from src.analysis import genre_popularity
from src.ui_data import load_table

st.title("📖 Genre Popularity")
st.write("**Business question:** which genres bring in the most revenue?")

# Raw tables from SQLite.
invoice_items = load_table("invoice_items")
books = load_table("books")
genres = load_table("genres")

st.markdown("**pandas analysis** (joining + aggregation):")
st.code(
    "items['revenue'] = items['unit_price'] * items['quantity']\n"
    "items = items.merge(books, on='book_id')     # find each book's genre\n"
    "items = items.merge(genres, on='genre_id')   # find the genre name\n"
    "result = items.groupby('genre').agg(\n"
    "    revenue=('revenue', 'sum'), units_sold=('quantity', 'sum'))",
    language="python",
)

result = genre_popularity(invoice_items, books, genres)

st.subheader("Result")
st.dataframe(result, use_container_width=True)
st.bar_chart(result.set_index("genre")["revenue"])

if not result.empty:
    top = result.iloc[0]
    st.success(
        f"**Interpretation:** *{top['genre']}* leads on revenue "
        f"(${top['revenue']:.2f} from {int(top['units_sold'])} units). "
        "Revenue combines popularity and price -- a pricier genre can lead "
        "even with fewer units sold."
    )
