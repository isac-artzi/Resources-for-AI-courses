"""Streamlit-facing data helpers (the only module that imports Streamlit).

Keeping the ``streamlit`` import out of the rest of ``src`` means those modules
stay easy to unit-test without launching a web app. The functions here add
Streamlit's caching so the database is built once and each table is read once
per session, not on every click.
"""

import streamlit as st

from db.build_sqlite import build
from src.db import get_connection, read_table
from src.model import train_and_compare
from src.paths import DB_PATH


@st.cache_resource
def ensure_database() -> bool:
    """Build the SQLite database if it does not exist yet.

    The ``.sqlite`` file is a generated artifact that we do not commit, so on a
    fresh deploy (or a fresh clone) it will be missing. We build it on demand.
    ``st.cache_resource`` guarantees this runs at most once per app process.
    """
    if not DB_PATH.exists():
        build()
    return True


@st.cache_data
def load_table(name: str):
    """Read a single table into a DataFrame, cached for the session.

    ``st.cache_data`` remembers the returned DataFrame so repeated page views
    do not re-query SQLite unnecessarily.
    """
    ensure_database()
    conn = get_connection()
    try:
        return read_table(conn, name)
    finally:
        conn.close()


@st.cache_resource
def get_trained_models():
    """Train all three models once and cache the fitted models + scoreboard.

    Returns ``(fitted_models, comparison_df)``. ``cache_resource`` keeps the
    trained models in memory so pages do not retrain on every interaction.
    """
    customers = load_table("customers")
    return train_and_compare(customers)
