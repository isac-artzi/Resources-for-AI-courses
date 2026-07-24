"""Streamlit-facing data helpers (the only module that imports Streamlit).

Keeping the ``streamlit`` import out of the rest of ``src`` means those modules
stay easy to unit-test without launching a web app. The functions here add
Streamlit's caching so the database is built once and the pipeline (prepare,
compare, fit) runs once per session, not on every click.

The app computes the comparison and model *live* so it works even before you run
the DVC pipeline. ``dvc repro`` produces the same numbers as reproducible files.
"""

import streamlit as st

from db.build_sqlite import build
from src.db import get_connection, read_table
from src.paths import DB_PATH
from src.prepare import prepare
from src.train import compare, fit_final


@st.cache_resource
def ensure_database() -> bool:
    """Build the SQLite database if it does not exist yet."""
    if not DB_PATH.exists():
        build()
    return True


@st.cache_data
def load_table(name: str):
    """Read a single table into a DataFrame, cached for the session."""
    ensure_database()
    conn = get_connection()
    try:
        return read_table(conn, name)
    finally:
        conn.close()


@st.cache_data
def get_prepared():
    """Return the cleaned (prepared) houses DataFrame, cached."""
    return prepare(load_table("houses"))


@st.cache_data
def get_metrics():
    """Cross-validate baseline vs engineered features once, cached."""
    return compare(get_prepared())


@st.cache_resource
def get_model():
    """Fit the engineered model once and cache the (model, columns) bundle."""
    return fit_final(get_prepared())
