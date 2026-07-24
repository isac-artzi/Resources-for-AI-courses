"""Streamlit-facing data helpers (the only module that imports Streamlit).

Keeping the ``streamlit`` import out of the rest of ``src`` means those modules
stay easy to unit-test without launching a web app. The functions here add
Streamlit's caching so the database is built once and the heavier computations
(clustering, PCA, rule mining) run once per session, not on every click.
"""

import streamlit as st

from db.build_sqlite import build
from src.cluster import cluster_scores
from src.db import get_connection, read_table
from src.paths import DB_PATH
from src.rules import mine_rules


@st.cache_resource
def ensure_database() -> bool:
    """Build the SQLite database if it does not exist yet.

    The ``.sqlite`` file is a generated artifact that we do not commit, so on a
    fresh deploy (or clone) it will be missing. We build it on demand.
    ``st.cache_resource`` guarantees this runs at most once per app process.
    """
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
def get_cluster_scores():
    """Score a range of ``k`` values once and cache the elbow/silhouette table."""
    return cluster_scores(load_table("shoppers"))


@st.cache_data
def get_rules(min_support: float, min_confidence: float):
    """Mine association rules once per (support, confidence) choice, cached."""
    return mine_rules(
        load_table("shoppers"),
        min_support=min_support,
        min_threshold=min_confidence,
    )
