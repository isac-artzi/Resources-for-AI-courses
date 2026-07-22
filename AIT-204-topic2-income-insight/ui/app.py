"""Streamlit UI -- Cloud #1 (deployed on Streamlit Community Cloud).

This is a THIN client:
  * every training job and every prediction is an HTTPS call to the FastAPI
    service (no model code, no torch/sklearn import here),
  * the ONLY database access is a read-only anon-key query against the `runs`
    table for the 'Run History' tab (no SQL writes here).

Configuration comes from st.secrets (see .streamlit/secrets.toml.example):
    API_URL              -> your Render.com base URL
    SUPABASE_URL         -> https://<ref>.supabase.co
    SUPABASE_ANON_KEY    -> the public anon key (safe to ship to the browser)
"""
from __future__ import annotations

import pandas as pd
import requests
import streamlit as st
from supabase import create_client

API_URL = st.secrets["API_URL"].rstrip("/")

st.set_page_config(page_title="Income-Insight", page_icon="💼", layout="wide")
st.title("💼 Income-Insight — A Live Tabular MLP Classifier")
st.caption("Streamlit (this UI) → FastAPI (MLP + sklearn) → Supabase (data). Three clouds.")


@st.cache_resource
def supabase_anon():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])


def api_get(path: str, **params):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=600)
def feature_schema():
    return api_get("/schema")


concepts, train_tab, predict_tab, history_tab, card_tab = st.tabs(
    ["Concepts", "Train", "Predict", "Run History", "Model Card"]
)

# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------
with concepts:
    st.header("A Multi-Layer Perceptron for Binary Classification")
    st.markdown(
        "We predict whether income exceeds \\$50K. One hidden layer with a "
        "ReLU non-linearity feeds a single logit; a sigmoid turns it into a "
        "probability:"
    )
    st.latex(r"h = \mathrm{ReLU}(W_1 x + b_1), \qquad z = W_2 h + b_2, \qquad p = \sigma(z)")
    st.markdown("We train by minimizing **binary cross-entropy** over the training set:")
    st.latex(
        r"\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}"
        r"\Big[y_i \log p_i + (1-y_i)\log(1-p_i)\Big]"
    )
    st.markdown(
        "**Backpropagation** applies the chain rule layer by layer to get "
        "$\\partial \\mathcal{L}/\\partial W_1$ and "
        "$\\partial \\mathcal{L}/\\partial W_2$, and Adam takes the update step. "
        "Numeric features are standardized and categoricals are one-hot encoded "
        "by an sklearn `ColumnTransformer` before they ever reach the network."
    )

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
with train_tab:
    st.header("Train a model")

    with st.expander("1) Create a synthetic dataset", expanded=True):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Name", value="demo")
        n_rows = c2.slider("Rows", 200, 10000, 2000, step=200)
        noise = c3.number_input("Label noise", value=1.0, min_value=0.0)
        if st.button("Create dataset"):
            ds = api_post("/datasets", {"name": name, "n_rows": int(n_rows), "noise": float(noise)})
            st.session_state["dataset_id"] = ds["id"]
            st.success(
                f"Created dataset id={ds['id']} "
                f"(positive rate {ds['positive_rate']:.2%})"
            )

    st.subheader("2) Train")
    dataset_id = st.number_input(
        "dataset_id", value=int(st.session_state.get("dataset_id", 1)), step=1
    )
    c1, c2, c3, c4 = st.columns(4)
    hidden_dim = c1.number_input("Hidden units", value=32, min_value=1, step=8)
    lr = c2.number_input("Learning rate", value=0.01, format="%.4f")
    batch_size = c3.number_input("Batch size", value=32, min_value=1, step=1)
    epochs = c4.number_input("Epochs", value=100, min_value=1, step=10)

    if st.button("Run training", type="primary"):
        with st.spinner("Training on the FastAPI service..."):
            resp = api_post(
                "/train",
                {
                    "dataset_id": int(dataset_id),
                    "hidden_dim": int(hidden_dim),
                    "lr": float(lr),
                    "batch_size": int(batch_size),
                    "epochs": int(epochs),
                },
            )
        st.session_state["last_run_id"] = resp["run_id"]
        m = resp["metrics"]
        st.success(f"Run {resp['run_id']} complete.")
        mc = st.columns(5)
        mc[0].metric("Accuracy", f"{m['accuracy']:.3f}")
        mc[1].metric("Precision", f"{m['precision']:.3f}")
        mc[2].metric("Recall", f"{m['recall']:.3f}")
        mc[3].metric("F1", f"{m['f1']:.3f}")
        mc[4].metric("ROC-AUC", f"{m['roc_auc']:.3f}")

# ---------------------------------------------------------------------------
# Predict  (form is built from the API's /schema contract)
# ---------------------------------------------------------------------------
with predict_tab:
    st.header("Predict")
    run_id = st.number_input(
        "run_id", value=int(st.session_state.get("last_run_id", 1)), step=1
    )
    try:
        schema = feature_schema()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load feature schema: {exc}")
        schema = None

    if schema:
        features: dict = {}
        cols = st.columns(len(schema["numeric_features"]) or 1)
        for col, feat in zip(cols, schema["numeric_features"]):
            features[feat] = col.number_input(feat, value=40, step=1)
        ccols = st.columns(len(schema["categorical_features"]) or 1)
        for col, feat in zip(ccols, schema["categorical_features"]):
            features[feat] = col.selectbox(feat, schema["categories"][feat])

        if st.button("Predict", type="primary"):
            resp = api_post("/predict", {"run_id": int(run_id), "features": features})
            st.metric("Predicted income", resp["income"])
            st.caption(f"P(>50K) = {resp['proba']:.3f} — logged to Supabase by the API.")

# ---------------------------------------------------------------------------
# Run History  (read-only anon-key query against Supabase)
# ---------------------------------------------------------------------------
with history_tab:
    st.header("Run History")
    st.caption("Read directly from Supabase with the anon key — no API call.")
    try:
        rows = (
            supabase_anon()
            .table("runs")
            .select(
                "id,dataset_id,hidden_dim,lr,batch_size,epochs,"
                "accuracy,precision,recall,f1,roc_auc,created_at"
            )
            .order("created_at", desc=True)
            .limit(50)
            .execute()
            .data
        )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No runs yet. Train a model on the Train tab.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read runs: {exc}")

# ---------------------------------------------------------------------------
# Model Card  (rendered from a markdown file in the repo)
# ---------------------------------------------------------------------------
with card_tab:
    st.header("Model Card")
    try:
        with open("MODEL_CARD.md", "r", encoding="utf-8") as fh:
            st.markdown(fh.read())
    except FileNotFoundError:
        st.warning("MODEL_CARD.md not found.")
