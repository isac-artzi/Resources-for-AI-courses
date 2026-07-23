"""Streamlit UI -- Cloud #1 (deployed on Streamlit Community Cloud).

This is a THIN client:
  * every training job and every prediction is an HTTPS call to the FastAPI
    service (no model code, no torch import here),
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

st.set_page_config(page_title="Regress-It", page_icon="📈", layout="wide")
st.title("📈 Regress-It — A Live Linear-Regression Service")
st.caption("Streamlit (this UI) → FastAPI (model) → Supabase (data). Three clouds.")


@st.cache_resource
def supabase_anon():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])


def api_get(path: str):
    r = requests.get(f"{API_URL}{path}", timeout=60)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


concepts, train_tab, predict_tab, history_tab, card_tab = st.tabs(
    ["Concepts", "Train", "Predict", "Run History", "Model Card"]
)

# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------
with concepts:
    st.header("Gradient Descent & the Chain Rule")
    st.markdown(
        "We fit $y = w x + b$ by minimizing the mean squared error over the "
        "training set:"
    )
    st.latex(r"\mathcal{L}(w, b) = \frac{1}{n}\sum_{i=1}^{n}\left(w x_i + b - y_i\right)^2")
    st.markdown("Gradient descent updates the parameters against the gradient:")
    st.latex(
        r"\frac{\partial \mathcal{L}}{\partial w} = "
        r"\frac{2}{n}\sum_i x_i\,(w x_i + b - y_i), \qquad "
        r"\frac{\partial \mathcal{L}}{\partial b} = "
        r"\frac{2}{n}\sum_i (w x_i + b - y_i)"
    )
    st.latex(r"w \leftarrow w - \eta\,\frac{\partial \mathcal{L}}{\partial w}, \qquad "
             r"b \leftarrow b - \eta\,\frac{\partial \mathcal{L}}{\partial b}")
    st.markdown(
        "The **chain rule** is what lets us push the error at the output back to "
        "each parameter — the same engine that powers backpropagation in deeper "
        "networks later in the course. The learning rate $\\eta$ controls the step "
        "size: too small and training crawls; too large and the loss diverges."
    )

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
with train_tab:
    st.header("Train a model")

    with st.expander("1) Create a synthetic dataset", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        name = c1.text_input("Name", value="demo")
        slope = c2.number_input("True slope", value=2.5)
        intercept = c3.number_input("True intercept", value=1.0)
        noise = c4.number_input("Noise (std)", value=2.0, min_value=0.0)
        n_points = st.slider("Number of points", 100, 5000, 500, step=100)
        if st.button("Create dataset"):
            ds = api_post(
                "/datasets",
                {
                    "name": name,
                    "slope": slope,
                    "intercept": intercept,
                    "noise": noise,
                    "n_points": n_points,
                },
            )
            st.session_state["dataset_id"] = ds["id"]
            st.success(f"Created dataset id={ds['id']}")

    st.subheader("2) Train")
    dataset_id = st.number_input(
        "dataset_id", value=int(st.session_state.get("dataset_id", 1)), step=1
    )
    c1, c2, c3 = st.columns(3)
    lr = c1.number_input("Learning rate", value=0.01, format="%.4f")
    batch_size = c2.number_input("Batch size", value=32, min_value=1, step=1)
    epochs = c3.number_input("Epochs", value=100, min_value=1, step=10)

    if st.button("Run training", type="primary"):
        with st.spinner("Training on the FastAPI service..."):
            resp = api_post(
                "/train",
                {
                    "dataset_id": int(dataset_id),
                    "lr": float(lr),
                    "batch_size": int(batch_size),
                    "epochs": int(epochs),
                },
            )
        st.session_state["last_run_id"] = resp["run_id"]
        m = resp["metrics"]
        st.success(f"Run {resp['run_id']} complete.")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("MSE", f"{m['mse']:.3f}")
        mc2.metric("MAE", f"{m['mae']:.3f}")
        mc3.metric("R²", f"{m['r2']:.3f}")

        # Fitted-line overlay against the raw dataset points.
        ds = api_get(f"/runs/{resp['run_id']}")
        w = resp["weights"]
        st.markdown(
            f"Fitted line: **y = {w['slope']:.3f}·x + {w['intercept']:.3f}**"
        )

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
with predict_tab:
    st.header("Predict")
    run_id = st.number_input(
        "run_id", value=int(st.session_state.get("last_run_id", 1)), step=1
    )
    x = st.number_input("x", value=3.0)
    if st.button("Predict"):
        resp = api_post("/predict", {"run_id": int(run_id), "x": float(x)})
        st.metric(f"ŷ for x={resp['x']}", f"{resp['yhat']:.4f}")
        st.caption("This prediction was logged to Supabase by the API.")

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
            .select("id,dataset_id,lr,batch_size,epochs,mse,mae,r2,created_at")
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
