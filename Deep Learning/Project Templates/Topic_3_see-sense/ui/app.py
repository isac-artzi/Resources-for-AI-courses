"""Streamlit UI -- Cloud #1 (deployed on Streamlit Community Cloud).

This is a THIN client:
  * every training job and every prediction is an HTTPS call to the FastAPI
    service (no model code, no torch import here),
  * the ONLY database access is a read-only anon-key query against the `runs`
    table for the 'Run History' tab (no SQL writes here).

Grad-CAM images arrive from the API already rendered as base64 PNGs; the UI just
decodes and displays them -- no image processing libraries needed here.

Configuration comes from st.secrets (see .streamlit/secrets.toml.example):
    API_URL              -> your Render.com base URL
    SUPABASE_URL         -> https://<ref>.supabase.co
    SUPABASE_ANON_KEY    -> the public anon key (safe to ship to the browser)
"""
from __future__ import annotations

import base64

import pandas as pd
import requests
import streamlit as st
from supabase import create_client

API_URL = st.secrets["API_URL"].rstrip("/")

st.set_page_config(page_title="See-Sense", page_icon="🔍", layout="wide")
st.title("🔍 See-Sense — A Live CNN Classifier with Grad-CAM")
st.caption("Streamlit (this UI) → FastAPI (CNN + Grad-CAM) → Supabase (data). Three clouds.")


@st.cache_resource
def supabase_anon():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])


def api_get(path: str, **params):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=180)
    r.raise_for_status()
    return r.json()


def _show_prediction(resp: dict):
    st.success(f"Predicted: **{resp['class_name']}**  (confidence {resp['confidence']:.2%})")
    if resp.get("true_class"):
        st.caption(f"Ground truth for this generated sample: {resp['true_class']}")
    ic, cc = st.columns(2)
    ic.image(base64.b64decode(resp["input_png"]), caption="Input", width=196)
    cc.image(base64.b64decode(resp["cam_png"]), caption="Grad-CAM (red = important)", width=196)
    st.caption(f"Image sha256 {resp['image_sha256'][:16]}… logged to Supabase (no pixels stored).")


concepts, train_tab, predict_tab, history_tab, card_tab = st.tabs(
    ["Concepts", "Train", "Predict", "Run History", "Model Card"]
)

# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------
with concepts:
    st.header("Convolutions & Grad-CAM")
    st.markdown(
        "A convolutional layer slides a small learnable kernel $K$ over the "
        "image, computing a feature map $A$:"
    )
    st.latex(r"A_{ij} = \sum_{m}\sum_{n} K_{mn}\, X_{\,i+m,\;j+n}")
    st.markdown(
        "Stacked conv + ReLU + pooling layers build up increasingly abstract "
        "features; a final linear layer maps them to class scores, trained with "
        "cross-entropy."
    )
    st.markdown(
        "**Grad-CAM** explains a prediction by weighting each feature map $A^k$ "
        "by how much it raises the score $y^c$ of the predicted class, then "
        "keeping only positive evidence:"
    )
    st.latex(
        r"\alpha_k^c = \frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A^k_{ij}},"
        r"\qquad L^c = \mathrm{ReLU}\!\left(\sum_k \alpha_k^c A^k\right)"
    )
    st.markdown(
        "The heatmap $L^c$ is upsampled and overlaid on the image so a "
        "non-technical viewer can literally see *where the network looked*."
    )

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
with train_tab:
    st.header("Train a model")

    with st.expander("1) Register a synthetic image dataset", expanded=True):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Name", value="demo")
        n_rows = c2.slider("Images", 200, 4000, 800, step=200)
        noise = c3.number_input("Pixel noise", value=0.1, min_value=0.0, max_value=1.0)
        if st.button("Create dataset"):
            ds = api_post("/datasets", {"name": name, "n_rows": int(n_rows), "noise": float(noise)})
            st.session_state["dataset_id"] = ds["id"]
            st.success(f"Created dataset id={ds['id']} ({ds['n_classes']} classes, {ds['img_size']}px)")

    st.subheader("2) Train")
    dataset_id = st.number_input(
        "dataset_id", value=int(st.session_state.get("dataset_id", 1)), step=1
    )
    c1, c2, c3 = st.columns(3)
    lr = c1.number_input("Learning rate", value=0.01, format="%.4f")
    batch_size = c2.number_input("Batch size", value=32, min_value=1, step=1)
    epochs = c3.number_input("Epochs", value=15, min_value=1, step=5)

    if st.button("Run training", type="primary"):
        with st.spinner("Training the CNN on the FastAPI service..."):
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
        mc1, mc2 = st.columns(2)
        mc1.metric("Accuracy", f"{m['accuracy']:.3f}")
        mc2.metric("Macro F1", f"{m['macro_f1']:.3f}")

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
with predict_tab:
    st.header("Predict")
    run_id = st.number_input(
        "run_id", value=int(st.session_state.get("last_run_id", 1)), step=1
    )
    try:
        class_info = api_get("/classes")
        st.caption("Classes: " + ", ".join(class_info["classes"]))
    except Exception:  # noqa: BLE001
        pass

    st.subheader("Option A — upload an image")
    up = st.file_uploader("PNG / JPG", type=["png", "jpg", "jpeg"])
    if up is not None and st.button("Classify upload", type="primary"):
        with st.spinner("Running the CNN + Grad-CAM..."):
            r = requests.post(
                f"{API_URL}/predict",
                data={"run_id": int(run_id)},
                files={"file": (up.name, up.getvalue(), up.type)},
                timeout=180,
            )
            r.raise_for_status()
            _show_prediction(r.json())

    st.subheader("Option B — classify a generated sample")
    c1, c2 = st.columns(2)
    seed = c1.number_input("Sample seed", value=0, step=1)
    if c2.button("Generate & classify"):
        with st.spinner("Generating a sample and running Grad-CAM..."):
            _show_prediction(
                api_post("/predict_sample", {"run_id": int(run_id), "seed": int(seed)})
            )

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
            .select("id,dataset_id,lr,batch_size,epochs,accuracy,macro_f1,created_at")
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
# Model Card
# ---------------------------------------------------------------------------
with card_tab:
    st.header("Model Card")
    try:
        with open("MODEL_CARD.md", "r", encoding="utf-8") as fh:
            st.markdown(fh.read())
    except FileNotFoundError:
        st.warning("MODEL_CARD.md not found.")
