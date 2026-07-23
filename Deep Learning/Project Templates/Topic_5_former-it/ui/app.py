"""Streamlit UI -- Cloud #1 (deployed on Streamlit Community Cloud).

This is a THIN client:
  * every training job and every prediction is an HTTPS call to the FastAPI
    service (no model code, no torch import here),
  * the ONLY database access is a read-only anon-key query against the `runs`
    table for the 'Run History' tab (no SQL writes here).

The per-head self-attention matrices arrive from the API as plain nested lists;
the UI renders them as heatmaps with matplotlib -- no ML libraries needed.

Configuration comes from st.secrets (see .streamlit/secrets.toml.example):
    API_URL              -> your Render.com base URL
    SUPABASE_URL         -> https://<ref>.supabase.co
    SUPABASE_ANON_KEY    -> the public anon key (safe to ship to the browser)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from supabase import create_client

API_URL = st.secrets["API_URL"].rstrip("/")

st.set_page_config(page_title="Former-It", page_icon="🔠", layout="wide")
st.title("🔠 Former-It — A Live Transformer Classifier")
st.caption("Streamlit (this UI) → FastAPI (Transformer encoder) → Supabase (data). Three clouds.")


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


def _attention_heatmaps(resp: dict):
    tokens = resp["sequence"]
    attn = np.asarray(resp["attention"])          # [L, H, T, T]
    labels = [f"{i}:{t}" for i, t in enumerate(tokens)]
    for layer in range(resp["n_layers"]):
        st.markdown(f"**Layer {layer + 1}**")
        cols = st.columns(resp["n_heads"])
        for head in range(resp["n_heads"]):
            fig, ax = plt.subplots(figsize=(2.6, 2.6))
            ax.imshow(attn[layer, head], cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticklabels(labels, fontsize=6)
            ax.set_xlabel("attends to", fontsize=7)
            ax.set_ylabel("query pos", fontsize=7)
            ax.set_title(f"head {head + 1}", fontsize=8)
            fig.tight_layout()
            cols[head].pyplot(fig)
            plt.close(fig)


def _show_prediction(resp: dict):
    st.success(
        f"Predicted: **{resp['class_name']}**  (confidence {resp['confidence']:.2%})"
    )
    if resp.get("true_class"):
        st.caption(f"Ground truth for this generated sample: {resp['true_class']}")
    st.subheader("Per-head self-attention (rows sum to 1)")
    st.caption(
        "For a palindrome, heads can learn the **anti-diagonal**: position i "
        "attending to position n−1−i (its mirror)."
    )
    _attention_heatmaps(resp)
    st.caption(
        f"Sequence sha256 {resp['sequence_sha256'][:16]}… logged to Supabase "
        "(no raw symbols stored)."
    )


concepts, train_tab, predict_tab, history_tab, card_tab = st.tabs(
    ["Concepts", "Train", "Predict", "Run History", "Model Card"]
)

# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------
with concepts:
    st.header("Self-Attention & the Transformer Encoder")
    st.markdown(
        "A Transformer replaces recurrence with **self-attention**: every "
        "position looks at every other position in parallel. Each position "
        "projects to a query $q$, key $k$, and value $v$; the attention from "
        "position $i$ to position $j$ is a softmax over scaled dot products:"
    )
    st.latex(
        r"\alpha_{ij} = \mathrm{softmax}_j\!\left(\frac{q_i\cdot k_j}{\sqrt{d_h}}\right),"
        r"\qquad z_i = \sum_j \alpha_{ij}\, v_j"
    )
    st.markdown(
        "Multiple **heads** run this in parallel with different projections, so "
        "different heads can specialise. Positional encodings inject order, and "
        "residual + layer-norm blocks stack into an encoder. This template builds "
        "all of that from scratch (no `nn.TransformerEncoder`) so the internals "
        "stay legible."
    )
    st.info(
        "The task: decide whether a short symbol sequence is a **palindrome**. "
        "That needs comparing position i with position n−1−i — a comparison a "
        "single attention head can express, so the head's heatmap should light up "
        "along the anti-diagonal."
    )

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
with train_tab:
    st.header("Train a model")

    with st.expander("1) Register a synthetic sequence dataset", expanded=True):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Name", value="demo")
        n_rows = c2.slider("Sequences", 500, 8000, 2000, step=500)
        seq_len = c3.slider("Sequence length (even)", 4, 16, 8, step=2)
        if st.button("Create dataset"):
            ds = api_post(
                "/datasets",
                {"name": name, "n_rows": int(n_rows), "seq_len": int(seq_len)},
            )
            st.session_state["dataset_id"] = ds["id"]
            st.success(
                f"Created dataset id={ds['id']} "
                f"({ds['n_classes']} classes, vocab={ds['vocab_size']}, seq_len={ds['seq_len']})"
            )

    st.subheader("2) Train")
    dataset_id = st.number_input(
        "dataset_id", value=int(st.session_state.get("dataset_id", 1)), step=1
    )
    c1, c2, c3 = st.columns(3)
    lr = c1.number_input("Learning rate", value=0.005, format="%.4f")
    batch_size = c2.number_input("Batch size", value=64, min_value=1, step=1)
    epochs = c3.number_input("Epochs", value=20, min_value=1, step=5)

    if st.button("Run training", type="primary"):
        with st.spinner("Training the Transformer on the FastAPI service..."):
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

    st.subheader("Option A — type a symbol sequence")
    raw = st.text_input("Comma-separated symbol ids (1–5)", value="2, 4, 5, 3, 3, 5, 4, 2")
    if st.button("Classify sequence", type="primary"):
        try:
            sequence = [int(t.strip()) for t in raw.split(",") if t.strip() != ""]
        except ValueError:
            st.error("Could not parse symbol ids — use comma-separated integers.")
            sequence = None
        if sequence:
            with st.spinner("Running the Transformer..."):
                _show_prediction(
                    api_post("/predict", {"run_id": int(run_id), "sequence": sequence})
                )

    st.subheader("Option B — classify a generated sample")
    c1, c2 = st.columns(2)
    seed = c1.number_input("Sample seed", value=0, step=1)
    if c2.button("Generate & classify"):
        with st.spinner("Generating a sample and running attention..."):
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
