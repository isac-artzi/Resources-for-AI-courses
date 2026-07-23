"""Streamlit UI -- Cloud #1 (deployed on Streamlit Community Cloud).

This is a THIN client:
  * every training job and every prediction is an HTTPS call to the FastAPI
    service (no model code, no torch import here),
  * the ONLY database access is a read-only anon-key query against the `runs`
    table for the 'Run History' tab (no SQL writes here).

The per-timestep attention weights arrive from the API as plain numbers; the UI
just renders them as a bar chart next to the tokens -- no ML libraries needed.

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

st.set_page_config(page_title="Attend-It", page_icon="🎯", layout="wide")
st.title("🎯 Attend-It — A Live Attention Classifier")
st.caption("Streamlit (this UI) → FastAPI (LSTM + attention) → Supabase (data). Three clouds.")


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
    st.success(
        f"Predicted: **{resp['class_name']}**  (confidence {resp['confidence']:.2%})"
    )
    if resp.get("true_class"):
        st.caption(f"Ground truth for this generated sample: {resp['true_class']}")

    # Attention over each token: the peak reveals which token the model relied on.
    tokens = resp["sequence"]
    attn = resp["attention"]
    df = pd.DataFrame(
        {"token": [f"[{i}] tok={t}" for i, t in enumerate(tokens)], "attention": attn}
    ).set_index("token")
    st.subheader("Per-token attention (weights sum to 1)")
    st.bar_chart(df, use_container_width=True)
    peak = max(range(len(attn)), key=lambda i: attn[i])
    st.caption(
        f"Attention peaks on position {peak} (token {tokens[peak]}). "
        f"Sequence sha256 {resp['sequence_sha256'][:16]}… logged to Supabase "
        "(no raw tokens stored)."
    )


concepts, train_tab, predict_tab, history_tab, card_tab = st.tabs(
    ["Concepts", "Train", "Predict", "Run History", "Model Card"]
)

# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------
with concepts:
    st.header("Recurrence & Additive Attention")
    st.markdown(
        "An LSTM reads the token sequence left-to-right, producing a hidden "
        "state $h_t$ at each step. Instead of using only the last state, "
        "**additive (Bahdanau) attention** scores every hidden state and forms "
        "a weighted summary."
    )
    st.markdown("Each hidden state is scored, then softmaxed into weights that sum to 1:")
    st.latex(
        r"e_t = v^{\top}\tanh(W h_t),"
        r"\qquad \alpha_t = \frac{\exp(e_t)}{\sum_{k}\exp(e_k)}"
    )
    st.markdown("The context vector is the attention-weighted sum of hidden states:")
    st.latex(r"c = \sum_t \alpha_t\, h_t")
    st.markdown(
        "A linear classifier maps $c$ to class scores (cross-entropy loss). "
        "Because each $\\alpha_t$ says how much the model leaned on token $t$, "
        "the weights are a built-in explanation you can literally plot."
    )
    st.info(
        "In this demo every sequence's class is decided by a single marker token "
        "placed just after a randomly-positioned trigger. A model that solves the "
        "task must *attend* to that marker — so the attention bar should spike there."
    )

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
with train_tab:
    st.header("Train a model")

    with st.expander("1) Register a synthetic sequence dataset", expanded=True):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Name", value="demo")
        n_rows = c2.slider("Sequences", 200, 4000, 800, step=200)
        seq_len = c3.slider("Sequence length", 4, 32, 12, step=2)
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
    lr = c1.number_input("Learning rate", value=0.01, format="%.4f")
    batch_size = c2.number_input("Batch size", value=32, min_value=1, step=1)
    epochs = c3.number_input("Epochs", value=12, min_value=1, step=2)

    if st.button("Run training", type="primary"):
        with st.spinner("Training the attention model on the FastAPI service..."):
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

    st.subheader("Option A — type a token sequence")
    raw = st.text_input(
        "Comma-separated token ids (0–23)", value="7, 1, 3, 9, 12, 5, 8, 20, 6, 11, 4, 15"
    )
    if st.button("Classify sequence", type="primary"):
        try:
            sequence = [int(t.strip()) for t in raw.split(",") if t.strip() != ""]
        except ValueError:
            st.error("Could not parse token ids — use comma-separated integers.")
            sequence = None
        if sequence:
            with st.spinner("Running the attention classifier..."):
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
