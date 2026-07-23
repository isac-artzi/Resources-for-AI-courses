"""Streamlit UI -- Cloud #1 (deployed on Streamlit Community Cloud).

This is a THIN client:
  * every training job, generation, and prediction is an HTTPS call to the
    FastAPI service (no model code, no torch import here),
  * the ONLY database access is a read-only anon-key query against the `runs`
    table for the 'Run History' tab (no SQL writes here).

The transfer-learning story -- fine-tuned-from-pretrained vs from-scratch -- comes
back from the API as plain numbers; the UI shows the gap as a bar chart with
pandas. No ML libraries needed.

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

st.set_page_config(page_title="Fine-It", page_icon="🔤", layout="wide")
st.title("🔤 Fine-It — Pretrain, then Fine-Tune")
st.caption("Streamlit (this UI) → FastAPI (char Transformer) → Supabase (data). Three clouds.")


@st.cache_resource
def supabase_anon():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])


def api_get(path: str, **params):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


concepts, pre_tab, ft_tab, gen_tab, predict_tab, history_tab, card_tab = st.tabs(
    ["Concepts", "Pretrain", "Fine-tune", "Generate", "Predict", "Run History", "Model Card"]
)

# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------
with concepts:
    st.header("Transfer Learning: Pretrain, then Fine-Tune")
    st.markdown(
        "A **causal character Transformer** is trained in two phases. First, "
        "**self-supervised pretraining**: predict the next character across an "
        "unlabelled corpus. The model minimises the cross-entropy"
    )
    st.latex(r"\mathcal{L}_{\text{LM}} = -\sum_t \log p_\theta(x_{t+1}\mid x_{\le t})")
    st.markdown(
        "Learning to predict the next character forces the trunk to encode each "
        "**dialect's transition structure**. Then **fine-tuning**: attach a small "
        "classification head and train on a *tiny* labelled set — once warm-started "
        "from the pretrained trunk, and once from scratch as a baseline."
    )
    st.info(
        "The task: decide which **dialect** generated a string. Every dialect has "
        "the *same* character frequencies and differs only in its character-to-"
        "character transitions, so a bag-of-characters view is useless. The "
        "pretrained model already learned those transitions, so it beats the "
        "from-scratch model on the small labelled set — that is the transfer gap "
        "the Fine-tune tab plots."
    )

# ---------------------------------------------------------------------------
# Pretrain  (self-supervised LM)
# ---------------------------------------------------------------------------
with pre_tab:
    st.header("1) Pretrain the language model")

    with st.expander("Register a synthetic corpus", expanded=True):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Name", value="demo")
        n_rows = c2.slider("Corpus size", 500, 8000, 4000, step=500)
        seq_len = c3.slider("Sequence length", 8, 48, 20, step=4)
        if st.button("Create dataset"):
            ds = api_post(
                "/datasets",
                {"name": name, "n_rows": int(n_rows), "seq_len": int(seq_len)},
            )
            st.session_state["dataset_id"] = ds["id"]
            st.success(
                f"Created dataset id={ds['id']} "
                f"({ds['n_classes']} dialects, vocab={ds['vocab_size']}, seq_len={ds['seq_len']})"
            )

    st.subheader("Run pretraining")
    dataset_id = st.number_input(
        "dataset_id", value=int(st.session_state.get("dataset_id", 1)), step=1
    )
    c1, c2, c3 = st.columns(3)
    lr = c1.number_input("Learning rate", value=0.005, format="%.4f", key="pre_lr")
    batch_size = c2.number_input("Batch size", value=64, min_value=1, step=1, key="pre_bs")
    epochs = c3.number_input("Epochs", value=8, min_value=1, step=1, key="pre_ep")

    if st.button("Run pretraining", type="primary"):
        with st.spinner("Pretraining the next-char LM on the FastAPI service..."):
            resp = api_post(
                "/pretrain",
                {
                    "dataset_id": int(dataset_id),
                    "lr": float(lr),
                    "batch_size": int(batch_size),
                    "epochs": int(epochs),
                },
            )
        st.session_state["pretrain_run_id"] = resp["run_id"]
        st.success(f"Pretrain run {resp['run_id']} complete.")
        st.metric("Held-out next-char loss", f"{resp['val_loss']:.3f}")
        st.caption("Now fine-tune on the Fine-tune tab, or sample text on the Generate tab.")

# ---------------------------------------------------------------------------
# Fine-tune  (pretrained vs scratch)
# ---------------------------------------------------------------------------
with ft_tab:
    st.header("2) Fine-tune a classifier")
    st.caption(
        "Warm-start from the pretrained trunk AND train a from-scratch baseline on "
        "the same small labelled set — then compare."
    )
    dataset_id = st.number_input(
        "dataset_id", value=int(st.session_state.get("dataset_id", 1)), step=1, key="ft_ds"
    )
    pretrain_run_id = st.number_input(
        "pretrain_run_id", value=int(st.session_state.get("pretrain_run_id", 1)), step=1
    )
    c1, c2, c3, c4 = st.columns(4)
    n_labeled = c1.number_input("Labelled examples", value=240, min_value=30, step=30)
    lr = c2.number_input("Learning rate", value=0.004, format="%.4f", key="ft_lr")
    batch_size = c3.number_input("Batch size", value=32, min_value=1, step=1, key="ft_bs")
    epochs = c4.number_input("Epochs", value=6, min_value=1, step=1, key="ft_ep")

    if st.button("Run fine-tuning", type="primary"):
        with st.spinner("Fine-tuning pretrained + scratch models..."):
            resp = api_post(
                "/finetune",
                {
                    "dataset_id": int(dataset_id),
                    "pretrain_run_id": int(pretrain_run_id),
                    "n_labeled": int(n_labeled),
                    "lr": float(lr),
                    "batch_size": int(batch_size),
                    "epochs": int(epochs),
                },
            )
        st.session_state["finetune_run_id"] = resp["run_id"]
        st.success(f"Fine-tune run {resp['run_id']} complete.")
        pre_acc = resp["pretrained"]["accuracy"] if resp["pretrained"] else 0.0
        scr_acc = resp["scratch"]["accuracy"]
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Pretrained → fine-tuned", f"{pre_acc:.3f}")
        mc2.metric("From scratch", f"{scr_acc:.3f}")
        mc3.metric("Transfer gap", f"{pre_acc - scr_acc:+.3f}")
        chart = pd.DataFrame(
            {"accuracy": [pre_acc, scr_acc]},
            index=["pretrained → fine-tuned", "from scratch"],
        )
        st.bar_chart(chart)
        st.caption(
            "Same labelled data, same epochs — the only difference is the starting "
            "weights. Pretraining is what closes the gap."
        )

# ---------------------------------------------------------------------------
# Generate  (temperature sampling from the LM)
# ---------------------------------------------------------------------------
with gen_tab:
    st.header("Generate text from the pretrained LM")
    run_id = st.number_input(
        "pretrain run_id", value=int(st.session_state.get("pretrain_run_id", 1)), step=1,
        key="gen_run",
    )
    c1, c2, c3 = st.columns(3)
    prompt = c1.text_input("Prompt (chars a–d)", value="ab")
    length = c2.slider("Characters to generate", 1, 64, 24)
    temperature = c3.slider("Temperature", 0.1, 2.0, 0.8, step=0.1)
    if st.button("Generate", type="primary"):
        with st.spinner("Sampling from the LM..."):
            resp = api_post(
                "/generate",
                {
                    "run_id": int(run_id),
                    "prompt": prompt,
                    "length": int(length),
                    "temperature": float(temperature),
                },
            )
        st.code(resp["text"], language=None)
        st.caption(
            f"Prompt `{resp['prompt']}` + {len(resp['generated'])} sampled characters. "
            "Lower temperature is more repetitive; higher is more random."
        )

# ---------------------------------------------------------------------------
# Predict  (classify a string's dialect)
# ---------------------------------------------------------------------------
with predict_tab:
    st.header("Classify a string's dialect")
    run_id = st.number_input(
        "finetune run_id", value=int(st.session_state.get("finetune_run_id", 1)), step=1,
        key="pred_run",
    )
    try:
        class_info = api_get("/classes")
        st.caption("Classes: " + ", ".join(class_info["classes"]))
    except Exception:  # noqa: BLE001
        pass

    st.subheader("Option A — type a character sequence")
    raw = st.text_input("Characters (a–d)", value="abcdabcdabcd")
    if st.button("Classify", type="primary"):
        ids = [ord(c) - ord("a") + 1 for c in raw.strip() if "a" <= c <= "d"]
        if len(ids) < 2:
            st.error("Enter at least 2 characters from a–d.")
        else:
            with st.spinner("Classifying..."):
                resp = api_post("/predict", {"run_id": int(run_id), "sequence": ids})
            st.success(
                f"Predicted: **{resp['class_name']}**  (confidence {resp['confidence']:.2%})"
            )
            st.caption(
                f"Sequence sha256 {resp['sequence_sha256'][:16]}… logged to Supabase "
                "(no raw characters stored)."
            )

    st.subheader("Option B — classify a generated sample")
    c1, c2 = st.columns(2)
    seed = c1.number_input("Sample seed", value=0, step=1)
    if c2.button("Generate & classify"):
        with st.spinner("Generating a sample and classifying..."):
            resp = api_post("/predict_sample", {"run_id": int(run_id), "seed": int(seed)})
        st.success(
            f"Predicted: **{resp['class_name']}**  (confidence {resp['confidence']:.2%})"
        )
        if resp.get("true_class"):
            st.caption(f"Ground truth for this generated sample: {resp['true_class']}")
        st.caption(
            f"Sequence sha256 {resp['sequence_sha256'][:16]}… logged (no raw characters stored)."
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
            .select(
                "id,dataset_id,run_type,lr,batch_size,epochs,"
                "accuracy,macro_f1,scratch_accuracy,val_loss,created_at"
            )
            .order("created_at", desc=True)
            .limit(50)
            .execute()
            .data
        )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No runs yet. Pretrain then fine-tune a model.")
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
