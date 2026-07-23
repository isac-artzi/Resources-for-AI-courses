"""Streamlit UI -- Cloud #1 (deployed on Streamlit Community Cloud).

This is a THIN client:
  * every training job and every latent-space operation is an HTTPS call to the
    FastAPI service (no model code, no torch import here),
  * the ONLY database access is a read-only anon-key query against the `runs`
    table for the 'Run History' tab (no SQL writes here).

Decoded images arrive from the API already rendered as base64 PNGs; the UI just
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

st.set_page_config(page_title="Gen-It", page_icon="🎨", layout="wide")
st.title("🎨 Gen-It — A Live Variational Autoencoder")
st.caption("Streamlit (this UI) → FastAPI (VAE) → Supabase (data). Three clouds.")


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


concepts, train_tab, gen_tab, recon_tab, interp_tab, scatter_tab, history_tab, card_tab = st.tabs(
    ["Concepts", "Train", "Generate", "Reconstruct", "Interpolate", "Latent Scatter", "Run History", "Model Card"]
)

# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------
with concepts:
    st.header("Variational Autoencoders")
    st.markdown(
        "A **VAE** learns to compress each image into a small **latent vector** "
        "$z$ and reconstruct it. The encoder outputs a Gaussian posterior "
        "$q(z\\mid x)=\\mathcal{N}(\\mu, \\sigma^2)$; we sample $z$ with the "
        "**reparameterisation trick** so gradients flow:"
    )
    st.latex(r"z = \mu + \sigma \odot \varepsilon,\qquad \varepsilon \sim \mathcal{N}(0, I)")
    st.markdown(
        "Training maximises the **evidence lower bound (ELBO)** — equivalently, "
        "minimises a reconstruction term plus a KL term that pulls the posterior "
        "toward the prior $\\mathcal{N}(0, I)$:"
    )
    st.latex(
        r"\mathcal{L} = \underbrace{\mathbb{E}_{q}\big[-\log p(x\mid z)\big]}_{\text{reconstruction}}"
        r" + \underbrace{\mathrm{KL}\!\big(q(z\mid x)\,\|\,\mathcal{N}(0,I)\big)}_{\text{regulariser}}"
    )
    st.markdown(
        "Because the latent space here is **2-D**, you can literally move a point "
        "around with sliders (Generate), walk a line between two images "
        "(Interpolate), and see where each class lands (Latent Scatter)."
    )

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
with train_tab:
    st.header("Train a model")

    with st.expander("1) Register a synthetic image dataset", expanded=True):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Name", value="demo")
        n_rows = c2.slider("Images", 500, 8000, 1500, step=500)
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
    lr = c1.number_input("Learning rate", value=0.001, format="%.4f")
    batch_size = c2.number_input("Batch size", value=64, min_value=1, step=1)
    epochs = c3.number_input("Epochs", value=20, min_value=1, step=5)

    if st.button("Run training", type="primary"):
        with st.spinner("Training the VAE on the FastAPI service..."):
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
        mc1.metric("Recon loss", f"{m['recon_loss']:.2f}")
        mc2.metric("KL", f"{m['kl']:.2f}")
        mc3.metric("ELBO", f"{m['elbo']:.2f}")


def _run_id_input(key: str) -> int:
    return int(
        st.number_input(
            "run_id", value=int(st.session_state.get("last_run_id", 1)), step=1, key=key
        )
    )


# ---------------------------------------------------------------------------
# Generate  (move a point around the 2-D latent space with sliders)
# ---------------------------------------------------------------------------
with gen_tab:
    st.header("Generate — decode a latent point")
    run_id = _run_id_input("gen_run")
    st.caption("Drag the sliders to move through the 2-D latent space and decode.")
    c1, c2 = st.columns(2)
    z0 = c1.slider("z₀", -3.0, 3.0, 0.0, step=0.1)
    z1 = c2.slider("z₁", -3.0, 3.0, 0.0, step=0.1)
    if st.button("Decode point", type="primary"):
        with st.spinner("Decoding..."):
            resp = api_post("/generate", {"run_id": run_id, "z": [z0, z1]})
        st.image(base64.b64decode(resp["image_png"]), caption=f"z = {resp['z']}", width=196)
    if st.button("Sample the prior z ~ N(0, I)"):
        with st.spinner("Sampling..."):
            resp = api_post("/generate", {"run_id": run_id, "seed": 0})
        st.image(base64.b64decode(resp["image_png"]), caption=f"z = {resp['z']}", width=196)

# ---------------------------------------------------------------------------
# Reconstruct  (encode a synthetic sample, decode it, score the error)
# ---------------------------------------------------------------------------
with recon_tab:
    st.header("Reconstruct — encode then decode")
    run_id = _run_id_input("recon_run")
    c1, c2 = st.columns(2)
    seed = c1.number_input("Sample seed", value=0, step=1, key="recon_seed")
    try:
        class_info = api_get("/classes")
        options = ["(random)"] + class_info["classes"]
    except Exception:  # noqa: BLE001
        options = ["(random)"]
    choice = c2.selectbox("Force class", options, index=0)
    if st.button("Reconstruct", type="primary"):
        payload = {"run_id": run_id, "seed": int(seed)}
        if choice != "(random)":
            payload["true_class"] = options.index(choice) - 1
        with st.spinner("Encoding + decoding..."):
            resp = api_post("/reconstruct", payload)
        ic, rc = st.columns(2)
        ic.image(base64.b64decode(resp["input_png"]), caption="Input", width=196)
        rc.image(base64.b64decode(resp["recon_png"]), caption="Reconstruction", width=196)
        st.caption(
            f"Latent z = {[round(v, 3) for v in resp['z']]} · "
            f"recon MSE {resp['recon_mse']:.4f} · true class {resp['true_class']}"
        )
        st.caption(
            f"Image sha256 {resp['image_sha256'][:16]}… logged to Supabase (no pixels stored)."
        )

# ---------------------------------------------------------------------------
# Interpolate  (walk a straight line between two latents)
# ---------------------------------------------------------------------------
with interp_tab:
    st.header("Interpolate — walk between two images")
    run_id = _run_id_input("interp_run")
    c1, c2, c3 = st.columns(3)
    seed_a = c1.number_input("Seed A", value=0, step=1)
    seed_b = c2.number_input("Seed B", value=1, step=1)
    steps = c3.slider("Steps", 2, 16, 8)
    if st.button("Interpolate", type="primary"):
        with st.spinner("Decoding the latent walk..."):
            resp = api_post(
                "/interpolate",
                {"run_id": run_id, "seed_a": int(seed_a), "seed_b": int(seed_b), "steps": int(steps)},
            )
        cols = st.columns(len(resp["frames"]))
        for col, frame in zip(cols, resp["frames"]):
            col.image(base64.b64decode(frame), width=80)
        st.caption(
            f"z_a = {[round(v, 2) for v in resp['z_a']]} → z_b = {[round(v, 2) for v in resp['z_b']]}"
        )

# ---------------------------------------------------------------------------
# Latent Scatter  (see where each class lands in the 2-D latent space)
# ---------------------------------------------------------------------------
with scatter_tab:
    st.header("Latent Scatter — the shape of the latent space")
    run_id = _run_id_input("scatter_run")
    c1, c2 = st.columns(2)
    n_points = c1.slider("Points", 50, 1000, 200, step=50)
    seed = c2.number_input("Batch seed", value=123, step=1, key="scatter_seed")
    if st.button("Encode batch", type="primary"):
        with st.spinner("Encoding..."):
            resp = api_post(
                "/latent_scatter", {"run_id": run_id, "n_points": int(n_points), "seed": int(seed)}
            )
        df = pd.DataFrame(
            {
                "z0": [p["z"][0] for p in resp["points"]],
                "z1": [p["z"][1] for p in resp["points"]],
                "class": [p["class_name"] for p in resp["points"]],
            }
        )
        st.scatter_chart(df, x="z0", y="z1", color="class")
        st.caption("Each point is one image encoded to its posterior-mean latent, coloured by its known class.")

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
            .select("id,dataset_id,lr,batch_size,epochs,recon_loss,kl,elbo,created_at")
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
