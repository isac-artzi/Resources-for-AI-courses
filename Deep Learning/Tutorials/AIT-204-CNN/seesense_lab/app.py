"""Streamlit front end (given). Run with:  streamlit run seesense_lab/app.py"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# `streamlit run` puts this file's folder on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seesense_lab.predict import Predictor, to_db_record

CKPT = Path("checkpoints/model.pt")
HISTORY = Path("checkpoints/history.json")

st.set_page_config(page_title="See-Sense lab", layout="wide")
st.title("See-Sense lab")

if not CKPT.exists():
    st.warning("No checkpoint yet. Run `python -m seesense_lab.train` first.")
    st.stop()


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor.from_checkpoint(CKPT)


predictor = load_predictor()
classify_tab, training_tab = st.tabs(["Classify an image", "Training run"])

with classify_tab:
    st.caption(f"Model run `{predictor.run_id}`. Images are resized to 32x32 before prediction.")
    upload = st.file_uploader("Upload a JPG or PNG", type=["jpg", "jpeg", "png"])
    if upload:
        try:
            with st.spinner("Classifying..."):
                result = predictor.predict(upload.getvalue())
        except ValueError as exc:
            st.error(str(exc))
        else:
            left, right = st.columns(2)
            left.image(upload, caption="Your upload", use_container_width=True)
            right.image(f"data:image/png;base64,{result['gradcam_png_b64']}",
                        caption=f"Grad-CAM for '{result['label']}'", use_container_width=True)
            st.metric("Prediction", result["label"], f"{result['confidence']:.1%} confident")
            st.bar_chart(pd.DataFrame(result["top5"]).set_index("label"))
            with st.expander("What the project would store in Supabase"):
                st.json(to_db_record(result))

with training_tab:
    if HISTORY.exists():
        hist = pd.DataFrame(json.loads(HISTORY.read_text())).set_index("epoch")
        c1, c2, c3 = st.columns(3)
        c1.subheader("Accuracy (%)")
        c1.line_chart(hist[["train_acc", "test_acc"]])
        c2.subheader("Loss")
        c2.line_chart(hist[["train_loss", "test_loss"]])
        c3.subheader("Learning rate")
        c3.line_chart(hist[["lr"]])
    else:
        st.info("No history.json found.")
