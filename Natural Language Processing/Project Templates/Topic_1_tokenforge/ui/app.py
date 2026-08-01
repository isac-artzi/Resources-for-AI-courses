"""Streamlit UI — Cloud #1, deployed on Streamlit Community Cloud.

FINISHED CODE. All five required tabs exist and are wired to the API. Until you
implement api/nlp.py the product tabs will show a clear "not implemented yet"
message instead of results — that is the template telling you where to work, not
a bug.

Two rules this file follows:

* It is a thin client. It never imports torch, transformers, or api.db. It calls
  HTTP endpoints and renders JSON.
* The one direct database read (History) uses the ANON key, which row-level
  security restricts to SELECT on runs. The service-role key never leaves the
  API tier.

You are expected to change the CONCEPTS tab — that content is yours to write.
The rest you should only need to extend.
"""
from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

API_URL = st.secrets.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "")

st.set_page_config(page_title="TokenForge", page_icon="🔤", layout="wide")


def call_api(path: str, payload: dict | None = None, method: str = "post"):
    """POST/GET against the API and turn failures into readable messages."""
    try:
        fn = requests.post if method == "post" else requests.get
        resp = fn(f"{API_URL}{path}", json=payload, timeout=60)
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_URL}. {exc}")
        return None
    if resp.status_code == 501:
        st.warning(
            "The API is running but this feature is not implemented yet:\n\n"
            f"> {resp.json().get('detail', '')}\n\n"
            "Implement the matching function in `api/nlp.py`."
        )
        return None
    if resp.status_code >= 400:
        st.error(f"API returned {resp.status_code}: {resp.text[:400]}")
        return None
    return resp.json()


st.title("TokenForge")
st.caption("Preprocessing and tokenization as a service — Streamlit ⇄ FastAPI ⇄ Postgres")

tab_concepts, tab_clean, tab_compare, tab_history, tab_card = st.tabs(
    ["Concepts", "Clean & Tokenize", "Compare Tokenizers", "Run History", "Model Card"]
)

# ---------------------------------------------------------------------------
with tab_concepts:
    st.header("Concepts")
    st.info(
        "**This tab is yours to write.** Explain, in your own words and for an "
        "analyst who is not an engineer: what preprocessing does and what it "
        "destroys; why subword tokenization exists; and what an out-of-vocabulary "
        "rate actually measures. Use your own examples from the Compare tab — "
        "screenshots of your own output beat a textbook diagram."
    )
    st.markdown(
        """
        Suggested structure (delete this list once you have written the tab):

        1. From characters to tokens — the three granularities and what each costs.
        2. What each classical step throws away, with a sentence that changes meaning.
        3. WordPiece vs BPE in one worked example.
        4. Why OOV rate near zero is a feature of subwords, not a bug in your test.
        """
    )

# ---------------------------------------------------------------------------
with tab_clean:
    st.header("Clean & Tokenize")
    text = st.text_area(
        "Text to process",
        value="The analysts weren't convinced — the #NLP models kept dropping emojis 🙂 and URLs like https://example.org/docs.",
        height=140,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        lowercase = st.checkbox("Lowercase", value=True)
        strip_punct = st.checkbox("Strip punctuation", value=True)
    with c2:
        rm_stop = st.checkbox("Remove stop words", value=True)
        rm_digits = st.checkbox("Remove digits", value=False)
    with c3:
        mode = st.radio("Normalization", ["lemmatize", "stem", "neither"], index=0)

    if st.button("Run preprocessing", type="primary"):
        data = call_api(
            "/preprocess",
            {
                "text": text,
                "options": {
                    "lowercase": lowercase,
                    "strip_punctuation": strip_punct,
                    "remove_stopwords": rm_stop,
                    "remove_digits": rm_digits,
                    "stem": mode == "stem",
                    "lemmatize": mode == "lemmatize",
                },
            },
        )
        if data:
            m1, m2, m3 = st.columns(3)
            m1.metric("Tokens before", data["token_count_before"])
            m2.metric("Tokens after", data["token_count_after"])
            delta = data["token_count_before"] - data["token_count_after"]
            m3.metric("Removed", delta)

            left, right = st.columns(2)
            with left:
                st.subheader("Before")
                st.write(data["original_text"])
                st.code(" · ".join(data["original_tokens"]), language=None)
            with right:
                st.subheader("After")
                st.write(data["cleaned_text"])
                st.code(" · ".join(data["cleaned_tokens"]), language=None)

            st.subheader("Steps applied, in order")
            for i, step in enumerate(data["steps_applied"], start=1):
                st.write(f"{i}. {step}")

# ---------------------------------------------------------------------------
with tab_compare:
    st.header("Compare Tokenizers")
    st.caption(
        "Two different algorithms, one text. The assignment asks for at least two."
    )
    ctext = st.text_area(
        "Text to tokenize",
        value="Tokenization of hyperparameter unhappiness in Zürich costs $3.50.",
        height=110,
        key="compare_text",
    )
    names = st.multiselect(
        "Hugging Face tokenizers",
        ["bert-base-uncased", "gpt2", "roberta-base", "xlm-roberta-base", "t5-small"],
        default=["bert-base-uncased", "gpt2"],
    )
    if st.button("Tokenize", type="primary"):
        if len(names) < 2:
            st.warning("Pick at least two tokenizers so there is something to compare.")
        else:
            data = call_api("/tokenize", {"text": ctext, "tokenizers": names})
            if data:
                cols = st.columns(len(data["results"]))
                for col, r in zip(cols, data["results"]):
                    with col:
                        st.subheader(r["tokenizer_name"])
                        st.caption(f"{r['algorithm']} · vocab {r['vocab_size']:,}")
                        st.metric("Pieces", r["token_count"])
                        st.metric("OOV rate", f"{r['oov_rate']:.3f}")
                        st.code("\n".join(r["tokens"]), language=None)
                if data.get("vocabulary_overlap"):
                    st.subheader("Vocabulary overlap (Jaccard)")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"pair": k, "overlap": v}
                                for k, v in data["vocabulary_overlap"].items()
                            ]
                        ),
                        use_container_width=True,
                    )

# ---------------------------------------------------------------------------
with tab_history:
    st.header("Run History")
    st.caption("Read directly from Postgres with the anon key — no API call.")
    if not (SUPABASE_URL and SUPABASE_ANON_KEY):
        st.info(
            "Add SUPABASE_URL and SUPABASE_ANON_KEY to `.streamlit/secrets.toml` "
            "(and to the Secrets box on Streamlit Cloud) to enable this tab."
        )
    else:
        try:
            from supabase import create_client

            rows = (
                create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
                .table("runs")
                .select("*")
                .order("created_at", desc=True)
                .limit(100)
                .execute()
                .data
            )
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("No runs logged yet. Process some text and come back.")
        except Exception as exc:
            st.error(f"Supabase read failed: {exc}")

# ---------------------------------------------------------------------------
with tab_card:
    st.header("Model Card")
    st.caption("Rendered from MODEL_CARD.md in the repository root.")
    try:
        with open("MODEL_CARD.md", encoding="utf-8") as fh:
            st.markdown(fh.read())
    except FileNotFoundError:
        st.warning("MODEL_CARD.md not found — fill it in and commit it.")
