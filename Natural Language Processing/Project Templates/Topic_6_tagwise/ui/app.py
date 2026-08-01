"""Streamlit UI — Cloud #1, deployed on Streamlit Community Cloud.

FINISHED CODE. All five required tabs exist and are wired up. Until you implement
api/nlp.py the product tabs will show a clear "not implemented yet" message
instead of results — that is the template telling you where to work, not a bug.

Two rules this file follows:

* It is a thin client. It never imports torch, transformers, or api.db. It calls
  HTTP endpoints and renders JSON.
* The two direct database reads (History, and the confusion matrix fallback) use
  the ANON key, which row-level security restricts to SELECT. The service-role
  key never leaves the API tier.

You are expected to change the CONCEPTS tab — that content is yours to write.
The rest you should only need to extend.
"""
from __future__ import annotations

import html

import pandas as pd
import requests
import streamlit as st

API_URL = st.secrets.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "")

st.set_page_config(page_title="TagWise", page_icon="🏷️", layout="wide")

# ---------------------------------------------------------------------------
# One colour per universal tag. Related classes share a family on purpose:
# nouns blue, verbs red, modifiers green, function words grey. When the tab is
# rendered the reader should be able to see the shape of the sentence without
# reading a single tag name, and a NOUN/VERB confusion should be visible as a
# blue word in a red position.
# ---------------------------------------------------------------------------
TAG_COLOURS = {
    "NOUN": "#CFE3F7", "PROPN": "#A9CCEF", "PRON": "#E3EDF6",
    "VERB": "#F8D0CE", "AUX": "#F6E0DF",
    "ADJ": "#D6EFD8", "ADV": "#E6F5E0",
    "ADP": "#EDE7F6", "DET": "#EFEFEF", "CCONJ": "#E8E8E8",
    "SCONJ": "#E0E0E0", "PART": "#F2F2F2",
    "NUM": "#FDF0C8", "SYM": "#FBE7D0", "PUNCT": "#F5F5F5",
    "INTJ": "#FCE3F0", "X": "#DDDDDD",
}
DEFAULT_COLOUR = "#DDDDDD"


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


def tag_chip(token: str, tag: str, used_fallback: bool) -> str:
    """One coloured token/tag chip as inline HTML."""
    colour = TAG_COLOURS.get(tag, DEFAULT_COLOUR)
    border = "2px dashed #B23A2E" if used_fallback else "1px solid rgba(0,0,0,0.12)"
    return (
        f'<span style="display:inline-block;margin:3px 4px;padding:4px 8px;'
        f'border-radius:6px;background:{colour};border:{border};'
        f'font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;">'
        f'<span style="font-size:1.02em;">{html.escape(token)}</span>'
        f'<span style="font-size:0.72em;color:#333;letter-spacing:0.04em;'
        f'margin-left:6px;">{html.escape(tag)}</span></span>'
    )


st.title("TagWise")
st.caption(
    "Part-of-speech tagging as a service — Streamlit ⇄ FastAPI ⇄ Postgres. "
    "A most-frequent-tag lookup baseline and a fine-tuned transformer, on the same splits."
)

tab_concepts, tab_tag, tab_compare, tab_history, tab_card = st.tabs(
    ["Concepts", "Tag a Sentence", "Baseline vs. Transformer", "History", "Model Card"]
)

# ---------------------------------------------------------------------------
with tab_concepts:
    st.header("Concepts")
    st.info(
        "**This tab is yours to write.** Explain, in your own words and for an "
        "analyst who is not an engineer: what a part-of-speech tag is and what "
        "downstream features depend on it; why the same word takes different tags "
        "in different sentences; how a lookup table that never looks at context "
        "still gets most words right; and what a model has to do to get the rest. "
        "Use your own examples from the tab next door — a screenshot of your own "
        "two taggers disagreeing beats a textbook diagram."
    )
    st.markdown(
        """
        Suggested structure (delete this list once you have written the tab):

        1. What the tags are, and why this service uses 17 of them rather than 45.
        2. Ambiguity: one sentence pair where a word changes class, tagged both ways.
        3. Why the lookup baseline is strong — and precisely what it cannot do.
        4. What unknown words are, what your fallback rules do about them, and
           where those rules break.
        5. What contextual embeddings add, in terms of the confusion matrix cells
           that changed.
        """
    )

# ---------------------------------------------------------------------------
with tab_tag:
    st.header("Tag a Sentence")
    sentence = st.text_area(
        "Sentence",
        value="They book the flight early, then read the book on the plane.",
        height=110,
    )
    c1, c2 = st.columns([1, 3])
    with c1:
        model = st.radio("Tagger", ["baseline", "transformer"], index=0)
    with c2:
        st.caption(
            "The baseline is a most-frequent-tag lookup with rule-based handling "
            "for words it has never seen. Tokens it had to guess are drawn with a "
            "dashed red border — that is where its errors live. Tag the same "
            "sentence with both taggers and compare the two rows."
        )

    if st.button("Tag it", type="primary"):
        data = call_api("/tag", {"sentence": sentence, "model": model})
        if data:
            m1, m2, m3 = st.columns(3)
            m1.metric("Tokens", len(data["tokens"]))
            m2.metric("Unknown words", data["unknown_count"])
            m3.metric("Model version", data["model_version"])

            st.markdown(
                "".join(
                    tag_chip(t["token"], t["tag"], t.get("used_fallback", False))
                    for t in data["tokens"]
                ),
                unsafe_allow_html=True,
            )

            rows = [
                {
                    "token": t["token"],
                    "tag": t["tag"],
                    "confidence": t.get("confidence"),
                    "unknown word": t.get("used_fallback", False),
                }
                for t in data["tokens"]
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.caption(
                "The lookup baseline reports no confidence, on purpose: it has no "
                "probability to report, and a column of 1.0s would say otherwise."
            )

# ---------------------------------------------------------------------------
with tab_compare:
    st.header("Baseline vs. Transformer")
    st.caption(
        "Read from the runs table through GET /runs. These are build-time numbers "
        "on the held-out split, not live traffic — the History tab is live traffic."
    )
    data = call_api("/runs", method="get")
    runs = (data or {}).get("runs", [])

    if not runs:
        st.info(
            "No runs logged yet. Build both taggers, call `api.db.insert_run` from "
            "your build script with accuracy, macro-F1 and the confusion matrix, "
            "then reload. `python db/seed.py` puts demo rows here in the meantime."
        )
    else:
        latest = {}
        for r in runs:  # runs arrive newest first
            latest.setdefault(r["model"], r)

        summary = pd.DataFrame(
            [
                {
                    "model": r["model"],
                    "tagset": r.get("tagset"),
                    "accuracy": r.get("accuracy"),
                    "macro F1": r.get("macro_f1"),
                    "model version": r.get("model_version"),
                    "built": r.get("created_at"),
                }
                for r in latest.values()
            ]
        )
        st.dataframe(summary, use_container_width=True)
        st.caption(
            "Accuracy and macro-F1 sit next to each other because they disagree. "
            "Macro-F1 weights a tag with twenty instances the same as one with "
            "twenty thousand, so a gap between the two models in macro-F1 is often "
            "two rare tags moving, not a general difference in quality. Check the "
            "per-tag numbers before you write a sentence about it."
        )

        if len(latest) == 2:
            a, b = latest.get("baseline"), latest.get("transformer")
            if a and b and a.get("accuracy") is not None and b.get("accuracy") is not None:
                if b["accuracy"] < a["accuracy"]:
                    st.warning(
                        "Your transformer is scoring below the lookup baseline. "
                        "That is a real result and it belongs in your report, with "
                        "your diagnosis. The usual causes, in order of how often "
                        "they are the cause: labels misaligned to subword pieces, "
                        "too few epochs, a learning rate off by an order of "
                        "magnitude, or the two models evaluated on different splits."
                    )

        st.subheader("Confusion matrix")
        choice = st.selectbox(
            "Which build", list(latest.keys()), format_func=lambda m: f"{m} tagger"
        )
        confusion = (latest[choice].get("metrics") or {}).get("confusion")
        if not confusion:
            st.info(
                "This run has no confusion matrix stored. Put it in the run's "
                '`metrics` column as {"confusion": {"labels": [...], "matrix": [[...]]}}.'
            )
        else:
            labels = confusion["labels"]
            frame = pd.DataFrame(confusion["matrix"], index=labels, columns=labels)
            st.dataframe(frame, use_container_width=True)
            st.caption(
                "Rows are the gold tag, columns are what the model predicted, so "
                "the diagonal is correct and everything else is an error you can "
                "name. Seventeen tags fit on a screen and can be read; a 45-tag "
                "version of this figure has 2,025 cells and tells a reader nothing."
            )
            off_diagonal = [
                (labels[i], labels[j], int(frame.iloc[i, j]))
                for i in range(len(labels))
                for j in range(len(labels))
                if i != j and frame.iloc[i, j]
            ]
            off_diagonal.sort(key=lambda t: -t[2])
            if off_diagonal:
                st.markdown("**Largest confusions**")
                st.dataframe(
                    pd.DataFrame(
                        off_diagonal[:10], columns=["gold", "predicted", "tokens"]
                    ),
                    use_container_width=True,
                )
                st.caption(
                    "Each of these needs a worked example in your report: the "
                    "sentence, the gold tag, what each model said, and why."
                )

# ---------------------------------------------------------------------------
with tab_history:
    st.header("History")
    st.caption(
        "Every served request, read directly from Postgres with the anon key — no "
        "API call. This is the `taggings` table, not `runs`: builds are not "
        "requests, and a page of build rows would tell a user nothing about what "
        "the service has actually done."
    )
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
                .table("taggings")
                .select("*")
                .order("created_at", desc=True)
                .limit(100)
                .execute()
                .data
            )
            if rows:
                frame = pd.DataFrame(rows)
                if "tag_sequence" in frame:
                    frame["tag_sequence"] = frame["tag_sequence"].apply(
                        lambda tags: " ".join(tags) if isinstance(tags, list) else tags
                    )
                st.dataframe(frame, use_container_width=True)
                st.caption(
                    "The sentence itself is not here and never will be — only its "
                    "sha256. Two rows with the same hash had the same input."
                )
            else:
                st.info("No taggings logged yet. Tag a sentence and come back.")
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
