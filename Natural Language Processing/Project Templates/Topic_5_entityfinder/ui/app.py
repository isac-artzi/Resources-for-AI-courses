"""Streamlit UI — Cloud #1, deployed on Streamlit Community Cloud.

FINISHED CODE. All five required tabs exist and are wired to the API. Until you
implement api/nlp.py the product tabs will show a clear "not implemented yet"
message instead of results — that is the template telling you where to work, not
a bug.

Three rules this file follows:

* It is a thin client. It never imports torch, transformers, or api.db. It calls
  HTTP endpoints and renders JSON.
* The one direct database read (training runs) uses the ANON key, which
  row-level security restricts to SELECT on `runs`. The service-role key never
  leaves the API tier.
* Every WRITE goes through the API. The Review Queue tab posts to /review rather
  than inserting into Postgres, because an anon INSERT policy would let anyone
  with the public key fabricate reviewer decisions in your audit trail.

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

st.set_page_config(page_title="EntityFinder", page_icon="🏷️", layout="wide")

# One colour per entity type. Chosen to stay legible with dark text on top; if
# you add types, add colours, and check the contrast rather than picking by eye.
TYPE_COLORS = {
    "PER": "#CFE8FF",
    "ORG": "#D9F2D9",
    "LOC": "#FFE7C2",
    "MISC": "#EADCF8",
}
FALLBACK_COLOR = "#E4E7EB"


def call_api(path: str, payload: dict | None = None, method: str = "post", params: dict | None = None):
    """POST/GET against the API and turn failures into readable messages."""
    try:
        fn = requests.post if method == "post" else requests.get
        resp = fn(f"{API_URL}{path}", json=payload, params=params, timeout=120)
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


def highlight(text: str, entities: list[dict]) -> str:
    """Render the document as HTML with every entity span marked inline.

    Two details in here are worth reading, because they are where naive versions
    of this function go wrong:

    * The text is HTML-escaped BEFORE the markup is added. A document containing
      "<script>" is a document, not an instruction.
    * Spans are walked in offset order and any span that starts before the
      previous one ended is skipped. Standard BIO decoding cannot produce
      overlapping entities, so if you see spans being dropped here, your decoder
      is emitting something impossible and that is the bug to chase.
    """
    spans = sorted(entities, key=lambda e: (e["start_char"], e["end_char"]))
    out: list[str] = []
    cursor = 0
    for e in spans:
        start, end = e["start_char"], e["end_char"]
        if start < cursor or end > len(text):
            continue
        out.append(html.escape(text[cursor:start]))
        color = TYPE_COLORS.get(e["entity_type"], FALLBACK_COLOR)
        label = html.escape(str(e["entity_type"]))
        conf = e.get("confidence")
        title = f"{label} · confidence {conf:.2f}" if isinstance(conf, (int, float)) else label
        out.append(
            f'<span style="background:{color};border-radius:4px;padding:1px 4px;'
            f'margin:0 1px;" title="{title}">{html.escape(text[start:end])}'
            f'<span style="font-size:0.7em;color:#444;padding-left:4px;">{label}</span>'
            "</span>"
        )
        cursor = end
    out.append(html.escape(text[cursor:]))
    return (
        '<div style="line-height:2.1;font-size:1.02em;border:1px solid #E0E5EB;'
        'border-radius:8px;padding:16px;background:#FFFFFF;">' + "".join(out) + "</div>"
    )


def legend() -> str:
    chips = "".join(
        f'<span style="background:{color};border-radius:4px;padding:2px 8px;'
        f'margin-right:8px;font-size:0.85em;">{t}</span>'
        for t, color in TYPE_COLORS.items()
    )
    return f'<div style="margin-bottom:10px;">{chips}</div>'


st.title("EntityFinder")
st.caption("Named-entity recognition as a service — Streamlit ⇄ FastAPI ⇄ Postgres")

with st.sidebar:
    st.subheader("Reviewer")
    reviewer_id = st.text_input(
        "Reviewer id",
        value="reviewer-1",
        help="Stored with every decision. This is an audit trail, so it has to be "
        "someone rather than nobody.",
    )
    st.subheader("Review threshold")
    threshold = st.slider(
        "Confidence below which a prediction needs a human",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.05,
        help="Your team's number. Defend it in the report: too high and reviewers "
        "drown, too low and the errors worth catching never surface.",
    )

tab_concepts, tab_extract, tab_compare, tab_review, tab_card = st.tabs(
    ["Concepts", "Extract Entities", "CRF vs. Transformer", "Review Queue", "Model Card"]
)

# ---------------------------------------------------------------------------
with tab_concepts:
    st.header("Concepts")
    st.info(
        "**This tab is yours to write.** Explain, in your own words and for an "
        "analyst who is not an engineer: what it means to treat entity "
        "recognition as tagging one token at a time; why an entity is a span and "
        "not a word; and what the confidence number next to each highlight does "
        "and does not tell them. Use your own examples from the Extract tab — "
        "screenshots of your own output beat a textbook diagram."
    )
    st.markdown(
        """
        Suggested structure (delete this list once you have written the tab):

        1. Sequence labelling in one picture: one tag per token, BIO, and why the
           `B-` prefix has to exist.
        2. From tags to spans, and why the span is the unit that gets scored.
        3. What the CRF sees (the features you wrote) versus what the transformer
           sees (context it learned). One sentence where that difference decides
           the answer.
        4. Why entity-level F1 is lower than the token accuracy an analyst might
           quote at you, and why the lower number is the honest one.
        5. What a confidence of 0.62 means, and what it does not mean.
        """
    )

# ---------------------------------------------------------------------------
with tab_extract:
    st.header("Extract Entities")
    text = st.text_area(
        "Document text",
        value=(
            "Ada Lovelace worked with Charles Babbage in London, and the Analytical "
            "Society published her notes on the Analytical Engine in 1843."
        ),
        height=160,
    )
    c1, c2 = st.columns([1, 3])
    with c1:
        model = st.radio("Model", ["transformer", "crf"], index=0)
    if st.button("Extract", type="primary"):
        data = call_api("/extract", {"text": text, "model": model})
        if data:
            m1, m2, m3 = st.columns(3)
            m1.metric("Entities found", data["entity_count"])
            low = sum(1 for e in data["entities"] if e["confidence"] < threshold)
            m2.metric(f"Below {threshold:.2f}", low)
            m3.metric("Latency (ms)", data.get("latency_ms") or 0)

            st.markdown(legend(), unsafe_allow_html=True)
            st.markdown(highlight(data["text"], data["entities"]), unsafe_allow_html=True)

            if data["entities"]:
                st.subheader("Spans")
                st.dataframe(
                    pd.DataFrame(data["entities"])[
                        ["text", "entity_type", "start_char", "end_char", "confidence"]
                    ],
                    use_container_width=True,
                )
                st.caption(
                    f"Logged as extraction {data.get('extraction_id')} under model "
                    f"version {data['model_version']}. The document itself was not "
                    f"stored — only its hash, {data['text_sha256'][:12]}…"
                )
            else:
                st.info(
                    "No entities returned. That is a result, not a crash — check "
                    "whether it is right before you assume the model is broken."
                )

# ---------------------------------------------------------------------------
with tab_compare:
    st.header("CRF vs. Transformer")
    st.caption(
        "The same text through both models, plus the entity-level scores each one "
        "recorded when it was trained."
    )
    ctext = st.text_area(
        "Text to compare",
        value="The Bank of England said Washington would review the London office.",
        height=110,
        key="compare_text",
    )
    if st.button("Run both models", type="primary"):
        results = {}
        for name in ("crf", "transformer"):
            data = call_api("/extract", {"text": ctext, "model": name})
            if data:
                results[name] = data
        if results:
            st.markdown(legend(), unsafe_allow_html=True)
            cols = st.columns(len(results))
            for col, (name, data) in zip(cols, results.items()):
                with col:
                    st.subheader(name)
                    st.metric("Entities", data["entity_count"])
                    st.markdown(
                        highlight(data["text"], data["entities"]), unsafe_allow_html=True
                    )
        if len(results) == 2:
            crf_spans = {
                (e["start_char"], e["end_char"], e["entity_type"])
                for e in results["crf"]["entities"]
            }
            tr_spans = {
                (e["start_char"], e["end_char"], e["entity_type"])
                for e in results["transformer"]["entities"]
            }
            union = crf_spans | tr_spans
            agree = crf_spans & tr_spans
            st.subheader("Where they disagree")
            st.write(
                f"Agreed on {len(agree)} of {len(union)} spans. "
                "Neither column is the truth — disagreement tells you where to look, "
                "not who is right. Read the ones only one model found, and decide by "
                "hand which is correct. Those cases are your error analysis."
            )
            only_crf = sorted(crf_spans - tr_spans)
            only_tr = sorted(tr_spans - crf_spans)
            d1, d2 = st.columns(2)
            d1.write("**CRF only**")
            d1.write([f"{ctext[s:e]} ({t})" for s, e, t in only_crf] or "—")
            d2.write("**Transformer only**")
            d2.write([f"{ctext[s:e]} ({t})" for s, e, t in only_tr] or "—")

    st.divider()
    st.subheader("Training runs")
    st.caption("Read directly from Postgres with the anon key — no API call.")
    if not (SUPABASE_URL and SUPABASE_ANON_KEY):
        st.info(
            "Add SUPABASE_URL and SUPABASE_ANON_KEY to `.streamlit/secrets.toml` "
            "(and to the Secrets box on Streamlit Cloud) to enable this table."
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
                df = pd.DataFrame(rows)
                keep = [
                    c
                    for c in [
                        "id",
                        "model_type",
                        "dataset",
                        "precision",
                        "recall",
                        "f1",
                        "model_version",
                        "created_at",
                    ]
                    if c in df.columns
                ]
                st.dataframe(df[keep], use_container_width=True)
                st.caption(
                    "These are ENTITY-level scores. If you populated them from a "
                    "token-level report, every comparison on this page is flattering "
                    "and wrong."
                )
            else:
                st.info("No training runs logged yet. Train something and come back.")
        except Exception as exc:
            st.error(f"Supabase read failed: {exc}")

# ---------------------------------------------------------------------------
with tab_review:
    st.header("Review Queue")
    st.caption(
        f"Predictions scoring below {threshold:.2f}, lowest confidence first. "
        "Accepting or correcting one writes a new row; the model's original "
        "prediction is kept either way."
    )
    queue = call_api(
        "/review_queue",
        method="get",
        params={"threshold": threshold, "limit": 50},
    )
    if queue is None:
        # call_api already reported why. Do not st.stop() here — that would take
        # the Model Card tab down with it.
        queue = {"count": -1, "items": []}
    if queue["count"] < 0:
        st.warning("The queue could not be loaded. The message above says why.")
    elif queue["count"] == 0:
        st.info(
            "Nothing queued. Either every prediction cleared the threshold, every "
            "one has already been reviewed, or nothing has been extracted yet — "
            "run the Extract tab and come back."
        )
    else:
        st.write(f"**{queue['count']}** queued for a human.")
        for item in queue["items"]:
            with st.container(border=True):
                head, score = st.columns([4, 1])
                head.markdown(
                    f"**{item['text']}** — predicted `{item['entity_type']}` "
                    f"at chars {item['start_char']}–{item['end_char']}"
                )
                score.metric("Confidence", f"{item['confidence']:.2f}")
                if item.get("context"):
                    st.caption(item["context"])
                st.caption(
                    f"entity {item['entity_id']} · extraction {item['extraction_id']} · "
                    f"{item.get('model') or '?'} {item.get('model_version') or ''}"
                )

                with st.form(f"review-{item['entity_id']}"):
                    decision = st.radio(
                        "Decision",
                        ["accept", "correct", "reject"],
                        horizontal=True,
                        key=f"decision-{item['entity_id']}",
                        help="accept: it is right. correct: real entity, wrong type "
                        "or boundary. reject: there is no entity here at all.",
                    )
                    f1, f2, f3 = st.columns(3)
                    corrected_type = f1.selectbox(
                        "Corrected type",
                        ["(unchanged)", "PER", "ORG", "LOC", "MISC"],
                        key=f"type-{item['entity_id']}",
                    )
                    corrected_start = f2.number_input(
                        "Corrected start",
                        min_value=0,
                        value=int(item["start_char"]),
                        key=f"start-{item['entity_id']}",
                    )
                    corrected_end = f3.number_input(
                        "Corrected end",
                        min_value=1,
                        value=int(item["end_char"]),
                        key=f"end-{item['entity_id']}",
                    )
                    note = st.text_input("Note (optional)", key=f"note-{item['entity_id']}")
                    submitted = st.form_submit_button("Submit decision", type="primary")

                if submitted:
                    payload: dict = {
                        "entity_id": item["entity_id"],
                        "decision": decision,
                        "reviewer_id": reviewer_id,
                        "note": note or None,
                    }
                    if decision == "correct":
                        if corrected_type != "(unchanged)":
                            payload["corrected_type"] = corrected_type
                        if int(corrected_start) != item["start_char"]:
                            payload["corrected_start_char"] = int(corrected_start)
                        if int(corrected_end) != item["end_char"]:
                            payload["corrected_end_char"] = int(corrected_end)
                        if len(payload) <= 4:
                            st.warning(
                                "Choose a different type or move a boundary — a "
                                "correction has to correct something."
                            )
                            payload = {}
                    if payload:
                        saved = call_api("/review", payload)
                        if saved:
                            st.success(
                                f"Saved review {saved['review_id']}. The model's "
                                f"original answer (`{saved['original_type']}` at "
                                f"{saved['original_start_char']}–"
                                f"{saved['original_end_char']}, confidence "
                                f"{saved['original_confidence']:.2f}) is preserved "
                                "next to your decision."
                            )
                            st.rerun()

# ---------------------------------------------------------------------------
with tab_card:
    st.header("Model Card")
    st.caption("Rendered from MODEL_CARD.md in the repository root.")
    try:
        with open("MODEL_CARD.md", encoding="utf-8") as fh:
            st.markdown(fh.read())
    except FileNotFoundError:
        st.warning("MODEL_CARD.md not found — fill it in and commit it.")
