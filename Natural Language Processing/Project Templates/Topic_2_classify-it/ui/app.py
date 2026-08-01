"""Streamlit UI — Cloud #1, deployed on Streamlit Community Cloud.

FINISHED CODE. All five required tabs exist and are wired to the API. Until you
implement api/nlp.py the product tabs will show a clear "not implemented yet"
message instead of results — that is the template telling you where to work, not
a bug.

Two rules this file follows:

* It is a thin client. It never imports torch, transformers, sklearn, or api.db.
  It calls HTTP endpoints and renders JSON. If you catch yourself wanting to load
  a model here, the logic belongs in the API — and Streamlit Community Cloud
  does not have the memory for it anyway.
* The two direct database reads (Recent Predictions) use the ANON key, which
  row-level security restricts to SELECT. The service-role key never leaves the
  API tier.

You are expected to change the CONCEPTS tab — that content is yours to write —
and to fill in the worked examples in the comparison tab. The rest you should
only need to extend.
"""
from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

API_URL = st.secrets.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "")

st.set_page_config(page_title="Classify-It", page_icon="🏷️", layout="wide")


def call_api(path: str, payload: dict | None = None, method: str = "post"):
    """POST/GET against the API and turn failures into readable messages."""
    try:
        fn = requests.post if method == "post" else requests.get
        resp = fn(f"{API_URL}{path}", json=payload, timeout=120)
    except requests.RequestException as exc:
        st.error(
            f"Could not reach the API at {API_URL}. {exc}\n\n"
            "On the Render free plan the first request after an idle period can "
            "take 30-60 seconds while the service wakes up. Try once more."
        )
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


@st.cache_data(ttl=60)
def fetch_schema():
    """The label schema, cached for a minute so every tab can ask for it."""
    try:
        resp = requests.get(f"{API_URL}/schema", timeout=60)
    except requests.RequestException:
        return None
    return resp.json() if resp.status_code == 200 else None


st.title("Classify-It")
st.caption(
    "Single-label binary text classification — Streamlit ⇄ FastAPI ⇄ Postgres. "
    "Two models, one held-out split, every answer logged."
)

tab_concepts, tab_classify, tab_compare, tab_recent, tab_card = st.tabs(
    [
        "Concepts",
        "Classify",
        "Baseline vs. Transformer",
        "Recent Predictions",
        "Model Card",
    ]
)

# ---------------------------------------------------------------------------
with tab_concepts:
    st.header("Concepts")
    st.info(
        "**This tab is yours to write.** Explain, for an operations analyst who "
        "is not an engineer: what the two models actually do with a sentence, "
        "why a model pretrained on a large corpus can beat one trained only on "
        "your few thousand labeled rows, and what the probability next to a "
        "prediction does and does not promise. Use your own outputs from the "
        "Classify tab — a case your baseline gets wrong and your transformer "
        "gets right is worth more than any diagram."
    )
    st.markdown(
        """
        Suggested structure (delete this list once you have written the tab):

        1. Bag of words versus context. Take one sentence where the same word
           carries two meanings and show what each model can and cannot see.
        2. What transfer learning transfers. The encoder arrives already knowing
           the language; fine-tuning only teaches it your two labels.
        3. Why we still ship the baseline: it trains in seconds, it is
           inspectable feature by feature, and on keyword-driven text it is hard
           to beat. "The transformer is better" is a measurement, not a given.
        4. The four metrics, in one sentence each, and why accuracy on its own
           is the one that will mislead your stakeholders.
        5. What "calibrated probability" means: of the inputs the model scores
           0.8, about 80% should really carry that label.
        """
    )

# ---------------------------------------------------------------------------
with tab_classify:
    st.header("Classify")

    schema = fetch_schema()
    if schema:
        with st.expander("What this model predicts", expanded=False):
            st.write(f"**Task:** {schema.get('task', '')}")
            st.write(f"**Labels:** {', '.join(schema.get('labels', []))}")
            st.write(f"**Positive label:** {schema.get('positive_label', '')}")
            for label, definition in (schema.get("label_definitions") or {}).items():
                st.write(f"- **{label}** — {definition}")
            counts = schema.get("class_counts") or {}
            if counts:
                total = sum(counts.values()) or 1
                st.write(
                    "**Class balance:** "
                    + ", ".join(f"{k} {v:,} ({v / total:.1%})" for k, v in counts.items())
                )
            if schema.get("dataset_source"):
                st.caption(
                    f"Corpus: {schema.get('dataset_name', '')} — "
                    f"{schema['dataset_source']}"
                )
    else:
        st.caption(
            "GET /schema is not answering yet — implement `nlp.label_schema()` "
            "and this panel will describe your labels."
        )

    model_kind = st.radio(
        "Model",
        ["transformer", "baseline"],
        horizontal=True,
        help="Switch and re-run the same text. Disagreements between the two are "
        "the raw material for your report.",
    )

    single, batch = st.tabs(["One message", "A batch"])

    with single:
        text = st.text_area(
            "Text to classify",
            value="My order still hasn't arrived and nobody has replied to my last two emails.",
            height=130,
        )
        if st.button("Classify", type="primary", key="classify_one"):
            data = call_api("/predict", {"text": text, "model_kind": model_kind})
            if data:
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted label", data["label"])
                c2.metric("Calibrated probability", f"{data['probability']:.3f}")
                c3.metric(
                    "Latency",
                    f"{data['latency_ms']:.0f} ms" if data.get("latency_ms") else "—",
                )
                st.caption(
                    f"Answered by `{data['model_version']}` ({data['model_kind']}). "
                    f"Logged as prediction #{data.get('prediction_id') or '—'}. "
                    f"Input hash `{data['text_sha256'][:16]}…` — the message itself "
                    "is not stored."
                )
                if data["probability"] < 0.6:
                    st.warning(
                        "This prediction is barely above chance. Your model card "
                        "should say what the service does with cases like this — "
                        "route to a human is a legitimate answer."
                    )

    with batch:
        st.caption(
            "One text per line, up to 64. The API classifies them in a single "
            "pass and returns them in the same order."
        )
        raw = st.text_area(
            "Texts",
            value=(
                "thanks, that fixed it\n"
                "I have been charged twice and need this refunded today\n"
                "how do I change my address?"
            ),
            height=160,
            key="batch_text",
        )
        if st.button("Classify batch", type="primary", key="classify_batch"):
            texts = [line.strip() for line in raw.splitlines() if line.strip()]
            if not texts:
                st.warning("Nothing to classify — every line was blank.")
            elif len(texts) > 64:
                st.warning(
                    f"{len(texts)} lines. The API caps a batch at 64 so the free "
                    "plan does not run out of memory. Split the file."
                )
            else:
                data = call_api(
                    "/predict_batch", {"texts": texts, "model_kind": model_kind}
                )
                if data:
                    rows = [
                        {
                            "text": t,
                            "label": p["label"],
                            "probability": round(p["probability"], 3),
                            "model_version": p["model_version"],
                        }
                        for t, p in zip(texts, data["predictions"])
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    st.caption(
                        f"{data['count']} predictions logged. The order of the "
                        "response matches the order you sent — that is a promise "
                        "the API makes and the tests check."
                    )

# ---------------------------------------------------------------------------
with tab_compare:
    st.header("Baseline vs. Transformer")
    st.caption(
        "Held-out metrics from GET /runs, plus a live head-to-head on one input."
    )

    runs_data = call_api("/runs?limit=50", method="get")
    runs = (runs_data or {}).get("runs", [])

    if not runs:
        st.info(
            "No training runs logged yet. Fit the baseline, fine-tune the "
            "transformer, and write both to the `runs` table — or run "
            "`python db/seed.py` to see the layout with placeholder rows."
        )
    else:
        latest = {}
        for r in runs:  # runs arrive newest first
            latest.setdefault(r["model_kind"], r)

        cols = st.columns(max(len(latest), 1))
        for col, kind in zip(cols, ["baseline", "transformer"]):
            run = latest.get(kind)
            with col:
                st.subheader(kind.title())
                if not run:
                    st.info(f"No {kind} run logged yet.")
                    continue
                m = run.get("metrics") or {}
                st.caption(
                    f"`{run['model_version']}` · trained on {run.get('n_train') or '—'} "
                    f"rows · evaluated on {run.get('n_eval') or '—'}"
                )
                a, b = st.columns(2)
                a.metric("Accuracy", f"{m.get('accuracy', 0):.3f}")
                b.metric("F1", f"{m.get('f1', 0):.3f}")
                c, d = st.columns(2)
                c.metric("Precision", f"{m.get('precision', 0):.3f}")
                d.metric("Recall", f"{m.get('recall', 0):.3f}")
                st.caption(
                    f"for positive label `{m.get('positive_label', '—')}`, "
                    f"support {m.get('support_positive', '—')}"
                )
                with st.expander("Hyperparameters"):
                    st.json(run.get("hyperparameters") or {})

        base_m = (latest.get("baseline") or {}).get("metrics") or {}
        tran_m = (latest.get("transformer") or {}).get("metrics") or {}
        if base_m and tran_m:
            st.subheader("Difference (transformer − baseline)")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "metric": k,
                            "baseline": round(base_m.get(k, 0), 4),
                            "transformer": round(tran_m.get(k, 0), 4),
                            "delta": round(tran_m.get(k, 0) - base_m.get(k, 0), 4),
                        }
                        for k in ("accuracy", "precision", "recall", "f1")
                    ]
                ),
                use_container_width=True,
            )
            if tran_m.get("accuracy", 0) > base_m.get("accuracy", 0) and tran_m.get(
                "f1", 0
            ) <= base_m.get("f1", 0):
                st.warning(
                    "The transformer wins on accuracy but not on F1. On an "
                    "imbalanced corpus that usually means it is better at "
                    "agreeing with the majority class, not better at the task. "
                    "Say so in your report rather than quoting the accuracy."
                )

        with st.expander("All logged training runs"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "id": r["id"],
                            "model_kind": r["model_kind"],
                            "model_version": r["model_version"],
                            "accuracy": (r.get("metrics") or {}).get("accuracy"),
                            "f1": (r.get("metrics") or {}).get("f1"),
                            "n_train": r.get("n_train"),
                            "created_at": r.get("created_at"),
                        }
                        for r in runs
                    ]
                ),
                use_container_width=True,
            )

    st.divider()
    st.subheader("Head to head on one input")
    st.caption(
        "Send the same text to both models and look at where they disagree. The "
        "assignment asks for three input types where the transformer wins, each "
        "with a worked example — this is where you find them. Negation, "
        "word-sense ambiguity, and paraphrase with no shared keywords are the "
        "usual places to look; confirm with your own data rather than assuming."
    )
    head_text = st.text_area(
        "Text",
        value="I wouldn't say I'm unhappy with the service.",
        height=100,
        key="head_to_head",
    )
    if st.button("Ask both models", key="both"):
        left, right = st.columns(2)
        for col, kind in ((left, "baseline"), (right, "transformer")):
            with col:
                st.markdown(f"**{kind.title()}**")
                data = call_api("/predict", {"text": head_text, "model_kind": kind})
                if data:
                    st.metric(data["label"], f"{data['probability']:.3f}")
                    st.caption(data["model_version"])

    st.info(
        "**Your write-up goes here.** Replace this box with the three input "
        "types you found and one worked example of each: the text, what each "
        "model answered, the true label, and one sentence on why the encoder "
        "had the advantage."
    )

# ---------------------------------------------------------------------------
with tab_recent:
    st.header("Recent Predictions")
    st.caption(
        "Read directly from the `predictions` table with the anon key — no API "
        "call. This is the serving audit trail: one row per question the live "
        "service answered."
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
                .table("predictions")
                .select("*")
                .order("created_at", desc=True)
                .limit(200)
                .execute()
                .data
            )
            if rows:
                df = pd.DataFrame(rows)
                c1, c2, c3 = st.columns(3)
                c1.metric("Predictions logged", len(df))
                if "probability" in df:
                    c2.metric("Mean probability", f"{df['probability'].mean():.3f}")
                if "latency_ms" in df and df["latency_ms"].notna().any():
                    c3.metric("Median latency", f"{df['latency_ms'].median():.0f} ms")

                if "predicted_label" in df:
                    st.subheader("Predicted label mix")
                    st.bar_chart(df["predicted_label"].value_counts())
                    st.caption(
                        "Compare this to the class balance on the Classify tab. "
                        "If the live mix is far from the training mix, the traffic "
                        "is not the population you trained on, and your held-out "
                        "metrics do not describe what is happening in production."
                    )

                st.subheader("Rows")
                st.dataframe(df, use_container_width=True)
                st.caption(
                    "No input text, by design — only its sha256. Two rows with "
                    "the same hash were the same message."
                )
            else:
                st.info(
                    "No predictions logged yet. Classify something on the "
                    "Classify tab and come back."
                )
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
