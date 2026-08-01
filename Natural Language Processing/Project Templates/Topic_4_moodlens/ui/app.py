"""Streamlit UI — Cloud #1, deployed on Streamlit Community Cloud.

FINISHED CODE. All six required tabs exist and are wired up. Until you implement
api/nlp.py the product tabs show a clear "not implemented yet" message instead of
results, and the metrics tabs show the seed row — that is the template telling
you where to work, not a bug.

Three rules this file follows:

* It is a thin client. It never imports torch, transformers, sklearn or api.db.
  It calls HTTP endpoints and renders JSON.
* Serving goes through the API; evaluation is read straight from the database.
  Scoring text is a POST to /predict. Drawing the confusion matrix is a SELECT
  on `runs` with the ANON key, which row-level security restricts to reads. The
  service-role key never leaves the API tier.
* It does not compute metrics. If you catch yourself calculating an F1 in this
  file, the number belongs in a training run that wrote its results down, not in
  a chart that recomputes something different every time someone opens the page.

You are expected to write the CONCEPTS tab and to fill in the failure cases that
the Bias Audit tab renders (they live in MODEL_CARD.md). The rest you should
only need to extend.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import streamlit as st

def secret(name: str, default: str = "") -> str:
    """Read a Streamlit secret, tolerating the absence of a secrets file.

    ``st.secrets`` raises when no secrets.toml exists anywhere, which would stop
    the app dead on a fresh fork. The template's promise is that the UI runs
    before you have configured anything, so a missing secret is a default, not a
    crash — the tabs that need one say what to add instead.

    Note where Streamlit looks: ``.streamlit/secrets.toml`` relative to the
    DIRECTORY YOU RUN FROM, plus your home directory. On Streamlit Community
    Cloud you paste the values into the Secrets box and no file is involved.
    """
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


API_URL = secret("API_URL", "http://127.0.0.1:8000").rstrip("/")
SUPABASE_URL = secret("SUPABASE_URL")
SUPABASE_ANON_KEY = secret("SUPABASE_ANON_KEY")

LABEL_ORDER = ["negative", "positive"]

st.set_page_config(page_title="MoodLens", page_icon="🎭", layout="wide")


def call_api(path: str, payload: dict | None = None, method: str = "post"):
    """POST/GET against the API and turn failures into readable messages."""
    try:
        fn = requests.post if method == "post" else requests.get
        resp = fn(f"{API_URL}{path}", json=payload, timeout=120)
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


@st.cache_data(ttl=60, show_spinner=False)
def latest_run() -> Optional[dict]:
    """Newest row from `runs`, read directly from Postgres with the anon key.

    Cached for a minute: every tab that draws a chart calls this, and without the
    cache a page with four charts is four round trips to Supabase.
    """
    if not (SUPABASE_URL and SUPABASE_ANON_KEY):
        return None
    try:
        from supabase import create_client

        rows = (
            create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
            .table("runs")
            .select("*")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data
        )
        return rows[0] if rows else None
    except Exception as exc:  # noqa: BLE001 — surfaced to the user below
        st.error(f"Supabase read failed: {exc}")
        return None


def require_run() -> Optional[dict]:
    """Fetch the latest run, or explain what is missing. Warns on the seed row."""
    if not (SUPABASE_URL and SUPABASE_ANON_KEY):
        st.info(
            "Add SUPABASE_URL and SUPABASE_ANON_KEY to `.streamlit/secrets.toml` "
            "(and to the Secrets box on Streamlit Cloud) to enable this tab."
        )
        return None
    run = latest_run()
    if not run:
        st.info(
            "No training runs logged yet. Apply `db/migrations/001_init.sql`, run "
            "`python db/seed.py` to see the layout, then log a real run from your "
            "training script with `api.db.insert_run(...)`."
        )
        return None
    if str(run.get("model_version", "")).startswith("seed"):
        st.error(
            "**This is the seed row. Every number below is invented fixture data.** "
            "It is here to show the shape of `runs.metrics`. Train a model, log the "
            "run, and delete the seed row before you screenshot anything."
        )
    return run


def metrics_of(run: dict) -> dict:
    """The `metrics` jsonb column, defensively."""
    m = run.get("metrics") or {}
    return m if isinstance(m, dict) else {}


def model_card_text() -> Optional[str]:
    """MODEL_CARD.md, whether the app was started from the root or from ui/."""
    here = Path(__file__).resolve().parent
    for candidate in (Path("MODEL_CARD.md"), here.parent / "MODEL_CARD.md"):
        try:
            return candidate.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            continue
    return None


def model_card_section(heading: str) -> Optional[str]:
    """Return the body of one `## heading` section of MODEL_CARD.md.

    Both the Model Card tab and the Bias Audit tab read that one file, so the
    failure cases you document cannot drift out of sync with the model card that
    is supposed to contain them.
    """
    text = model_card_text()
    if text is None:
        return None
    lines = text.splitlines()
    out: list[str] = []
    capturing = False
    for line in lines:
        if line.startswith("## "):
            if capturing:
                break
            capturing = line[3:].strip().lower() == heading.strip().lower()
            continue
        if capturing:
            out.append(line)
    body = "\n".join(out).strip()
    return body or None


st.title("MoodLens")
st.caption(
    "Document and aspect-based sentiment as a service — Streamlit ⇄ FastAPI ⇄ Postgres"
)

(
    tab_concepts,
    tab_score,
    tab_aspects,
    tab_performance,
    tab_bias,
    tab_card,
) = st.tabs(
    [
        "Concepts",
        "Score Text",
        "Aspect Breakdown",
        "Model Performance",
        "Bias Audit",
        "Model Card",
    ]
)

# ---------------------------------------------------------------------------
with tab_concepts:
    st.header("Concepts")
    st.info(
        "**This tab is yours to write.** Explain, for an analyst who is not an "
        "engineer: what a sentiment score is and is not, what aspect-based "
        "sentiment adds, and what a calibrated probability means to someone who "
        "is about to threshold on it. Use your own examples from the tabs to the "
        "right — a screenshot of your own model getting a sarcastic review wrong "
        "is worth more than a textbook diagram."
    )
    st.markdown(
        """
        Suggested structure (delete this list once you have written the tab):

        1. Sentiment is about the text, not the person. A negative review is not
           an angry customer, and the difference matters the moment anyone acts
           on the score.
        2. Why fine-tuning beats a keyword list, in one worked example — and one
           example where the keyword list is fine.
        3. What aspect-based sentiment buys you: one review, three verdicts, and
           a different business decision than the single label would produce.
        4. Sarcasm and negation: two examples from your own testing, with what
           the model said and what it should have said.
        5. Calibration: "0.9 confident" should mean "right about nine times out
           of ten". Show your reliability curve and say whether it does.
        """
    )

# ---------------------------------------------------------------------------
with tab_score:
    st.header("Score Text")
    st.caption("POST /predict — one review, document-level sentiment.")

    text = st.text_area(
        "Review text",
        value=(
            "Well, that was two hours I'll never get back. The lead actor is "
            "genuinely wonderful, though, and the score is beautiful."
        ),
        height=150,
    )
    include = st.checkbox("Include the aspect breakdown", value=True)

    if st.button("Score", type="primary"):
        data = call_api("/predict", {"text": text, "include_aspects": include})
        if data:
            sentiment = data["sentiment"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Label", sentiment["label"])
            c2.metric("P(positive)", f"{sentiment['probability_positive']:.3f}")
            c3.metric("Confidence", f"{sentiment['confidence']:.3f}")
            c4.metric("Model", sentiment["model_name"])

            if not sentiment.get("calibrated", False):
                st.warning(
                    "This probability is **not calibrated** — it is a raw score. "
                    "Do not let anyone downstream threshold on it as if 0.7 meant "
                    "'right 70% of the time'."
                )
            else:
                delta = abs(
                    sentiment["probability_positive"]
                    - sentiment["raw_probability_positive"]
                )
                st.caption(
                    f"Calibrated. Raw score {sentiment['raw_probability_positive']:.3f}, "
                    f"calibrated {sentiment['probability_positive']:.3f} "
                    f"(moved {delta:.3f})."
                )

            if data.get("aspects"):
                st.subheader("Aspects")
                st.dataframe(
                    pd.DataFrame(data["aspects"])[["aspect", "label", "score"]],
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption(
                f"Logged as prediction {data.get('prediction_id')} · "
                f"model {data['model_version']} · sha256 {data['text_sha256'][:12]}…"
            )

    st.divider()
    st.subheader("Batch")
    st.caption("POST /predict_batch — one line per review, up to 64.")
    batch_text = st.text_area(
        "One review per line",
        value=(
            "Great, another sequel that breaks everything the first one built.\n"
            "I cannot recommend this enough.\n"
            "It was fine. Not bad, not good."
        ),
        height=120,
        key="batch_text",
    )
    batch_aspects = st.checkbox(
        "Include aspects (slower)", value=False, key="batch_aspects"
    )
    if st.button("Score batch"):
        texts = [line.strip() for line in batch_text.splitlines() if line.strip()]
        if not texts:
            st.warning("Nothing to score.")
        elif len(texts) > 64:
            st.warning(f"{len(texts)} lines — the API accepts at most 64 per call.")
        else:
            data = call_api(
                "/predict_batch", {"texts": texts, "include_aspects": batch_aspects}
            )
            if data:
                rows = [
                    {
                        "text": t[:80] + ("…" if len(t) > 80 else ""),
                        "label": r["sentiment"]["label"],
                        "p(positive)": round(r["sentiment"]["probability_positive"], 3),
                        "confidence": round(r["sentiment"]["confidence"], 3),
                    }
                    for t, r in zip(texts, data["results"])
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                if data.get("label_counts"):
                    st.bar_chart(pd.Series(data["label_counts"], name="reviews"))

# ---------------------------------------------------------------------------
with tab_aspects:
    st.header("Aspect Breakdown")
    st.caption(
        "One review, one verdict per aspect, each with the words it was based on. "
        "The aspects are a decision you made and documented — they are not a "
        "property of the data."
    )

    atext = st.text_area(
        "Review text",
        value=(
            "The performances are superb and the cinematography is gorgeous, but "
            "the script is a mess and the ending makes no sense at all."
        ),
        height=150,
        key="aspect_text",
    )
    if st.button("Break it down", type="primary", key="aspect_button"):
        data = call_api("/predict", {"text": atext, "include_aspects": True})
        if data:
            overall = data["sentiment"]["label"]
            st.metric("Overall", overall)

            aspects = data.get("aspects", [])
            if not aspects:
                st.info("The API returned no aspects for this text.")
            else:
                cols = st.columns(len(aspects))
                for col, a in zip(cols, aspects):
                    with col:
                        st.subheader(a["aspect"])
                        st.metric(a["label"], f"{a['score']:.2f}")
                        if a.get("evidence"):
                            for snippet in a["evidence"]:
                                st.markdown(f"> {snippet}")
                        else:
                            st.caption("No supporting text found.")

                disagreements = [
                    a["aspect"]
                    for a in aspects
                    if a["label"] in {"negative", "positive"} and a["label"] != overall
                ]
                if disagreements:
                    st.success(
                        "Aspects disagreeing with the overall label: "
                        + ", ".join(disagreements)
                        + ". This is the case aspect-based sentiment exists for."
                    )
                else:
                    st.warning(
                        "No aspect disagreed with the overall label on this review. "
                        "That is fine once. If it never happens, your aspect step is "
                        "probably just repeating the document classifier — test it on "
                        "a deliberately mixed review before you claim otherwise."
                    )

# ---------------------------------------------------------------------------
with tab_performance:
    st.header("Model Performance")
    st.caption(
        "Held-out metrics, read straight from the `runs` table. Nothing on this "
        "tab is computed in the browser."
    )

    run = require_run()
    if run:
        st.caption(
            f"Run {run['id']} · {run.get('model_version')} · base "
            f"{run.get('base_model')} · dataset {run.get('dataset')} · "
            f"{run.get('created_at')}"
        )
        documents = metrics_of(run).get("documents", [])
        if not documents:
            st.info("This run has no `metrics.documents` payload.")
        else:
            st.subheader("Transformer vs baseline")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "model": d.get("model_name"),
                            "n": d.get("n"),
                            "accuracy": d.get("accuracy"),
                            "macro precision": d.get("macro_precision"),
                            "macro recall": d.get("macro_recall"),
                            "macro F1": d.get("macro_f1"),
                            "ROC AUC": d.get("roc_auc"),
                        }
                        for d in documents
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Both rows are required. If the transformer does not beat TF-IDF "
                "here, that is a result to report and explain, not a bug to hide."
            )

            names = [d.get("model_name", f"model {i}") for i, d in enumerate(documents)]
            chosen = st.selectbox("Show curves for", names)
            doc = documents[names.index(chosen)]
            labels = doc.get("labels") or LABEL_ORDER

            st.subheader("Per-class metrics")
            per_class = doc.get("per_class") or {}
            if per_class:
                st.dataframe(
                    pd.DataFrame(per_class).T.reset_index(names="class"),
                    use_container_width=True,
                    hide_index=True,
                )

            left, right = st.columns(2)

            with left:
                st.subheader("Confusion matrix")
                cm = doc.get("confusion_matrix") or []
                if cm:
                    st.dataframe(
                        pd.DataFrame(
                            cm,
                            index=[f"gold {c}" for c in labels],
                            columns=[f"predicted {c}" for c in labels],
                        ),
                        use_container_width=True,
                    )
                    st.caption(
                        "Rows are the truth, columns are the prediction. Say that "
                        "out loud in your demo — a transposed matrix turns a "
                        "false-positive problem into a false-negative one."
                    )
                else:
                    st.info("No confusion matrix in this run.")

            with right:
                st.subheader("ROC curve")
                roc = doc.get("roc_points") or []
                if roc:
                    roc_df = pd.DataFrame(roc).sort_values("fpr")
                    roc_df["chance"] = roc_df["fpr"]
                    st.line_chart(roc_df, x="fpr", y=["tpr", "chance"])
                    st.caption(
                        "The curve describes the SCORES at every threshold. It can "
                        "look excellent for a model whose 0.5 threshold is in the "
                        "wrong place — which is why the confusion matrix is next "
                        "to it."
                    )
                else:
                    st.info("No ROC points in this run.")

            st.subheader("Calibration (reliability curve)")
            bins = doc.get("calibration_bins") or []
            if bins:
                cal_df = pd.DataFrame(bins)
                cal_df["perfect"] = cal_df["mean_predicted"]
                st.line_chart(
                    cal_df, x="mean_predicted", y=["observed_positive_rate", "perfect"]
                )
                st.dataframe(cal_df, use_container_width=True, hide_index=True)
                st.caption(
                    "The x-axis is what the model claimed; `observed_positive_rate` "
                    "is what actually happened. If your curve sits exactly on "
                    "`perfect` first time, check that you did not plot the mean "
                    "predicted probability against itself."
                )
            else:
                st.info(
                    "No calibration bins in this run. An uncalibrated model is "
                    "allowed — saying nothing about it is not."
                )

            st.subheader("Per-aspect metrics")
            aspects = metrics_of(run).get("aspects", [])
            if aspects:
                st.dataframe(pd.DataFrame(aspects), use_container_width=True, hide_index=True)
                st.caption(
                    "`n_evaluated` is the column to read first. Three-decimal F1 on "
                    "forty annotated reviews is a number with an error bar nobody "
                    "has drawn."
                )
            else:
                st.info("No `metrics.aspects` payload in this run.")

# ---------------------------------------------------------------------------
with tab_bias:
    st.header("Bias Audit")
    st.caption(
        "Where the model is worse, on which kinds of input, and what that means "
        "for how it may be used."
    )

    run = require_run()
    if run:
        slices = metrics_of(run).get("slices", [])
        if not slices:
            st.info(
                "No `metrics.slices` payload in this run. Call "
                "`nlp.evaluate_slices()` once per slice in your training script "
                "and store the concatenated list."
            )
        else:
            df = pd.DataFrame(slices)
            slice_names = sorted(df["slice_name"].unique())
            if len(slice_names) < 2:
                st.warning(
                    f"Only one slice ({slice_names[0]}) is reported. The assignment "
                    "requires at least two."
                )
            for name in slice_names:
                sub = df[df["slice_name"] == name].copy()
                observed = bool(sub["observed"].all()) if "observed" in sub else True
                st.subheader(f"Slice: {name}")
                if not observed:
                    st.warning(
                        "This attribute was **inferred, not observed.** A gap in "
                        "these numbers may be a gap in the sentiment model, a gap "
                        "in whatever assigned the attribute, or both — and this "
                        "table cannot tell you which. Report it as a hypothesis."
                    )
                st.dataframe(
                    sub[["bucket", "n", "accuracy", "macro_f1"]],
                    use_container_width=True,
                    hide_index=True,
                )
                if sub["n"].sum() > 0:
                    st.bar_chart(sub.set_index("bucket")[["accuracy", "macro_f1"]])
                small = sub[(sub["n"] > 0) & (sub["n"] < 100)]["bucket"].tolist()
                if small:
                    st.caption(
                        "Fewer than 100 items in: "
                        + ", ".join(small)
                        + ". Read the gap next to the sample size before calling "
                        "it bias."
                    )

        st.divider()
        st.subheader("Documented failures")
        st.caption(
            "Read from the 'Documented failures' section of MODEL_CARD.md, so the "
            "audit and the model card cannot disagree. At least three, each with "
            "the ethical risk it illustrates and the use limitation it implies."
        )
        failures = model_card_section("Documented failures")
        if failures:
            st.markdown(failures)
        else:
            st.warning(
                "No 'Documented failures' section found in MODEL_CARD.md. This is a "
                "graded requirement: three real failures from your own testing, each "
                "with the risk and the limitation."
            )

        st.divider()
        st.subheader("Recent predictions")
        st.caption("GET /audit — what the live service actually served, hashes only.")
        limit = st.slider("How many", 10, 200, 50, step=10)
        only = st.selectbox("Label filter", ["(all)", "negative", "positive"])
        payload_path = f"/audit?limit={limit}"
        if only != "(all)":
            payload_path += f"&label={only}"
        data = call_api(payload_path, method="get")
        if data is not None:
            rows = data.get("predictions", [])
            if rows:
                audit_df = pd.DataFrame(rows)
                audit_df["text_sha256"] = audit_df["text_sha256"].str.slice(0, 12) + "…"
                st.dataframe(audit_df, use_container_width=True, hide_index=True)
            else:
                st.info("Nothing logged yet. Score some text and come back.")

# ---------------------------------------------------------------------------
with tab_card:
    st.header("Model Card")
    st.caption("Rendered from MODEL_CARD.md in the repository root.")
    card = model_card_text()
    if card:
        st.markdown(card)
    else:
        st.warning("MODEL_CARD.md not found — fill it in and commit it.")
