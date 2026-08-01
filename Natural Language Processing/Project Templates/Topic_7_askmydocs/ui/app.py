"""Streamlit UI — Cloud #1, deployed on Streamlit Community Cloud.

FINISHED CODE. All four required tabs exist and are wired to the API. Until you
implement api/nlp.py the product tabs will show a clear "not implemented yet"
message instead of results — that is the template telling you where to work, not
a bug.

Two rules this file follows:

* It is a thin client. It never imports torch, transformers, sentence-transformers
  or api.db. It calls HTTP endpoints and renders JSON. The generator lives in the
  API tier because that is the tier with the memory budget and the model cache;
  putting it here would mean loading it into every browser session.
* The direct database read (Retrieval Audit) uses the ANON key, which row-level
  security restricts to SELECT. The service-role key never leaves the API tier.

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

st.set_page_config(page_title="AskMyDocs", page_icon="📚", layout="wide")


def call_api(path: str, payload: dict | None = None, method: str = "post"):
    """POST/GET against the API and turn failures into readable messages."""
    try:
        fn = requests.post if method == "post" else requests.get
        # Generation on a free-plan instance is slow and the first call after
        # idle also has to wake the service up. 180 seconds is not generous.
        resp = fn(f"{API_URL}{path}", json=payload, timeout=180)
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
    if resp.status_code == 409:
        st.error(
            "The service refused this ingest because the two corpora overlap:\n\n"
            f"> {resp.json().get('detail', '')}"
        )
        return None
    if resp.status_code >= 400:
        st.error(f"API returned {resp.status_code}: {resp.text[:400]}")
        return None
    return resp.json()


def render_citations(passages: list[dict]) -> None:
    """Numbered passages with their similarity scores, as the citation list."""
    for p in passages:
        title = p.get("document_title") or f"document {p['document_id']}"
        with st.expander(
            f"[{p['rank']}] {title} · chunk {p['chunk_id']} · similarity {p['similarity']:.3f}"
        ):
            st.write(p["text"])


st.title("AskMyDocs")
st.caption(
    "Retrieval-augmented answers over your own documents — "
    "Streamlit ⇄ FastAPI ⇄ Postgres + pgvector"
)

with st.sidebar:
    st.subheader("What is in the store")
    st.caption("GET /sources")
    if st.button("Refresh sources"):
        st.session_state["sources"] = call_api("/sources", method="get")
    data = st.session_state.get("sources")
    if data:
        st.metric("Documents", data["document_count"])
        st.metric("Chunks", data["chunk_count"])
        df = pd.DataFrame(data["sources"])
        if not df.empty:
            st.dataframe(df[["title", "corpus", "chunk_count"]], use_container_width=True)
            # The one thing to look at here: no title should appear in both
            # corpora. If one does, the with/without comparison is contaminated.
            both = (
                df.groupby("title")["corpus"].nunique().pipe(lambda s: s[s > 1]).index.tolist()
            )
            if both:
                st.error(
                    "These titles appear in BOTH corpora, which breaks the "
                    f"comparison: {', '.join(both)}"
                )
    else:
        st.caption("No sources loaded yet.")

tab_concepts, tab_ask, tab_audit, tab_card = st.tabs(
    ["Concepts", "Ask", "Retrieval Audit", "Model Card"]
)

# ---------------------------------------------------------------------------
with tab_concepts:
    st.header("Concepts")
    st.info(
        "**This tab is yours to write.** Explain, for a knowledge-management "
        "lead who is not an engineer: what a language model is actually doing "
        "when it predicts the next token; what perplexity measures and what it "
        "does not; what an embedding is and why nearby vectors mean related "
        "text; and what retrieval-augmented generation adds on top. Use your own "
        "output from the Ask and Retrieval Audit tabs — a screenshot of your own "
        "retrieved passage beside your own answer beats any textbook diagram."
    )
    st.markdown(
        """
        Suggested structure (delete this list once you have written the tab):

        1. **Language modelling.** Next-token prediction, and why "predict the
           next word" turns out to be enough to produce fluent text — and why
           fluent is not the same as true.
        2. **Perplexity.** Your held-out number, what it means in words ("about
           as uncertain as choosing between N options"), and the two caveats:
           it is not comparable across tokenizers, and it says nothing about
           whether an answer is correct.
        3. **Embeddings.** One vector per passage; similar text lands nearby.
           Show two passages from your own collection with a high similarity and
           two with a low one, and say whether you agree with the model.
        4. **RAG.** The pipeline in five boxes: chunk, embed, store, retrieve,
           condition. Then your own worked example where retrieval changed the
           answer, with the passage quoted.
        5. **What it still gets wrong.** The retrieved-but-ignored case, the
           question your chunking cannot answer, the confident wrong citation.
        """
    )

# ---------------------------------------------------------------------------
with tab_ask:
    st.header("Ask")
    question = st.text_area(
        "Question",
        value="What is the escalation procedure for a request raised after hours?",
        height=90,
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        k = st.slider("Passages to retrieve (k)", min_value=1, max_value=20, value=5)
    with c2:
        compare = st.checkbox(
            "Also answer without retrieval (the control condition)",
            value=True,
            help="Same question, same model, same decoding — no passages. This is "
            "the comparison the assignment asks you to run on at least ten "
            "questions.",
        )

    if st.button("Ask", type="primary"):
        with st.spinner("Retrieving and generating..."):
            grounded = call_api(
                "/ask", {"question": question, "k": k, "use_retrieval": True}
            )
            ungrounded = (
                call_api("/ask", {"question": question, "k": k, "use_retrieval": False})
                if compare
                else None
            )

        if grounded:
            if compare and ungrounded:
                left, right = st.columns(2)
                with left:
                    st.subheader("With retrieval")
                    st.write(grounded["answer"])
                with right:
                    st.subheader("Without retrieval")
                    st.write(ungrounded["answer"])
                    st.caption(
                        "Read this one carefully. If it is already correct, either "
                        "the generator knew the answer from pretraining or the "
                        "fine-tuning corpus overlaps the retrieval collection — "
                        "and only one of those is your fault."
                    )
            else:
                st.subheader("Answer")
                st.write(grounded["answer"])

            m1, m2, m3 = st.columns(3)
            m1.metric("Passages retrieved", len(grounded.get("retrieved", [])))
            m2.metric("Passages cited", len(grounded.get("cited_chunk_ids", [])))
            m3.metric("Prompt tokens", grounded.get("prompt_token_count") or 0)

            retrieved = grounded.get("retrieved", [])
            if retrieved:
                st.subheader("Sources")
                render_citations(retrieved)

                cited = set(grounded.get("cited_chunk_ids", []))
                ignored = [p for p in retrieved if p["chunk_id"] not in cited]
                if ignored:
                    st.caption(
                        f"{len(ignored)} of {len(retrieved)} retrieved passages were "
                        "not cited. That is worth a look: a passage that was found "
                        "and then ignored is a prompting problem, not a retrieval "
                        "problem, and the two have different fixes."
                    )
            else:
                st.warning(
                    "Nothing was retrieved. Either the store is empty, or the query "
                    "embedding and the stored embeddings came from different models."
                )

# ---------------------------------------------------------------------------
with tab_audit:
    st.header("Retrieval Audit")
    st.caption(
        "Read directly from Postgres with the anon key — no API call. Every "
        "question ever asked, what came back for it, and how close it was."
    )
    if not (SUPABASE_URL and SUPABASE_ANON_KEY):
        st.info(
            "Add SUPABASE_URL and SUPABASE_ANON_KEY to `.streamlit/secrets.toml` "
            "(and to the Secrets box on Streamlit Cloud) to enable this tab."
        )
    else:
        try:
            from supabase import create_client

            client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
            rows = (
                client.table("retrievals")
                .select(
                    "id, query_id, chunk_id, similarity, rank, created_at, "
                    "chunks(document_id, ordinal, content), "
                    "queries(query_sha256, k, model_version, created_at)"
                )
                .order("id", desc=True)
                .limit(200)
                .execute()
                .data
            )
            if not rows:
                st.info("No retrievals logged yet. Ask a question and come back.")
            else:
                flat = []
                for r in rows:
                    chunk = r.get("chunks") or {}
                    query = r.get("queries") or {}
                    text = chunk.get("content") or ""
                    flat.append(
                        {
                            "query_id": r["query_id"],
                            # The question itself is never stored — only its hash.
                            # Same hash means the same question was asked twice.
                            "query_sha256": (query.get("query_sha256") or "")[:12],
                            "k": query.get("k"),
                            "rank": r["rank"],
                            "similarity": r["similarity"],
                            "chunk_id": r["chunk_id"],
                            "document_id": chunk.get("document_id"),
                            "passage": text[:160] + ("..." if len(text) > 160 else ""),
                            "model_version": query.get("model_version"),
                        }
                    )
                df = pd.DataFrame(flat)
                st.dataframe(df, use_container_width=True)

                st.subheader("Similarity by rank")
                st.caption(
                    "The shape of this is diagnostic. A steep drop from rank 1 to "
                    "rank 5 means the retriever is discriminating. A flat line near "
                    "the same value for every rank means it is not — every passage "
                    "looks equally relevant, which usually means the query and the "
                    "chunks were embedded by different models, or the vectors were "
                    "never normalised."
                )
                st.bar_chart(df.groupby("rank")["similarity"].mean())

                answers = (
                    client.table("answers")
                    .select("query_id, retrieval_used, cited_chunk_ids, generator_model")
                    .order("id", desc=True)
                    .limit(200)
                    .execute()
                    .data
                )
                if answers:
                    st.subheader("Answers logged")
                    st.dataframe(pd.DataFrame(answers), use_container_width=True)
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
