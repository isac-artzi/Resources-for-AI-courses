"""Streamlit UI — Cloud #1, deployed on Streamlit Community Cloud.

FINISHED CODE. All four required tabs exist and are wired to the API. Until you
implement api/nlp.py the product tabs will show a clear "not implemented yet"
message instead of results — that is the template telling you where to work, not
a bug.

Three rules this file follows:

* It is a thin client. It never imports torch, transformers, or api.db. It calls
  HTTP endpoints and renders JSON.
* Reads of the History tab go straight to Postgres with the ANON key, which
  row-level security limits to SELECT.
* Writes — the rating form — go through the API, which holds the service-role
  key. The browser never gets a key that can write.

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

STRATEGIES = ["greedy", "beam", "temperature", "top_k", "top_p"]

st.set_page_config(page_title="GenText", page_icon="✍️", layout="wide")


def call_api(path: str, payload: dict | None = None, method: str = "post"):
    """POST/GET against the API and turn failures into readable messages.

    The timeout is generous on purpose. Generation on a free CPU instance is slow
    (beam search especially), and the first call after an idle period also pays
    for the container waking up and the weights loading.
    """
    try:
        if method == "post":
            resp = requests.post(f"{API_URL}{path}", json=payload, timeout=180)
        else:
            resp = requests.get(f"{API_URL}{path}", params=payload, timeout=60)
    except requests.Timeout:
        st.error(
            "The API did not answer in time. On the free tier the first request "
            "after an idle period wakes the container and loads the model, which "
            "can take a minute. Try once more before assuming it is broken."
        )
        return None
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


def decoding_controls(key_prefix: str, default_strategy: str = "top_p") -> dict:
    """The decoding panel, shared by the Generate and Compare tabs.

    Controls that the selected strategy ignores are disabled rather than hidden.
    Seeing `top_k` greyed out under greedy decoding teaches the point; hiding it
    lets the user assume it was applied.
    """
    strategy = st.selectbox(
        "Decoding strategy",
        STRATEGIES,
        index=STRATEGIES.index(default_strategy),
        key=f"{key_prefix}_strategy",
    )
    samples = strategy in {"temperature", "top_k", "top_p"}

    c1, c2 = st.columns(2)
    with c1:
        max_new_tokens = st.slider(
            "Max new tokens", 16, 256, 80, step=8, key=f"{key_prefix}_len"
        )
        temperature = st.slider(
            "Temperature", 0.1, 2.0, 0.9, step=0.05,
            disabled=not samples,
            help="Scales the logits before the softmax. Below 1 makes the model "
                 "conservative; above 1 makes it adventurous. Ignored unless you "
                 "are sampling.",
            key=f"{key_prefix}_temp",
        )
        repetition_penalty = st.slider(
            "Repetition penalty", 1.0, 2.0, 1.0, step=0.05,
            help="1.0 is off. Small increases fix most loops; large ones stop the "
                 "model repeating words it legitimately needs, like a name.",
            key=f"{key_prefix}_rep",
        )
    with c2:
        top_k = st.slider(
            "top-k", 0, 200, 50, step=5,
            disabled=strategy != "top_k",
            help="Sample from the k most likely tokens only. Used by the top-k "
                 "strategy.",
            key=f"{key_prefix}_topk",
        )
        top_p = st.slider(
            "top-p (nucleus)", 0.05, 1.0, 0.95, step=0.01,
            disabled=strategy != "top_p",
            help="Sample from the smallest set of tokens whose probability adds "
                 "up to p. Used by the top-p strategy.",
            key=f"{key_prefix}_topp",
        )
        num_beams = st.slider(
            "Beams", 1, 12, 4,
            disabled=strategy != "beam",
            help="Beam width. Wider is slower and usually less varied.",
            key=f"{key_prefix}_beams",
        )

    use_seed = st.checkbox(
        "Fix the random seed", value=False,
        disabled=not samples,
        help="On for a demo you need to repeat; off to show that sampling gives "
             "a different answer every time.",
        key=f"{key_prefix}_useseed",
    )
    seed = st.number_input(
        "Seed", value=42, step=1, disabled=not (samples and use_seed),
        key=f"{key_prefix}_seed",
    )

    if strategy == "temperature":
        st.caption(
            "Temperature sampling here means the FULL vocabulary: no top-k, no "
            "top-p. That is what makes it a fair contrast with the two "
            "truncating strategies."
        )

    return {
        "strategy": strategy,
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "num_beams": int(num_beams),
        "repetition_penalty": float(repetition_penalty),
        "seed": int(seed) if (samples and use_seed) else None,
    }


def render_generation(data: dict) -> None:
    """Result panel used by the Generate tab."""
    st.text_area(
        "Generated text", value=data["generated_text"], height=220, disabled=True
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("New tokens", data.get("generated_token_count") or 0)
    d2 = data.get("distinct_2")
    m2.metric("distinct-2", f"{d2:.3f}" if isinstance(d2, (int, float)) else "—")
    ppl = data.get("perplexity")
    m3.metric("Perplexity", f"{ppl:.1f}" if isinstance(ppl, (int, float)) else "—")
    lat = data.get("latency_ms")
    m4.metric("Latency", f"{lat:.0f} ms" if isinstance(lat, (int, float)) else "—")
    if isinstance(d2, (int, float)) and d2 < 0.5:
        st.warning(
            "distinct-2 below 0.5 usually means the output is looping. Before you "
            "reach for more training: try sampling instead of greedy or beam, or "
            "nudge the repetition penalty."
        )
    st.caption(
        f"prompt sha256 `{data['prompt_sha256'][:16]}…` · model "
        f"`{data.get('model_version') or 'unknown'}` · generation id "
        f"`{data.get('generation_id') or 'not logged'}`"
    )


st.title("GenText")
st.caption(
    "Controllable text generation as a service — Streamlit ⇄ FastAPI ⇄ Postgres"
)

tab_concepts, tab_generate, tab_compare, tab_history = st.tabs(
    ["Concepts", "Generate", "Compare Decoding", "History"]
)

# ---------------------------------------------------------------------------
with tab_concepts:
    st.header("Concepts")
    st.info(
        "**This tab is yours to write.** Explain, for a content person who is not "
        "an engineer: what the model is actually doing at each step, how greedy, "
        "beam search and sampling differ, and what temperature, top-k and top-p "
        "each do to the distribution before a token is picked. Use outputs from "
        "your own Compare Decoding tab as the examples — your own repetitive "
        "greedy paragraph is worth more than a textbook diagram."
    )
    st.markdown(
        """
        Suggested structure (delete this list once you have written the tab):

        1. **One step at a time.** The model produces a probability over the whole
           vocabulary for the next token. Decoding is the rule for choosing one.
           Everything below is a different rule.
        2. **Greedy.** Always the most likely token. Deterministic, and it loops:
           once a phrase is the most likely continuation of itself, nothing breaks
           the cycle. Show your own loop.
        3. **Beam search.** Keep several partial sequences and return the highest
           scoring one. Better for tasks with a right answer (translation,
           summarization); for open-ended writing it converges on the safe,
           generic sentence, and widening the beam makes that worse, not better.
        4. **Sampling, and the two ways to make it safer.** Temperature reshapes
           the whole distribution. top-k truncates to a fixed count. top-p
           truncates to a probability mass, so it adapts: where the model is
           confident the nucleus is a handful of tokens, where it is unsure the
           nucleus is hundreds. Say why that adaptivity is the argument for top-p.
        5. **What each control costs.** High temperature buys variety and sells
           coherence. A tight nucleus buys safety and sells surprise. Put your own
           numbers on it from the Compare tab.
        6. **Why we store the settings.** Every row in History carries the exact
           parameters, so a good output can be reproduced and a bad one explained.
        """
    )

# ---------------------------------------------------------------------------
with tab_generate:
    st.header("Generate")
    prompt = st.text_area(
        "Prompt",
        value="Write a short product review of a pair of running shoes:",
        height=120,
    )
    params = decoding_controls("gen")

    if st.button("Generate", type="primary"):
        if not prompt.strip():
            st.warning("Enter a prompt first.")
        else:
            with st.spinner("Generating — the first call after idle is the slow one."):
                data = call_api("/generate", {"prompt": prompt, "params": params})
            if data:
                st.session_state["last_generation"] = data

    if st.session_state.get("last_generation"):
        render_generation(st.session_state["last_generation"])

        gen_id = st.session_state["last_generation"].get("generation_id")
        st.divider()
        st.subheader("Rate this output")
        if not gen_id:
            st.info(
                "This generation was not logged (Supabase is not configured), so "
                "there is no row to attach a rating to. Fill in .env and restart "
                "the API."
            )
        else:
            with st.form("rate_last"):
                r1, r2 = st.columns(2)
                with r1:
                    rater_id = st.text_input(
                        "Rater id", value="rater-a",
                        help="A label, not a name. This repository is public.",
                    )
                    overall = st.slider("Overall quality", 1, 5, 3)
                with r2:
                    coherence = st.slider("Coherence", 1, 5, 3)
                    fluency = st.slider("Fluency", 1, 5, 3)
                    relevance = st.slider("Relevance to the prompt", 1, 5, 3)
                notes = st.text_input("Notes (what made it that score?)")
                if st.form_submit_button("Submit rating"):
                    resp = call_api(
                        "/rate",
                        {
                            "generation_id": gen_id,
                            "rater_id": rater_id,
                            "rating": overall,
                            "coherence": coherence,
                            "fluency": fluency,
                            "relevance": relevance,
                            "notes": notes or None,
                        },
                    )
                    if resp:
                        st.success(
                            f"Stored. Generation {resp['generation_id']} now has "
                            f"{resp['rating_count']} rating(s); aggregate "
                            f"{resp['human_rating']}."
                        )

# ---------------------------------------------------------------------------
with tab_compare:
    st.header("Compare Decoding")
    st.caption(
        "One prompt, several strategies, side by side. This is the tab your "
        "report's decoding section comes from."
    )
    cprompt = st.text_area(
        "Prompt",
        value="The morning after the storm, the harbour",
        height=100,
        key="compare_prompt",
    )
    chosen = st.multiselect(
        "Strategies to run", STRATEGIES, default=["greedy", "beam", "top_p"]
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        c_len = st.slider("Max new tokens", 16, 256, 80, step=8, key="cmp_len")
    with c2:
        c_temp = st.slider("Temperature (sampling only)", 0.1, 2.0, 0.9, step=0.05,
                           key="cmp_temp")
    with c3:
        c_beams = st.slider("Beams (beam only)", 2, 12, 5, key="cmp_beams")

    st.caption(
        "Every strategy here runs sequentially against a free CPU instance. Three "
        "strategies at 80 tokens is comfortable; five with a wide beam will test "
        "your patience and possibly the request timeout."
    )

    if st.button("Run comparison", type="primary"):
        if not chosen:
            st.warning("Pick at least one strategy.")
        elif not cprompt.strip():
            st.warning("Enter a prompt first.")
        else:
            rows = []
            progress = st.progress(0.0)
            for i, strat in enumerate(chosen, start=1):
                payload = {
                    "prompt": cprompt,
                    "params": {
                        "strategy": strat,
                        "max_new_tokens": int(c_len),
                        "temperature": float(c_temp),
                        "top_k": 50,
                        "top_p": 0.95,
                        "num_beams": int(c_beams),
                        "repetition_penalty": 1.0,
                        "seed": None,
                    },
                }
                data = call_api("/generate", payload)
                progress.progress(i / len(chosen))
                if data:
                    rows.append(data)
            progress.empty()

            if rows:
                for r in rows:
                    with st.expander(f"{r['strategy']} — {r['generated_text'][:80]}…",
                                     expanded=True):
                        st.write(r["generated_text"])
                st.subheader("Side by side")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "strategy": r["strategy"],
                                "new tokens": r.get("generated_token_count"),
                                "distinct-1": r.get("distinct_1"),
                                "distinct-2": r.get("distinct_2"),
                                "perplexity": r.get("perplexity"),
                                "latency (ms)": r.get("latency_ms"),
                                "generation id": r.get("generation_id"),
                            }
                            for r in rows
                        ]
                    ),
                    use_container_width=True,
                )
                st.caption(
                    "Read the table and the text together. Greedy and beam should "
                    "show the lowest distinct-2 and often the lowest perplexity — "
                    "which is precisely why perplexity alone is not a quality "
                    "score. Beam search should also be the slowest by a wide "
                    "margin; that latency column is a real finding for your report."
                )

# ---------------------------------------------------------------------------
with tab_history:
    st.header("History")
    st.caption(
        "Read directly from Postgres with the anon key — no API call. Every "
        "generation, its settings, and its ratings."
    )
    if not (SUPABASE_URL and SUPABASE_ANON_KEY):
        st.info(
            "Add SUPABASE_URL and SUPABASE_ANON_KEY to `.streamlit/secrets.toml` "
            "(and to the Secrets box on Streamlit Cloud) to enable this tab."
        )
    else:
        try:
            from supabase import create_client

            supa = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
            rows = (
                supa.table("generations")
                .select("*")
                .order("created_at", desc=True)
                .limit(200)
                .execute()
                .data
            )
            if not rows:
                st.info(
                    "No generations logged yet. Generate something and come back. "
                    "(If you have generated text and this is still empty, the "
                    "insert is failing — check the API logs. Do not turn RLS off.)"
                )
            else:
                df = pd.DataFrame(rows)
                f1, f2 = st.columns(2)
                with f1:
                    pick = st.multiselect(
                        "Filter by strategy", sorted(df["strategy"].unique())
                    )
                with f2:
                    only_unrated = st.checkbox("Only unrated outputs", value=False)
                view = df
                if pick:
                    view = view[view["strategy"].isin(pick)]
                if only_unrated:
                    view = view[view["human_rating"].isna()]

                st.dataframe(view, use_container_width=True, height=340)

                rated = df[df["human_rating"].notna()]
                m1, m2, m3 = st.columns(3)
                m1.metric("Generations logged", len(df))
                m2.metric("Rated", len(rated))
                m3.metric(
                    "Mean rating",
                    f"{rated['human_rating'].mean():.2f}" if len(rated) else "—",
                )
                st.caption(
                    "The assignment asks for at least 20 outputs, each scored "
                    "independently by two people. The mean is the least "
                    "interesting number here — the disagreements are the finding."
                )

                st.divider()
                st.subheader("Rate an output")
                st.caption(
                    "Writes go through the API, not through this page: the anon "
                    "key can read but cannot write, by design."
                )
                ids = view["id"].tolist()
                if not ids:
                    st.info("Nothing matches the current filter.")
                else:
                    chosen_id = st.selectbox("Generation id", ids)
                    row = df[df["id"] == chosen_id].iloc[0]
                    st.text_area(
                        "Output", value=str(row["generated_text"]), height=160,
                        disabled=True, key="hist_text",
                    )
                    existing = row.get("ratings") or []
                    if isinstance(existing, list) and existing:
                        st.write("Ratings so far:")
                        st.dataframe(pd.DataFrame(existing), use_container_width=True)
                    with st.form("rate_from_history"):
                        h1, h2 = st.columns(2)
                        with h1:
                            h_rater = st.text_input("Rater id", value="rater-b")
                            h_overall = st.slider("Overall quality", 1, 5, 3)
                        with h2:
                            h_coh = st.slider("Coherence", 1, 5, 3)
                            h_flu = st.slider("Fluency", 1, 5, 3)
                            h_rel = st.slider("Relevance", 1, 5, 3)
                        h_notes = st.text_input("Notes")
                        if st.form_submit_button("Submit rating"):
                            resp = call_api(
                                "/rate",
                                {
                                    "generation_id": int(chosen_id),
                                    "rater_id": h_rater,
                                    "rating": h_overall,
                                    "coherence": h_coh,
                                    "fluency": h_flu,
                                    "relevance": h_rel,
                                    "notes": h_notes or None,
                                },
                            )
                            if resp:
                                st.success(
                                    f"Stored — {resp['rating_count']} rating(s) on "
                                    f"generation {resp['generation_id']}. Reload the "
                                    "tab to see it."
                                )
        except Exception as exc:
            st.error(f"Supabase read failed: {exc}")
