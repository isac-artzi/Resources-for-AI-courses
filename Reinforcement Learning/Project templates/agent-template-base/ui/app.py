"""
ui/app.py — the presentation tier. The only thing a non-technical stakeholder
ever needs to open.

Three rules this file obeys, and yours must too:

  1. It contains NO policy code and NO training code. Every decision comes
     back from the service tier through ui/service.py.
  2. It never issues SQL that changes state. Read-only views only, through the
     anon key.
  3. It degrades visibly. A paused database or a missing artifact produces a
     clearly worded panel, never a stack trace. Supabase free-tier projects
     pause after a week idle, so this will happen to you — probably the night
     before a deadline.

Run it:  streamlit run ui/app.py
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from shared.config import get_settings          # noqa: E402
from ui import service                          # noqa: E402

settings = get_settings()

st.set_page_config(page_title="Agent Product", layout="wide")


# ---------------------------------------------------------------------------
# Health banner. Shown before anything else, because every tab below depends
# on the tiers it reports on.
# ---------------------------------------------------------------------------
def health_banner() -> dict:
    try:
        h = service.healthz()
    except service.ServiceError as exc:
        st.error(f"**The service tier is unavailable.** {exc}")
        return {"status": "degraded", "policy_artifact_loaded": False,
                "data_tier_reachable": False}

    if h["status"] == "ok":
        st.success(
            f"Service healthy · mode `{settings.service_mode}` · "
            f"policy loaded · data tier reachable"
        )
    else:
        st.warning(
            f"**Running in a degraded state.** {h.get('detail') or ''}\n\n"
            "Results below may be incomplete. This banner is deliberate: an "
            "empty chart and a broken database look identical otherwise."
        )
    return h


health = health_banner()

st.title("Agent Product")
st.caption(
    "A reinforcement learning agent, trained offline and served as a NumPy "
    "artifact behind a typed HTTP contract."
)

TABS = st.tabs(["Concepts", "Watch", "Compare", "Run History", "Model Card"])


# ---------------------------------------------------------------------------
with TABS[0]:
    st.header("Concepts")
    st.markdown(
        """
This tab is the **Theoretical Brief** required of every product: 350–600 words
explaining the mathematics in language a non-specialist colleague could follow.
Replace this placeholder with the derivation your topic's objectives call for,
and render the equations with `st.latex`.
        """
    )
    st.latex(r"v_\pi(s) = \sum_a \pi(a\mid s)\sum_{s'} P(s'\mid s,a)\big[R + \gamma\,v_\pi(s')\big]")
    st.info(
        "Write for the stakeholder in the product brief, not for the grader. "
        "If a sentence would not survive being read aloud in a meeting, cut it."
    )


# ---------------------------------------------------------------------------
with TABS[1]:
    st.header("Watch the agent act")

    if not health.get("policy_artifact_loaded"):
        st.error(
            "No policy artifact is loaded, so there is nothing to watch. "
            "Train an agent and run `python -m train.export`, then reload."
        )
    else:
        try:
            available = [p["name"] for p in service.policies()["policies"]]
        except service.ServiceError as exc:
            available, _ = [], st.error(str(exc))

        col_a, col_b, col_c = st.columns([2, 1, 1])
        policy_name = col_a.selectbox("Policy", available or ["default"])
        episodes = col_b.slider("Evaluation episodes", 1, 100, 20)
        seed = col_c.number_input("Seed", value=0, step=1,
                                  help="Name the seed or the result is not evidence.")

        if st.button("Run a greedy evaluation"):
            try:
                r = service.rollout(policy_name, episodes, int(seed))
            except service.ServiceError as exc:
                st.error(str(exc))
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("Mean return", f"{r['mean_return']:.3f}")
                m2.metric("Standard error", f"{r['stderr_return']:.3f}",
                          help="s / sqrt(n). Quote this alongside the mean, never the mean alone.")
                m3.metric("Mean episode length", f"{r['mean_length']:.1f}")
                st.caption(
                    f"artifact `{r['policy_sha256'][:12]}…` · {r['episodes']} episodes · "
                    f"seed {r['seed']}"
                )
                if r["trajectory"]:
                    st.subheader("One full trajectory")
                    st.dataframe(pd.DataFrame(r["trajectory"]), use_container_width=True,
                                 height=280)
                st.bar_chart(pd.DataFrame({"return": r["returns"]}))


# ---------------------------------------------------------------------------
with TABS[2]:
    st.header("Compare configurations")
    st.markdown(
        "Every claim of the form *A beats B* must name the number of independent "
        "seeds behind it. This tab reads the run history and groups by "
        "configuration; a single-seed comparison is rendered but flagged."
    )
    try:
        data = service.runs(200)
    except service.ServiceError as exc:
        st.error(str(exc))
        data = {"runs": [], "degraded": True}

    if data.get("degraded"):
        st.warning("The data tier did not answer, so this table is empty rather than complete.")
    elif not data["runs"]:
        st.info("No runs logged yet. Train an agent — every episode should land in `episodes`.")
    else:
        df = pd.DataFrame(data["runs"])
        grouped = (
            df.groupby("algorithm")
            .agg(seeds=("seed", "nunique"),
                 mean_eval=("eval_mean_return", "mean"),
                 episodes=("episodes_logged", "sum"))
            .reset_index()
        )
        st.dataframe(grouped, use_container_width=True)
        thin = grouped[grouped["seeds"] < 3]
        if not thin.empty:
            st.warning(
                "These configurations have fewer than three seeds: "
                f"{', '.join(thin['algorithm'])}. That is not enough to separate the "
                "configuration's effect from the seed's."
            )


# ---------------------------------------------------------------------------
with TABS[3]:
    st.header("Run history")
    st.caption("Read-only. This tab issues no writes and holds no service-role key.")
    try:
        data = service.runs(100)
        if data["runs"]:
            st.dataframe(pd.DataFrame(data["runs"]), use_container_width=True, height=420)
        else:
            st.info("Nothing logged yet.")
    except service.ServiceError as exc:
        st.error(str(exc))

    st.subheader("Registered artifacts")
    try:
        st.dataframe(pd.DataFrame(service.policies()["policies"]), use_container_width=True)
    except service.ServiceError as exc:
        st.error(str(exc))


# ---------------------------------------------------------------------------
with TABS[4]:
    st.header("Model card")
    st.markdown(
        """
**What this agent does.** _One paragraph, in the stakeholder's language._

**What it does not do.** _State the boundary explicitly._

**Training data and environment.** _Environment id, number of episodes, seeds._

**Evaluation.** _Greedy return with a standard error, and the number of seeds._

**Limitations.** _At least four, each with how you would test whether it binds._

**Foreseeable misuse and reward-specification risk.** _What goes wrong if
someone optimises this reward harder than you did?_

**Responsible disclosure.** _If you found a failure mode, who would you tell,
when, and what would you say?_
        """
    )
    st.caption(
        "This section is part of the rubric, not an afterthought. A results "
        "table with no limitations section is an unfinished deliverable."
    )
