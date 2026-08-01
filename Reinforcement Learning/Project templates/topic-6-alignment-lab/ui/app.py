"""
ui/app.py — the presentation tier. The only thing a non-technical stakeholder
ever needs to open.

Four rules this file obeys, and yours must too:

  1. It contains NO policy code and NO training code. Every score comes back
     from the service tier through ui/service.py. Every chart is either a PNG
     the training tier wrote to `reports/` or an Altair chart over rows the
     service returned.
  2. It never issues SQL that changes state. Read-only views only, through the
     anon key.
  3. It degrades visibly. A paused database, a missing artifact or an unrun
     pipeline produces a clearly worded panel, never a stack trace and never a
     blank chart. Supabase free-tier projects pause after a week idle, so this
     will happen to you — probably the night before a deadline.
  4. IT DOES NOT GENERATE TEXT AND CANNOT. There is no text box that produces a
     completion, because there is no model in this process that could. The
     "Base vs Aligned" tab reads completions the training tier produced offline
     and persisted. That is the architecture note, rendered.

Run it:  streamlit run ui/app.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.config import get_settings          # noqa: E402
from ui import service                          # noqa: E402

settings = get_settings()
REPORTS = ROOT / "reports"

st.set_page_config(page_title="Alignment Lab", layout="wide")


def load_report(name: str) -> dict | None:
    """Read a JSON artifact the training tier wrote, or return None.

    None rather than an exception, and the caller renders a panel explaining
    which command produces the file. A fresh fork has no `reports/` directory
    and the app must still start and still be navigable.
    """
    path = REPORTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def show_figure(name: str, command: str) -> None:
    """Render a PNG from `reports/`, or say exactly how to produce it."""
    path = REPORTS / name
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.info(f"`reports/{name}` does not exist yet. Produce it with:\n\n```\n{command}\n```")


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
            "reward head loaded · data tier reachable"
        )
    else:
        st.warning(
            f"**Running in a degraded state.** {h.get('detail') or ''}\n\n"
            "Results below may be incomplete. This banner is deliberate: an "
            "empty chart and a broken database look identical otherwise."
        )
    return h


health = health_banner()

st.title("Alignment Lab")
st.caption(
    "A preference-scoring service. The reward model was trained offline from "
    "human comparisons and is served as a NumPy artifact behind a typed HTTP "
    "contract. **This service scores text; it does not generate any** — the "
    "completions below were produced offline and persisted."
)

TABS = st.tabs(
    ["Concepts", "Score It", "Base vs Aligned", "Reward Hacking",
     "Multi-Agent", "Run History", "Model Card"]
)


# ===========================================================================
with TABS[0]:
    st.header("Concepts")
    st.caption(
        "The Theoretical Brief, mirrored from the README. Written for the "
        "stakeholder in the product brief, not for the grader."
    )

    st.subheader("1. The three-stage RLHF pipeline")
    st.markdown(
        """
| Stage | What it fits | On what data | What comes out |
|---|---|---|---|
| **1. Supervised fine-tuning** | the policy, by maximum likelihood | demonstrations | `pi_ref`, the **reference** |
| **2. Reward modelling** | `r(x, y)`, by the Bradley–Terry loss | *comparisons* | a scalar proxy for human preference |
| **3. Policy optimisation** | the policy again, against `r` under a KL constraint | prompts | the aligned policy |

Stage 2 is the one that makes this a reinforcement learning problem rather than
a supervised one. Humans are unreliable at absolute scores and reliable at
comparisons, so the label is *which of these two is better* — and a comparison
is not a target you can regress onto. It is a preference over outcomes, which
is what a reward function is.

**Direct preference optimisation collapses stages 2 and 3 into one.** It never
fits an explicit reward model; it shows that the policy *is* one.
        """
    )

    st.subheader("2. The Bradley–Terry loss, derived")
    st.markdown(
        "The model says the probability a labeller prefers `y_c` to `y_r` is the "
        "logistic function of the reward difference:"
    )
    st.latex(r"P(y_c \succ y_r \mid x) \;=\; \sigma\big(r(x,y_c) - r(x,y_r)\big)")
    st.markdown("Take the negative log-likelihood of the observed comparisons:")
    st.latex(
        r"\mathcal{L}(\theta) \;=\; -\,\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}"
        r"\Big[\log \sigma\big(r_\theta(x,y_c) - r_\theta(x,y_r)\big)\Big]"
    )
    st.markdown(
        """
Three consequences, and every one of them shows up in this product:

* **Only differences are identified.** Add a constant to `r(x, ·)` for a fixed
  prompt and the loss does not move. So the number `POST /score` returns is
  meaningful in *comparison* and meaningless in isolation — which is why the
  "Score It" tab says so and why `POST /compare` exists.
* **The loss is unbounded below.** Nothing stops the model driving the margin
  to infinity on the training pairs. The same structural fact makes DPO at
  small `beta` degenerate; see the Reward Hacking tab.
* **The gradient vanishes on easy pairs.** `sigma'(m) -> 0` as the margin
  grows, so a confidently-correct comparison stops teaching. Held-out accuracy
  therefore plateaus long before the loss does.
        """
    )

    st.subheader("3. The KL-regularised objective and its closed form")
    st.latex(
        r"\max_{\pi}\;\; \mathbb{E}_{x,\,y\sim\pi}\big[r(x,y)\big] \;-\; "
        r"\beta\, \mathrm{KL}\big(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)"
    )
    st.markdown("Solving with a Lagrangian over the simplex gives the optimum in closed form:")
    st.latex(
        r"\pi^{*}(y\mid x) \;=\; \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,"
        r"\exp\!\Big(\tfrac{1}{\beta}\,r(x,y)\Big),\qquad "
        r"Z(x) = \sum_{y}\pi_{\text{ref}}(y\mid x)\,e^{r(x,y)/\beta}"
    )
    st.markdown(
        """
`Z(x)` sums over **every sequence the model could emit**, so it is intractable
for a language model — that intractability is the whole reason stage 3 was a
policy-gradient problem before DPO.

**This is not new mathematics.** It is the maximum-entropy objective from the
actor-critic topic with a different reference distribution. Using
`KL(pi || uniform) = -H(pi) + log|A|`, setting `pi_ref` uniform turns the
expression above into `E[r] + beta*H(pi)` up to a constant: the entropy bonus
is the special case, and `beta` here plays the role `alpha` played there.

Rearranging the closed form for the reward gives DPO:
        """
    )
    st.latex(
        r"r(x,y) \;=\; \beta \log\frac{\pi^{*}(y\mid x)}{\pi_{\text{ref}}(y\mid x)}"
        r"\;+\;\beta \log Z(x)"
    )
    st.markdown(
        "Substitute into the Bradley–Terry loss. The loss depends on `r` only "
        "through a **difference at the same prompt**, and `beta·log Z(x)` "
        "depends on the prompt alone — so it cancels exactly, and the "
        "intractable term disappears:"
    )
    st.latex(
        r"\mathcal{L}_{\text{DPO}} = -\log\sigma\!\left(\beta\left["
        r"\log\frac{\pi_\theta(y_c|x)}{\pi_{\text{ref}}(y_c|x)}"
        r"-\log\frac{\pi_\theta(y_r|x)}{\pi_{\text{ref}}(y_r|x)}\right]\right)"
    )
    st.info(
        "The reward model was not approximated away — it was **reparameterised**. "
        "`beta·log(pi/pi_ref)` is the *implicit reward*, and it is what the "
        "`implicit_reward_margin` and `implicit_reward_accuracy` columns in "
        "`alignment_runs` measure."
    )

    st.subheader("4. PPO-based RLHF vs DPO vs GRPO")
    st.markdown(
        """
| | PPO-based RLHF | DPO | GRPO |
|---|---|---|---|
| Models resident in training | **4** — policy, reference, reward model, critic | **2** — policy, reference | **3** — policy, reference, reward model |
| Needs an explicit reward model | yes | **no** | yes |
| Samples from the policy during training | yes (online) | **no** (offline) | yes (online) |
| Advantage estimate | learned critic `V(s)` | n/a | **group mean** over G sampled completions |
| Main failure mode | instability; four things to tune | off-policy: never sees its own outputs | reward-model quality is fully exposed |
| Library status (TRL 1.9.x) | `PPOTrainer` **experimental** | `DPOTrainer` stable | `GRPOTrainer` stable |

**GRPO's idea in one line.** Sample `G` completions for the same prompt, score
them all, and use the group's own mean as the baseline:
`A_i = (r_i - mean(r)) / std(r)`. The advantages sum to zero by construction —
that is what "subtract the mean" means — so no critic network is needed. What
is lost is per-token credit assignment: the critic could say *which part* of a
completion was good, and a group baseline can only say *which completion*.

**Why this product builds on DPO.** Two models instead of four is the
difference between fitting on a free GPU runtime and not, and `PPOTrainer`
being experimental in the current TRL is a practical reason on top of the
memory one.
        """
    )


# ===========================================================================
with TABS[1]:
    st.header("Score It")
    st.caption(
        "Paste any text. The deployed reward head scores it. Your text is "
        "**hashed, never stored** — the digest below is what goes to the audit log."
    )

    if not health.get("policy_artifact_loaded"):
        st.error(
            "No reward head is loaded, so there is nothing to score with. Run "
            "`python -m train.train --offline --quick`, then reload."
        )
    else:
        st.warning(
            "**The score is uncalibrated.** A Bradley–Terry reward model is "
            "identified only up to an additive constant, so a single number is "
            "meaningful in comparison and meaningless on its own. Use the "
            "comparison box below for any claim you intend to act on.",
            icon="!",
        )

        text = st.text_area(
            "Text to score",
            value="A specific, measured answer with a tested baseline and a stated caveat.",
            height=120,
            max_chars=8000,
            help="Capped at 8,000 characters — the same bound POST /score enforces.",
        )
        if st.button("Score", type="primary"):
            try:
                r = service.score(text)
            except service.ServiceError as exc:
                st.error(str(exc))
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Reward", f"{r['reward']:.3f}")
                c2.metric("Tokens seen", r["tokens"])
                c3.metric(
                    "Out-of-vocabulary", f"{r['oov_rate']:.0%}",
                    help=(
                        "Fraction of your tokens the head has never seen. A score "
                        "computed from mostly-OOV text is a score of an almost "
                        "empty vector."
                    ),
                )
                if r["oov_rate"] > 0.5:
                    st.warning(
                        f"{r['oov_rate']:.0%} of this text is outside the head's "
                        "vocabulary, so the score below is close to the model's "
                        "bias term and carries little information about your text."
                    )
                st.caption(
                    f"artifact `{r['policy_sha256'][:12]}…` · "
                    f"text digest `{r['text_sha256'][:16]}…` · "
                    f"{r['latency_ms']:.2f} ms"
                )

        st.divider()
        st.subheader("Compare two responses — the endpoint that matches the training objective")
        col_a, col_b = st.columns(2)
        a = col_a.text_area("Response A", value="Verified, reproducible, with a cited baseline.",
                            height=100, max_chars=8000)
        b = col_b.text_area("Response B", value="Obviously this is trivially guaranteed to work.",
                            height=100, max_chars=8000)
        if st.button("Compare"):
            try:
                r = service.compare(a, b)
            except service.ServiceError as exc:
                st.error(str(exc))
            else:
                winner = "A" if r["preferred"] == "a" else "B"
                st.success(f"**Response {winner}** is preferred.")
                m1, m2 = st.columns(2)
                m1.metric("Margin", f"{r['margin']:.3f}")
                m2.metric(
                    "P(a labeller agrees)", f"{r['probability']:.1%}",
                    help=(
                        "sigmoid(margin) — the Bradley–Terry probability, which is "
                        "the same functional form the loss was a likelihood of. "
                        "This is the number to quote."
                    ),
                )


# ===========================================================================
with TABS[2]:
    st.header("Base vs Aligned")
    st.caption(
        "Completions generated **offline, in the training tier**, persisted, and "
        "read back here. The service never generated any of this text."
    )

    try:
        data = service.completions(limit=500)
    except service.ServiceError as exc:
        st.error(str(exc))
        data = {"completions": [], "degraded": True}

    if data.get("degraded"):
        st.warning("The data tier did not answer, so this table is empty rather than complete.")
    elif not data["completions"]:
        st.info(
            "No completions have been persisted yet. Run "
            "`python -m train.train --offline --quick`, which generates from the "
            "reference policy and from each aligned variant and writes both to "
            "`completions`."
        )
    else:
        df = pd.DataFrame(data["completions"])
        prompts = df[["prompt_id", "prompt"]].drop_duplicates()
        choice = st.selectbox(
            "Prompt", prompts["prompt"].tolist(),
            help="Held-out prompts only — none of these appeared in the training split.",
        )
        pid = prompts.loc[prompts["prompt"] == choice, "prompt_id"].iloc[0]
        rows = df[df["prompt_id"] == pid].copy()

        available = sorted({b for b in rows["beta"].dropna().unique()})
        beta = st.select_slider(
            "beta (the KL coefficient)", options=available, value=available[0]
        ) if available else None
        st.caption(
            "Smaller beta = weaker KL constraint = the aligned policy is allowed "
            "further from the reference. See the Reward Hacking tab for what that buys."
        )

        left, right = st.columns(2)
        base = rows[rows["model_variant"] == "base"]
        aligned = rows[(rows["model_variant"] != "base") & (rows["beta"] == beta)]

        with left:
            st.subheader("Base (the reference policy)")
            if base.empty:
                st.info("No base completion for this prompt.")
            else:
                st.write(base.iloc[0]["text"])
                st.metric("Reward-model score", f"{base.iloc[0]['reward_score']:.3f}")
        with right:
            st.subheader(f"Aligned (beta = {beta})")
            if aligned.empty:
                st.info("No aligned completion at this beta for this prompt.")
            else:
                st.write(aligned.iloc[0]["text"])
                delta = None
                if not base.empty:
                    delta = aligned.iloc[0]["reward_score"] - base.iloc[0]["reward_score"]
                st.metric("Reward-model score",
                          f"{aligned.iloc[0]['reward_score']:.3f}",
                          delta=None if delta is None else f"{delta:+.3f}")

        st.divider()
        st.subheader("Aggregate over every held-out prompt")
        agg = (
            df.assign(variant=df.apply(
                lambda r: "base" if r["model_variant"] == "base" else f"beta={r['beta']:g}",
                axis=1))
            .groupby("variant")
            .agg(n=("text", "size"),
                 mean_reward=("reward_score", "mean"),
                 mean_true_quality=("true_quality", "mean"),
                 mean_tokens=("tokens", "mean"))
            .reset_index()
        )
        st.dataframe(agg, use_container_width=True)
        st.caption(
            "`mean_true_quality` is a signal the reward model never saw. Without "
            "a column like it, a rising `mean_reward` is not evidence of anything."
        )


# ===========================================================================
with TABS[3]:
    st.header("Reward Hacking")
    st.caption(
        "Mean reward-model score and KL from the reference, against beta, on the "
        "same axes — with the decoupling point annotated."
    )

    report = load_report("reward_hacking.json")
    show_figure("reward_hacking.png", "python -m train.reward_hacking --offline")

    if report:
        d = report.get("decoupling", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Decoupling at beta", f"{d.get('decoupling_beta') or 'not observed'}")
        c2.metric(
            "Proxy/target correlation, low pressure",
            "n/a" if d.get("corr_low_pressure_half") is None
            else f"{d['corr_low_pressure_half']:+.2f}",
        )
        c3.metric(
            "Proxy/target correlation, high pressure",
            "n/a" if d.get("corr_high_pressure_half") is None
            else f"{d['corr_high_pressure_half']:+.2f}",
        )
        st.dataframe(pd.DataFrame(report["rows"]), use_container_width=True)
        st.markdown(
            """
**How to read this.** Moving right along the x-axis loosens the KL constraint
and lets the policy travel further from the reference. The *proxy* — what the
reward model thinks — keeps rising. The *target* — the quality signal the
reward model never saw — stops rising and turns over. The two agree while the
policy stays near the distribution the reward model was fitted on, and come
apart once the optimiser leaves it.

**Why the KL penalty cannot prevent this, only delay it.** The penalty makes
distance from the reference expensive, so the policy reaches the region where
the proxy is wrong later — at a larger beta. It never makes the proxy correct
there. Push beta high enough to guarantee you never arrive and you also
guarantee you never improve, because `pi -> pi_ref` as `beta -> inf`.
            """
        )
    else:
        st.info(
            "No `reports/reward_hacking.json`. Run "
            "`python -m train.train --offline` (the full budget sweeps seven "
            "betas; `--quick` sweeps the three the build step requires and will "
            "not show the decoupling)."
        )


# ===========================================================================
with TABS[4]:
    st.header("Multi-Agent")
    st.caption(
        "What happens when the environment is another learner. Four panels: the "
        "iterated prisoner's dilemma, the non-stationarity comparison, a "
        "cooperative task, and the matching-pennies phase plot."
    )
    show_figure("multiagent.png", "python -m train.multiagent --offline")

    ma = load_report("multiagent.json")
    if ma:
        cols = st.columns(3)
        ipd = ma.get("ipd", {})
        cols[0].metric(
            "IPD reward per step (A)", f"{ipd.get('final_reward_a', float('nan')):.2f}",
            help="Mutual defection pays 1.0; mutual cooperation pays 3.0.",
        )
        cols[1].metric("IPD cooperation rate",
                       f"{ipd.get('final_cooperation_rate', float('nan')):.0%}")
        mp = ma.get("matching_pennies", {})
        cols[2].metric("Matching pennies, time-averaged P(tails)",
                       f"{mp.get('time_average_a', float('nan')):.3f}",
                       help="The unique Nash equilibrium is 0.5, and the learners orbit it.")

        ns = ma.get("nonstationarity", {}).get("verdict", {})
        if ns:
            st.subheader("Non-stationarity, as three ratios")
            st.dataframe(
                pd.DataFrame(
                    [{"measurement": k.replace("_", " "),
                      "learning opponent / frozen opponent": round(v, 2)}
                     for k, v in ns.items()]
                ),
                use_container_width=True,
            )
            st.caption(
                "Same agent, same algorithm, same hyperparameters. The only "
                "change is whether the opponent is learning. Every ratio above "
                "1 is the stationarity assumption being violated."
            )

    st.markdown(
        """
**Independent learning vs centralised training with decentralised execution.**
*(200–300 words in the README; summarise here.)*

Independent learners treat the other agents as part of the environment. That
is what makes the transition function time-dependent, and with it go
Q-learning's convergence guarantees — the target each agent regresses onto
moves whenever anyone else updates. CTDE keeps execution decentralised (each
agent still acts on its own observation, so deployment is unchanged) but trains
a **centralised critic** that sees the joint state and the joint action. The
critic's input therefore stops changing when the other agents learn, and the
non-stationarity is absorbed into a component that is thrown away before
deployment.

**Where the independent approach should be expected to fail:** any task with a
single shared reward and many agents, where an agent cannot tell its own
contribution from the others'. Our cooperative gridworld gives both agents an
identical number every step; with two agents the correlation between an agent's
own behaviour and the reward is still strong enough to learn from, and with ten
it is not.
        """
    )


# ===========================================================================
with TABS[5]:
    st.header("Run History")
    st.caption("Read-only. This tab issues no writes and holds no service-role key.")

    st.subheader("Alignment runs — one row per beta")
    try:
        runs = service.alignment_runs(100)
        if runs["runs"]:
            st.dataframe(pd.DataFrame(runs["runs"]), use_container_width=True)
            st.caption(
                "`implicit_reward_margin` is `beta·log(pi/pi_ref)` differenced over "
                "a pair. It is scaled by beta, so margins at different betas are "
                "**not** directly comparable — do not plot them together as though "
                "they were."
            )
        else:
            st.info("No alignment runs logged yet.")
    except service.ServiceError as exc:
        st.error(str(exc))

    st.subheader("Training runs")
    try:
        data = service.runs(100)
        if data["runs"]:
            st.dataframe(pd.DataFrame(data["runs"]), use_container_width=True, height=260)
        else:
            st.info("Nothing logged yet.")
    except service.ServiceError as exc:
        st.error(str(exc))

    st.subheader("Registered artifacts")
    try:
        st.dataframe(pd.DataFrame(service.policies()["policies"]), use_container_width=True)
    except service.ServiceError as exc:
        st.error(str(exc))

    heads = load_report("reward_heads.json")
    if heads:
        st.subheader("The two reward heads")
        st.dataframe(
            pd.DataFrame([
                {
                    "head": name,
                    "feature dim": heads[name]["feature_dim"],
                    "held-out accuracy": round(heads[name]["held_out_accuracy"], 4),
                    "+/- SE": round(heads[name]["accuracy_stderr"], 4),
                    "reward-vs-length r": round(heads[name]["length_bias"]["pearson_r"], 4),
                    "length decodability R^2": round(heads[name]["length_decodability_r2"], 4),
                    "length-matched accuracy": round(
                        heads[name]["length_matched"]["accuracy"], 4),
                    "artifact KB": round(heads[name]["artifact"]["bytes"] / 1024, 1),
                    "deployed": name == heads.get("deployed"),
                }
                for name in ("tfidf", "embedding")
            ]),
            use_container_width=True,
        )
        show_figure("reward_margins.png", "python -m train.reward_model --offline")
        show_figure("length_bias.png", "python -m train.reward_model --offline")


# ===========================================================================
with TABS[6]:
    st.header("Model card")
    st.markdown(
        """
**What this service does.** It assigns a scalar preference score to a piece of
text, using a reward model fitted from pairwise human comparisons, and it
serves a library of base and aligned completions that were generated offline.

**What it does not do.** _It does not generate text._ There is no language
model in the serving process and no `/generate` endpoint. It does not moderate,
detect policy violations, or judge factual accuracy — it predicts which of two
responses an annotator from the training distribution would have preferred, and
nothing else.

**The score is uncalibrated.** A Bradley–Terry reward is identified only up to
an additive constant per prompt. Comparisons are meaningful; absolute values
are not, and a threshold set on the raw score will not transfer to a
re-trained artifact.

**Deployed head.** _[name, vocabulary size, artifact size, SHA-256]._ The
stronger embedding head is registered and **not** deployed: computing its input
requires a transformer, which does not fit the serving memory budget. State
what that cost you in held-out accuracy.

**Training data.** _[dataset, number of comparisons, split, and — importantly —
who produced the preferences]._ If you used the offline fallback for any number
you quote, say so in the sentence that quotes it.

**Evaluation.** _[held-out pairwise accuracy against the 50% baseline, with a
standard error; the length-bias correlation; the length-matched accuracy]._

**Privacy.** Text submitted to `POST /score` is **not stored**. A SHA-256
digest is written to `audit_log` so that repeat submissions can be counted and
a past score attributed to a specific artifact. That is pseudonymisation, not
anonymisation: anyone holding a candidate text can confirm it was submitted by
hashing it themselves, and short or guessable inputs are effectively not
protected at all.

**Limitations.** _At least four, each with how you would test whether it binds._

**Foreseeable misuse and reward-specification risk.** This head is a proxy.
The Reward Hacking tab shows the point at which optimising it stopped improving
the thing it stood in for; anyone using this score as an optimisation target
rather than as a diagnostic should read that tab first.

**Responsible disclosure.** _If you found a failure mode, who would you tell,
when, and what would you say?_
        """
    )
    st.caption(
        "This section is part of the rubric, not an afterthought. A results "
        "table with no limitations section is an unfinished deliverable."
    )
