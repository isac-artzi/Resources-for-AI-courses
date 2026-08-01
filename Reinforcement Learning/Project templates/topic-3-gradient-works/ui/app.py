"""
ui/app.py — the presentation tier. The only thing a non-technical stakeholder
ever needs to open.

Three rules this file obeys, and yours must too:

  1. It contains NO policy code and NO training code. Every decision and every
     number comes back from the service tier through ui/service.py.
  2. It never issues SQL that changes state. Read-only views only, through the
     anon key.
  3. It degrades visibly. A paused database or a missing artifact produces a
     clearly worded panel, never a stack trace. Supabase free-tier projects
     pause after a week idle, so this will happen to you — probably the night
     before a deadline.

The Topic 3 tabs are Concepts, Ablation, **Gradient Variance**, Play, Run
History and Model Card. The third one is the point of the product: the client
asked why policy gradients are noisy and what fixes it, and that chart is the
answer. Everything else on this page supports it.

Run it:  streamlit run ui/app.py
"""

from __future__ import annotations

import pathlib
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from shared.config import get_settings          # noqa: E402
from ui import service                          # noqa: E402

settings = get_settings()

st.set_page_config(page_title="Gradient Works", layout="wide")

# The four cells of the 2x2, in a fixed order with fixed colours. Fixed so that
# "the blue line" means the same arm on every chart on this page and in every
# screenshot you paste into the report — a legend that reorders itself between
# tabs is how a reader misreads your result.
ARMS = ["vpg", "vpg+baseline", "vpg+is", "vpg+baseline+is"]
ARM_LABELS = {
    "vpg": "no baseline, on-policy",
    "vpg+baseline": "baseline, on-policy",
    "vpg+is": "no baseline, importance sampled",
    "vpg+baseline+is": "baseline + importance sampled",
}
ARM_SCALE = alt.Scale(domain=ARMS, range=["#9e9e9e", "#1f77b4", "#ff7f0e", "#2ca02c"])


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


# ---------------------------------------------------------------------------
# Data loading. Cached, because the Ablation tab needs one call per run, and
# twelve runs times two endpoints is twenty-four round trips on every widget
# interaction otherwise — which on the free tier is felt, not measured.
# ---------------------------------------------------------------------------


def _arm_of(row) -> str:
    """Which cell of the 2x2 a run belongs to, read from its hyperparameters.

    From `experiments.hyperparameters`, not from a string match on the algorithm
    name. The hyperparameters are what the run actually used; the name is a
    label someone typed, and the two disagree the first time an arm is renamed.
    """
    hp = row.get("hyperparameters") or {}
    if "use_baseline" in hp:
        return "vpg" + ("+baseline" if hp.get("use_baseline") else "") + (
            "+is" if hp.get("use_importance_sampling") else ""
        )
    return str(row.get("algorithm", "unknown"))


@st.cache_data(ttl=120, show_spinner=False)
def load_runs(limit: int = 200) -> pd.DataFrame:
    df = pd.DataFrame(service.runs(limit)["runs"])
    if df.empty:
        return df
    df["arm"] = df.apply(_arm_of, axis=1)
    return df


@st.cache_data(ttl=120, show_spinner=False)
def load_episodes(experiment_ids: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for eid in experiment_ids:
        try:
            body = service.episodes(eid)
        except service.ServiceError:
            # One unreachable run must not empty the whole chart. The seed count
            # shown next to the curve will be short, which is a visible failure.
            continue
        if body["episodes"]:
            f = pd.DataFrame(body["episodes"])
            f["experiment_id"] = eid
            frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def load_gradient_stats(experiment_ids: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for eid in experiment_ids:
        try:
            body = service.gradient_stats(eid)
        except service.ServiceError:
            continue
        if body["stats"]:
            f = pd.DataFrame(body["stats"])
            f["experiment_id"] = eid
            frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def seed_band(
    df: pd.DataFrame,
    x: str,
    y: str,
    bin_width: int,
    x_title: str,
    y_title: str,
    log_y: bool = False,
):
    """Mean across seeds, with a ± one standard error band.

    The band is across SEEDS, which is the only error bar that supports a claim
    about a configuration. Shading the spread across episodes within one seed
    would give a wider, prettier band that says nothing about whether the arm
    would work again.

    Binning first is not cosmetic either: a CartPole return series is a spiky
    step function and an unbinned mean of three of them is unreadable. The bin
    width is written on the axis so the reader knows what they are looking at.
    """
    d = df.copy()
    d["_bin"] = (d[x] // bin_width) * bin_width
    per_seed = d.groupby(["arm", "seed", "_bin"], as_index=False)[y].mean()
    agg = per_seed.groupby(["arm", "_bin"], as_index=False).agg(
        mean=(y, "mean"), sd=(y, "std"), n=(y, "count")
    )
    agg["sd"] = agg["sd"].fillna(0.0)
    agg["se"] = agg["sd"] / np.sqrt(agg["n"].clip(lower=1))
    agg["lo"] = agg["mean"] - agg["se"]
    agg["hi"] = agg["mean"] + agg["se"]
    if log_y:
        # A log axis cannot render a non-positive lower edge. Clipping is honest
        # here because the quantity plotted is a variance — non-negative by
        # construction — so a negative band edge is an artefact of the normal
        # approximation rather than a measurement.
        positive = agg["mean"][agg["mean"] > 0]
        floor = max(float(positive.min()) * 1e-3, 1e-12) if len(positive) else 1e-12
        agg["lo"] = agg["lo"].clip(lower=floor)

    y_scale = alt.Scale(type="log") if log_y else alt.Scale(zero=False)
    base = alt.Chart(agg)
    band = base.mark_area(opacity=0.22).encode(
        x=alt.X("_bin:Q", title=x_title),
        y=alt.Y("lo:Q", title=y_title, scale=y_scale),
        y2="hi:Q",
        color=alt.Color("arm:N", scale=ARM_SCALE, title="arm"),
    )
    line = base.mark_line(strokeWidth=2).encode(
        x="_bin:Q",
        y=alt.Y("mean:Q", scale=y_scale),
        color=alt.Color("arm:N", scale=ARM_SCALE),
        tooltip=["arm:N", "_bin:Q", "mean:Q", "se:Q", "n:Q"],
    )
    return (band + line).properties(height=380).interactive()


health = health_banner()

st.title("Gradient Works")
st.caption(
    "A policy gradient service that reports not only what its agent scored but "
    "the variance of the gradient estimate that produced it — because "
    "\"it is noisy\" is an adjective and this is the number."
)

TABS = st.tabs(
    ["Concepts", "Ablation", "Gradient Variance", "Play", "Run History", "Model Card"]
)


# ===========================================================================
with TABS[0]:
    st.header("Concepts")
    st.caption(
        "The Theoretical Brief, mirrored from the README. Written for the "
        "colleague who has to approve this for a control project, not for a grader."
    )

    st.subheader("1. Why the policy stops being a table")
    st.markdown(
        """
In the earlier topics a policy was a lookup: one row per state, and learning
meant writing numbers into cells. CartPole's observation is four real numbers —
cart position, cart velocity, pole angle, pole angular velocity — so there are
no cells. Discretising into bins is possible, and gets exponentially worse with
each dimension added; that is the practical face of the curse of dimensionality.

So we **parameterise** the policy instead: a small network maps the observation
to two logits, a softmax turns them into a distribution, and learning means
moving the weights. Nearby states now share parameters, so experience in one
state informs behaviour in states never visited — which is the whole reason
function approximation is worth its instability.
        """
    )
    st.latex(r"\pi_\theta(a \mid s) = \operatorname{softmax}\big(f_\theta(s)\big)_a")

    st.subheader("2. The policy gradient theorem, derived")
    st.markdown(
        "We want to maximise the expected return of a trajectory $\\tau$ under "
        "the policy:"
    )
    st.latex(r"J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[R(\tau)\big]")
    st.markdown(
        "The obstacle is that $\\theta$ appears in the *distribution* being "
        "averaged over, not in the thing being averaged. The **likelihood-ratio "
        "trick** moves it back into the integrand:"
    )
    st.latex(r"\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\,\nabla_\theta \log p_\theta(\tau)")
    st.latex(
        r"\nabla_\theta J(\theta) = \int \nabla_\theta p_\theta(\tau)\, R(\tau)\, d\tau"
        r"\; = \; \mathbb{E}_{\tau}\big[\nabla_\theta \log p_\theta(\tau)\, R(\tau)\big]"
    )
    st.markdown(
        "Now expand the trajectory likelihood. It factorises into the "
        "environment's dynamics and the policy — and the dynamics do not depend "
        "on $\\theta$, so they differentiate away:"
    )
    st.latex(
        r"\log p_\theta(\tau) = \log \rho(s_0) + \sum_t \Big[\log P(s_{t+1}\mid s_t,a_t)"
        r" + \log \pi_\theta(a_t \mid s_t)\Big]"
    )
    st.latex(
        r"\nabla_\theta J(\theta) = \mathbb{E}\Big[\textstyle\sum_t \nabla_\theta "
        r"\log \pi_\theta(a_t\mid s_t)\; G_t\Big], \qquad "
        r"G_t = \textstyle\sum_{k\ge t}\gamma^{\,k-t} r_k"
    )
    st.info(
        "**This is why the method is model-free.** The transition model vanished "
        "from the gradient, so we never have to know it — we only have to be able "
        "to sample from it. Everything after this point in the topic is about "
        "reducing the variance of that expectation, never about its correctness."
    )

    st.subheader("3. The relationship to maximum likelihood estimation")
    st.markdown(
        "Put the two gradients side by side. A supervised classifier trained by "
        "maximum likelihood on labelled pairs $(s_i, a_i)$ ascends"
    )
    st.latex(r"\nabla_\theta \sum_i \log \pi_\theta(a_i \mid s_i)")
    st.markdown("and the policy gradient ascends")
    st.latex(r"\nabla_\theta \sum_t \log \pi_\theta(a_t \mid s_t)\; G_t .")
    st.markdown(
        """
They are the same expression with one difference: **the return is a weight on
the log-likelihood term.** Policy gradient learning is maximum likelihood on the
agent's own behaviour, with each action's example weighted by how well things
went afterwards. An action followed by a large return is treated as a strongly
labelled training example; one followed by a poor return is pushed down.

Three consequences worth being able to state to a stakeholder:

* **There is no ground-truth label.** MLE is told the right answer; the policy
  gradient infers a soft one from the return, and the return is noisy. That
  noise is exactly the variance this product measures.
* **The data distribution moves.** MLE has a fixed training set; here the policy
  generates the data it will next be trained on, so a policy that stops
  exploring stops receiving informative examples.
* **When you have expert demonstrations, use MLE.** Behaviour cloning is
  supervised learning: it converges faster and is far easier to debug.
  Reinforcement learning earns its cost when nobody can produce the labels —
  when you can only score outcomes, not name correct actions.
        """
    )

    st.subheader("4. The baseline, and why it reduces variance for free")
    st.markdown("Subtract any function of the state — not of the action — from the return:")
    st.latex(
        r"\nabla_\theta J = \mathbb{E}\Big[\textstyle\sum_t \nabla_\theta \log "
        r"\pi_\theta(a_t\mid s_t)\,\big(G_t - b(s_t)\big)\Big]"
    )
    st.markdown("The subtracted term contributes **exactly zero** in expectation:")
    st.latex(
        r"\mathbb{E}_{a\sim\pi}\big[\nabla_\theta \log \pi_\theta(a\mid s)\, b(s)\big]"
        r" = b(s) \sum_a \pi_\theta(a\mid s)\,\nabla_\theta \log \pi_\theta(a \mid s)"
    )
    st.latex(
        r"= b(s)\sum_a \nabla_\theta \pi_\theta(a\mid s)"
        r" = b(s)\,\nabla_\theta \sum_a \pi_\theta(a\mid s) = b(s)\,\nabla_\theta 1 = 0"
    )
    st.markdown(
        "So the estimator stays **unbiased for any such $b$** — and its variance "
        "changes. Writing $g = \\nabla_\\theta \\log \\pi_\\theta(a\\mid s)$, the "
        "per-sample estimator's variance is"
    )
    st.latex(
        r"\operatorname{Var}\big[g\,(G-b)\big] = \mathbb{E}\big[g^2 (G-b)^2\big]"
        r" - \big(\mathbb{E}[g\,G]\big)^2"
    )
    st.markdown(
        """
Only the first term depends on $b$; it is a quadratic in $b$, minimised at
$b^\\star = \\mathbb{E}[g^2 G]\\,/\\,\\mathbb{E}[g^2]$ — a magnitude-weighted
average return, well approximated by $V(s)$. Hence a value network rather than a
constant.

The intuition is shorter than the algebra. Without a baseline, **every** action
taken in a good state has its probability increased, merely for having been
taken somewhere good. With one, only actions that did better than their own
state's average are reinforced. The advantage $A = G - V(s)$ asks the right
question: *was this action better than what I usually get from here?*

This claim is falsifiable, and this repository falsifies it — see the Gradient
Variance tab, and `tests/test_baseline_reduces_variance.py`, which fails the
build if it stops being true.
        """
    )

    st.subheader("5. Importance sampling: what it buys and what it costs")
    st.markdown(
        "Data collected under an old policy can estimate an expectation under a "
        "new one:"
    )
    st.latex(
        r"\mathbb{E}_{a\sim\pi_{\text{new}}}[f] = \mathbb{E}_{a\sim\pi_{\text{old}}}"
        r"\left[\frac{\pi_{\text{new}}(a\mid s)}{\pi_{\text{old}}(a\mid s)}\, f\right]"
    )
    st.markdown(
        """
**The benefit.** Vanilla policy gradient discards every batch after one update —
it is on-policy, and on-policy is sample-inefficient. Reweighting lets a batch be
reused, which matters exactly when samples are expensive: a physical robot, a
metered simulator, a human rater.

**The costs, all three of which you should be able to name.**

* *Variance.* The estimator is unbiased, but its variance grows with the
  mismatch between the policies. One state where the old policy assigned
  probability 0.01 and the new one assigns 0.9 contributes a weight of 90 and
  swamps the batch. The effective sample size $(\\sum w)^2 / \\sum w^2$ is the
  honest measure of how much data you actually have, and this product logs it.
* *The horizon.* The exact trajectory weight is the product of the per-step
  ratios. A product of 200 numbers near 1 is numerically either 0 or enormous,
  and its variance grows exponentially in the episode length. Every practical
  method, this one included, uses per-step ratios instead — which is **biased**,
  and buys a variance that does not explode.
* *Support.* The correction is valid only where the old policy had non-zero
  probability. An action the old policy never took cannot be evaluated at any
  weight, however large.

**Where it fails, and what fixed it.** As the policies diverge the weight
distribution becomes heavy-tailed: most weights near zero, a few enormous. PPO's
answer is to clip the ratio, which flattens the surrogate — and therefore zeroes
the gradient — outside a trust region, so an update that would move the policy
far beyond where the data is informative simply does not happen. The truncation
in `train/vpg.off_policy_update` is the one-sided, looser ancestor of that idea.
Watch the histogram in the Gradient Variance tab: when it piles up against the
ceiling, the batch is no longer worth reusing.
        """
    )


# ===========================================================================
with TABS[1]:
    st.header("Ablation — the 2×2")
    st.markdown(
        "Baseline on/off crossed with importance sampling on/off, at least three "
        "seeds per arm. Bands are ± one standard error **across seeds**, which is "
        "the only error bar that supports a claim about a configuration."
    )

    runs_df = load_runs(200)
    known = runs_df[runs_df["arm"].isin(ARMS)] if not runs_df.empty else pd.DataFrame()

    if known.empty:
        st.info(
            "No ablation runs logged yet. Run "
            "`python -m train.ablation --episodes 1000 --seeds 3` — twelve runs, "
            "all of them rows in `experiments`."
        )
    else:
        counts = known.groupby("arm")["seed"].nunique()
        thin = [a for a in ARMS if 0 < counts.get(a, 0) < 3]
        missing = [a for a in ARMS if counts.get(a, 0) == 0]
        if missing:
            st.error(
                "The 2×2 is incomplete — no runs for "
                + ", ".join(f"`{a}`" for a in missing)
                + ". A three-quarters ablation cannot separate the two factors."
            )
        if thin:
            st.warning(
                "Fewer than three seeds in: "
                + ", ".join(f"`{a}`" for a in thin)
                + ". A policy gradient on CartPole can reach 500 or plateau at 40 "
                "with nothing changed but the initialisation, so below three seeds "
                "you are measuring the seed rather than the arm."
            )

        bin_width = st.slider(
            "Smoothing window (episodes per point)", 5, 100, 20, step=5,
            help="Written on the axis so the reader knows what they are looking at.",
        )
        eps = load_episodes(tuple(known["experiment_id"]))
        if eps.empty:
            st.warning("Runs exist but no episode rows came back. Is the data tier awake?")
        else:
            eps = eps.join(known.set_index("experiment_id")[["arm", "seed"]], on="experiment_id")
            # `return` is a Python keyword, so the Pydantic field is `return_`
            # with an alias. Renaming once here keeps the rest of the tab readable.
            eps = eps.rename(columns={"return": "return_"})
            st.altair_chart(
                seed_band(eps, "episode_index", "return_", bin_width,
                          f"training episode (mean of {bin_width})", "return"),
                use_container_width=True,
            )

            st.subheader("Final comparison")
            cutoff = eps["episode_index"].max() - 100
            tail = eps[eps["episode_index"] >= cutoff]
            per_seed = tail.groupby(["arm", "seed"], as_index=False)["return_"].mean()
            table = per_seed.groupby("arm", as_index=False).agg(
                seeds=("seed", "nunique"), mean_return=("return_", "mean"), sd=("return_", "std")
            )
            table["sd"] = table["sd"].fillna(0.0)
            table["stderr_across_seeds"] = table["sd"] / np.sqrt(table["seeds"].clip(lower=1))
            table["arm"] = table["arm"].map(lambda a: f"{a} — {ARM_LABELS.get(a, '')}")
            st.dataframe(
                table.drop(columns=["sd"]).round(2), use_container_width=True, hide_index=True
            )
            st.caption(
                "Mean training return over the last 100 episodes. Note that the "
                "importance-sampling arms take more gradient steps per episode, so "
                "this table compares SAMPLE efficiency and not compute — say which "
                "one you are claiming."
            )


# ===========================================================================
with TABS[2]:
    st.header("Gradient Variance")
    st.markdown(
        "**The headline chart of this product.** The client's question was not "
        "*does it work*; it was *why is it noisy, and what fixes it*. This is the "
        "answer: the trace of the covariance of the per-episode gradient "
        "estimates, against update index, for all four arms."
    )

    runs_df = load_runs(200)
    known = runs_df[runs_df["arm"].isin(ARMS)] if not runs_df.empty else pd.DataFrame()
    if known.empty:
        st.info("Nothing logged yet — run the ablation.")
    else:
        gs = load_gradient_stats(tuple(known["experiment_id"]))
        if gs.empty:
            st.warning(
                "No `gradient_stats` rows. If `episodes` has rows and this does not, "
                "the training loop is logging returns but not gradients — check that "
                "`store.insert_gradient_stats` is actually reached."
            )
        else:
            gs = gs.join(known.set_index("experiment_id")[["arm", "seed"]], on="experiment_id")

            on_policy_only = st.checkbox(
                "On-policy updates only", value=True,
                help=(
                    "Off-policy updates reuse an older batch through a different "
                    "estimator, so mixing them into one series compares two things "
                    "at once. Untick to see both."
                ),
            )
            plot = gs[~gs["off_policy"]] if on_policy_only else gs
            step = st.slider("Smoothing window (updates per point)", 1, 20, 4)
            st.altair_chart(
                seed_band(plot, "update_index", "gradient_variance", step,
                          f"policy update (mean of {step})",
                          "trace of gradient covariance (log scale)", log_y=True),
                use_container_width=True,
            )
            st.warning(
                "**Read this chart with one caveat in front of you.** In CartPole "
                "the return *is* the episode length, so an arm that learns faster "
                "sums more terms into each gradient and shows higher raw variance "
                "for a reason that has nothing to do with its estimator. Compare "
                "early, before the arms separate in return — or quote the controlled "
                "measurement from `train.vpg.compare_baseline_variance`, where the "
                "trajectories and the parameters are held fixed and only the "
                "advantage changes."
            )

            st.subheader("Policy entropy")
            st.caption(
                "Exploration, for a softmax policy, is the entropy of its own "
                "distribution — there is no epsilon here, which is why "
                "`episodes.epsilon` is null on this topic's runs. Bounded above by "
                "ln 2 ≈ 0.693 for two actions. A run whose entropy collapses early "
                "has stopped exploring, and its flat learning curve now has a cause "
                "you can point at."
            )
            st.altair_chart(
                seed_band(plot, "update_index", "policy_entropy", step,
                          "policy update", "entropy (nats)"),
                use_container_width=True,
            )

            st.subheader("Importance weights")
            off = gs[gs["off_policy"]]
            if off.empty:
                st.info(
                    "No off-policy updates logged. Run an arm with "
                    "`--importance-sampling` to populate this section."
                )
            else:
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.metric(
                        "Median effective sample size",
                        f"{off['is_weight_ess'].median():.2f}",
                        help=(
                            "(Σw)² / (n·Σw²), as a fraction of n. 1.0 means the two "
                            "policies agree; 0.05 means twenty weighted samples are "
                            "doing the work of one, however healthy the mean looks."
                        ),
                    )
                    st.metric("Largest weight observed", f"{off['is_weight_max'].max():.2f}")
                    st.metric("Median 95th percentile", f"{off['is_weight_p95'].median():.2f}")
                with c2:
                    hists = [h for h in off["is_weight_histogram"] if isinstance(h, dict)]
                    if hists:
                        # Summed across updates. Stored as a histogram rather than
                        # as raw weights because one run at the report budget
                        # produces a few hundred thousand of them and the free tier
                        # has 500 MB.
                        counts = np.sum([h["counts"] for h in hists], axis=0)
                        edges = np.asarray(hists[0]["edges"], dtype=float)
                        hdf = pd.DataFrame(
                            {"weight": (edges[:-1] + edges[1:]) / 2.0, "steps": counts}
                        )
                        st.altair_chart(
                            alt.Chart(hdf).mark_bar().encode(
                                x=alt.X("weight:Q",
                                        title="importance weight  π_new / π_old"),
                                y=alt.Y("steps:Q", scale=alt.Scale(type="symlog")),
                            ).properties(height=260),
                            use_container_width=True,
                        )
                st.caption(
                    "Mass piled against the right-hand edge means the ratio is "
                    "hitting the truncation ceiling: the policies have drifted too "
                    "far for this batch to be worth reusing. The honest fix is to "
                    "reuse fewer batches, not to raise the ceiling."
                )


# ===========================================================================
with TABS[3]:
    st.header("Play — the deployed policy, live")

    if not health.get("policy_artifact_loaded"):
        st.error(
            "No policy artifact is loaded, so there is nothing to play against. "
            "Run `python -m train.train`, which exports one, then reload."
        )
    else:
        st.caption(
            "Everything on this tab is served by the NumPy forward pass in "
            "`api/forward.py` — the same ~20 KB archive and the same three matrix "
            "multiplies that run in production. No PyTorch is loaded in this "
            "process; `GET /version` reports that honestly."
        )
        try:
            available = [p["name"] for p in service.policies()["policies"]]
        except service.ServiceError as exc:
            available = []
            st.error(str(exc))

        col_a, col_b, col_c = st.columns([2, 1, 1])
        policy_name = col_a.selectbox("Policy", available or ["default"])
        episodes_n = col_b.slider("Evaluation episodes", 1, 100, 20)
        seed = col_c.number_input("Seed", value=0, step=1,
                                  help="Name the seed or the result is not evidence.")

        if st.button("Run a greedy evaluation", type="primary"):
            try:
                r = service.rollout(policy_name, episodes_n, int(seed))
            except service.ServiceError as exc:
                st.error(str(exc))
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("Mean return", f"{r['mean_return']:.1f}")
                m2.metric("Standard error", f"{r['stderr_return']:.2f}",
                          help="s / sqrt(n). Quote this alongside the mean, never the mean alone.")
                m3.metric("Mean episode length", f"{r['mean_length']:.1f}")
                st.caption(
                    f"artifact `{r['policy_sha256'][:12]}…` · {r['episodes']} episodes · "
                    f"seed {r['seed']} · **greedy**, which is a different question from "
                    "the training curve on the Ablation tab"
                )
                if r["trajectory"]:
                    st.subheader("One full trajectory")
                    traj = pd.DataFrame(r["trajectory"])
                    states = pd.DataFrame(
                        traj["state"].tolist(),
                        columns=["cart position", "cart velocity", "pole angle",
                                 "pole angular velocity"],
                    )
                    st.line_chart(states[["pole angle", "cart position"]])
                    st.caption(
                        "A good policy holds the pole angle in a narrow band around "
                        "zero while letting the cart drift. CartPole rewards survival, "
                        "not staying centred, and an agent that has worked that out is "
                        "doing the right thing rather than cheating."
                    )
                    st.dataframe(traj, use_container_width=True, height=240)
                st.bar_chart(pd.DataFrame({"return": r["returns"]}))

        st.divider()
        st.subheader("Ask the policy about one state")
        st.caption(
            "Move a slider and watch the decision change. The value shown is the "
            "probability the policy assigns to the action it chose — a policy "
            "gradient learns a *stochastic* policy, and those probabilities are "
            "what it learned."
        )
        c = st.columns(4)
        state = [
            c[0].slider("cart position", -2.4, 2.4, 0.0, 0.05),
            c[1].slider("cart velocity", -3.0, 3.0, 0.0, 0.05),
            c[2].slider("pole angle (rad)", -0.21, 0.21, 0.03, 0.005),
            c[3].slider("pole angular velocity", -3.0, 3.0, 0.0, 0.05),
        ]
        try:
            resp = service.act(state, policy_name, deterministic=True)
        except service.ServiceError as exc:
            st.error(str(exc))
        else:
            left, right = st.columns([1, 2])
            left.metric("Greedy action", "push left" if resp["action"] == 0 else "push right")
            left.caption(f"{resp['latency_ms']:.2f} ms · `{resp['policy_sha256'][:12]}…`")
            confidence = float(resp.get("value_estimate") or 0.0)
            right.progress(
                min(max(confidence, 0.0), 1.0),
                text=f"probability of the chosen action: {confidence:.3f}",
            )
            if confidence > 0.99:
                right.caption(
                    "Near-deterministic in this state. That is a finding, not a "
                    "problem — but check the entropy curve: a policy that went "
                    "deterministic everywhere and early stopped exploring."
                )


# ===========================================================================
with TABS[4]:
    st.header("Run history")
    st.caption("Read-only. This tab issues no writes and holds no service-role key.")
    try:
        data = service.runs(100)
        if data.get("degraded"):
            st.warning(
                "The data tier did not answer, so this table is empty rather than "
                "complete. A free-tier project pauses after a week idle."
            )
        elif data["runs"]:
            df = pd.DataFrame(data["runs"])
            st.dataframe(df, use_container_width=True, height=420)
            st.caption(f"{len(df)} runs · {int(df['episodes_logged'].sum())} episodes logged")
        else:
            st.info("Nothing logged yet.")
    except service.ServiceError as exc:
        st.error(str(exc))

    st.subheader("Registered artifacts")
    st.caption(
        "Size and checksum for every `.npz` the service can load. The checksum is "
        "what `/act` returns and `audit_log` stores, so six weeks from now "
        "\"which artifact produced this action\" is a join rather than a guess."
    )
    try:
        st.dataframe(pd.DataFrame(service.policies()["policies"]), use_container_width=True)
    except service.ServiceError as exc:
        st.error(str(exc))


# ===========================================================================
with TABS[5]:
    st.header("Model card")
    st.markdown(
        """
**What this agent does.** _One paragraph, in the stakeholder's language. What
does it control, under what conditions, and what does a good day look like?_

**What it does not do.** _State the boundary explicitly. This policy was trained
on CartPole-v1 and has never seen a pole of a different length, a different
gravity, or an observation with sensor noise._

**Architecture and artifact.** _4 → 64 → 64 → 2, ReLU hidden, softmax output.
Exported as float32 to a NumPy `.npz`; give the size and checksum here and keep
them in sync with the Run History tab._

**Training data and environment.** _Environment id, episodes per seed, number of
seeds, and the exact command. A result whose command is not written down is not
reproducible._

**Evaluation.** _Greedy return with a standard error across seeds, from
`evaluations`. Not the training curve — say which is which._

**Gradient variance.** _Topic-specific and required here: the measured variance
with and without the baseline on a fixed seed, from the controlled comparison,
and the ratio. This is the number the client actually asked for._

**Limitations.** _At least four, each with how you would test whether it binds.
Candidates worth taking seriously for this agent: the deployed policy was
selected as the best of N seeds, so its greedy return is an optimistic estimate
for a fresh seed; the importance-sampling arms use a biased per-step ratio; the
value baseline is fitted on a small batch and its explained variance moves a lot
between batches; and CartPole terminates at a fixed angle threshold, so the agent
has never been asked to recover from a state past it._

**Foreseeable misuse and reward-specification risk.** _CartPole's reward is +1
per surviving step, so this agent optimises survival and nothing else — not
smoothness, not energy, not staying centred. Any real control problem where
"keep going" is not the whole goal needs a different reward, and an agent that
maximises the wrong reward competently is more dangerous than one that fails
visibly._

**Responsible disclosure.** _If you found a failure mode, who would you tell,
when, and what would you say?_
        """
    )
    st.caption(
        "This section is part of the rubric, not an afterthought. A results table "
        "with no limitations section is an unfinished deliverable."
    )
