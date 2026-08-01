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

The Topic 4 tabs are Concepts, **Bake-Off**, **Entropy Sweep**, Play, Run
History and Model Card. The second and third are the product: the client asked
for a bake-off of three control agents and for evidence about what the
exploration temperature does. Everything else on this page supports them.

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

st.set_page_config(page_title="Control Bench", layout="wide")

# Fixed order, fixed colours. Fixed so that "the blue line" means the same
# algorithm on every chart on this page and in every screenshot you paste into
# the report — a legend that reorders itself between tabs is how a reader
# misreads your result.
ALGORITHMS = ["a2c", "ppo", "sac"]
ALGO_LABELS = {
    "a2c": "A2C — advantage actor-critic",
    "ppo": "PPO — clipped surrogate",
    "sac": "SAC — soft actor-critic",
}
ALGO_SCALE = alt.Scale(domain=ALGORITHMS, range=["#1f77b4", "#ff7f0e", "#2ca02c"])

ALPHA_ARMS = ["alpha=0.5", "alpha=0.01", "auto"]
ALPHA_SCALE = alt.Scale(domain=ALPHA_ARMS, range=["#d62728", "#7f7f7f", "#2ca02c"])

# The random-policy return in each environment. Duplicated from envs/ rather
# than imported, because the UI must not import the training-side environment
# registry to draw a reference line — and because a number the reader can see in
# the source of the page they are looking at is easier to check.
RANDOM_BASELINE = {"CartPole-v1": 22.0, "Acrobot-v1": -500.0, "Pendulum-v1": -1200.0}
OBS_LABELS = {
    "CartPole-v1": ["cart position", "cart velocity", "pole angle", "pole angular velocity"],
    "Acrobot-v1": ["cos θ1", "sin θ1", "cos θ2", "sin θ2", "ω1", "ω2"],
    "Pendulum-v1": ["cos θ", "sin θ", "angular velocity"],
}


# ---------------------------------------------------------------------------
# Health banner. Shown before anything else, because every tab below depends on
# the tiers it reports on.
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
            f"policies loaded · data tier reachable"
        )
    else:
        st.warning(
            f"**Running in a degraded state.** {h.get('detail') or ''}\n\n"
            "Results below may be incomplete. This banner is deliberate: an "
            "empty chart and a broken database look identical otherwise."
        )
    return h


# ---------------------------------------------------------------------------
# Data loading. Cached, because the Bake-Off tab needs one call per run, and
# twelve runs times two endpoints is twenty-four round trips on every widget
# interaction otherwise — which on the free tier is felt, not measured.
# ---------------------------------------------------------------------------


@st.cache_data(ttl=120, show_spinner=False)
def load_runs(limit: int = 200) -> pd.DataFrame:
    return pd.DataFrame(service.runs(limit)["runs"])


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
def load_policy_updates(experiment_ids: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for eid in experiment_ids:
        try:
            body = service.policy_updates(eid)
        except service.ServiceError:
            continue
        if body["stats"]:
            f = pd.DataFrame(body["stats"])
            f["experiment_id"] = eid
            frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def load_entropy_sweep() -> pd.DataFrame:
    try:
        return pd.DataFrame(service.entropy_sweep()["rows"])
    except service.ServiceError:
        return pd.DataFrame()


def seed_band(
    df: pd.DataFrame,
    x: str,
    y: str,
    bin_width: int,
    x_title: str,
    y_title: str,
    colour: str = "algorithm",
    scale: alt.Scale | None = None,
):
    """Mean across seeds, with a ± one standard error band.

    The band is across SEEDS, which is the only error bar that supports a claim
    about a configuration. Shading the spread across episodes within one seed
    would give a wider, prettier band that says nothing about whether the
    configuration would work again.

    Binning first is not cosmetic either: an episode-return series is a spiky
    step function and an unbinned mean of three of them is unreadable. The bin
    width is written on the axis so the reader knows what they are looking at.
    """
    d = df.copy()
    d["_bin"] = (d[x] // bin_width) * bin_width
    per_seed = d.groupby([colour, "seed", "_bin"], as_index=False)[y].mean()
    agg = per_seed.groupby([colour, "_bin"], as_index=False).agg(
        mean=(y, "mean"), sd=(y, "std"), n=(y, "count")
    )
    agg["sd"] = agg["sd"].fillna(0.0)
    agg["se"] = agg["sd"] / np.sqrt(agg["n"].clip(lower=1))
    agg["lo"] = agg["mean"] - agg["se"]
    agg["hi"] = agg["mean"] + agg["se"]

    colour_enc = alt.Color(f"{colour}:N", scale=scale, title=colour) if scale is not None \
        else alt.Color(f"{colour}:N", title=colour)
    base = alt.Chart(agg)
    band = base.mark_area(opacity=0.22).encode(
        x=alt.X("_bin:Q", title=x_title),
        y=alt.Y("lo:Q", title=y_title, scale=alt.Scale(zero=False)),
        y2="hi:Q",
        color=colour_enc,
    )
    line = base.mark_line(strokeWidth=2).encode(
        x="_bin:Q",
        y=alt.Y("mean:Q", scale=alt.Scale(zero=False)),
        color=colour_enc,
        tooltip=[f"{colour}:N", "_bin:Q", "mean:Q", "se:Q", "n:Q"],
    )
    return (band + line).properties(height=360).interactive()


health = health_banner()

st.title("Control Bench")
st.caption(
    "Three trained control agents behind one API contract — A2C on CartPole, "
    "PPO on Acrobot, SAC on Pendulum — with their learning curves compared at "
    "matched environment-step budgets and the effect of SAC's exploration "
    "temperature measured rather than asserted."
)

TABS = st.tabs(
    ["Concepts", "Bake-Off", "Entropy Sweep", "Play", "Run History", "Model Card"]
)


# ===========================================================================
with TABS[0]:
    st.header("Concepts")
    st.caption(
        "The Theoretical Brief, mirrored from the README. Written for the "
        "colleague who has to approve this for a control project, not for a grader."
    )

    st.subheader("1. Why a critic, and what the advantage buys")
    st.markdown(
        """
Topic 3's policy gradient weighted each `log π(a|s)` by the Monte Carlo return
`G_t` — a sum of hundreds of noisy rewards. An **actor-critic** method replaces
that sum with a learned estimate. The actor is `π_θ(a|s)`; the critic is
`V_φ(s)`; the weight becomes the **advantage**
        """
    )
    st.latex(r"A(s,a) = Q(s,a) - V(s)")
    st.markdown(
        """
Subtracting any function of the state leaves the gradient **unbiased** — the
proof is one line, because `E_a[∇ log π(a|s)·b(s)] = b(s)·∇ Σ_a π(a|s) = b(s)·∇1
= 0` — and changes only its variance. The intuition is shorter than the algebra:
without a baseline, *every* action taken in a good state has its probability
increased, merely for having been taken somewhere good. With one, only actions
that beat their own state's average are reinforced. The advantage asks the right
question: *was this action better than what I usually get from here?*

This is the pivot the whole topic turns on. Pure **value-based** methods (Topics
1–2) learn Q and derive a policy by argmax — which needs a discrete action space
and gives no natural way to be stochastic. Pure **policy-gradient** methods
(Topic 3) learn π directly and pay for it in variance. Actor-critic keeps both
objects and uses each for what it is good at.
        """
    )

    st.subheader("2. The trust region: PPO's clip versus TRPO's constraint")
    st.markdown("PPO maximises a clipped surrogate of the importance-weighted objective:")
    st.latex(
        r"L^{\text{CLIP}}(\theta) = \mathbb{E}\Big[\min\big(r_t(\theta)A_t,\;"
        r"\mathrm{clip}(r_t(\theta),\,1-\epsilon,\,1+\epsilon)\,A_t\big)\Big],\quad"
        r"r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}"
    )
    st.markdown(
        """
Read what the `min` does. When `A_t > 0`, the objective is capped at
`(1+ε)A_t`, so pushing the probability higher yields nothing and **the gradient
becomes exactly zero** past the ceiling. When `A_t < 0` the `min` selects the
unclipped term, so a bad action can still be pushed down without limit. The trust
region is enforced by making the objective *flat*, not by constraining anything.

TRPO enforces it as a genuine constraint:
        """
    )
    st.latex(
        r"\max_\theta\; \mathbb{E}\big[r_t(\theta)A_t\big]\quad\text{subject to}\quad"
        r"\mathbb{E}\big[D_{\mathrm{KL}}(\pi_{\text{old}}\,\|\,\pi_\theta)\big] \le \delta"
    )
    st.info(
        "**The practical difference.** TRPO's KL is a guarantee bought with a "
        "conjugate-gradient solve and a line search on a Fisher-vector product. "
        "PPO's clip is a heuristic bought with two lines of code, and it does NOT "
        "bound the KL — it only makes large moves unrewarding. Whether the KL "
        "actually stayed small is therefore an empirical question, which is why "
        "`policy_updates.kl_divergence` is a logged column and why the Bake-Off "
        "tab plots it. Your own run is the evidence."
    )

    st.subheader("3. The performance difference lemma")
    st.markdown("For any two policies π and π′, with `d_π` the discounted state distribution:")
    st.latex(
        r"J(\pi') - J(\pi) \;=\; \frac{1}{1-\gamma}\,"
        r"\mathbb{E}_{s\sim d_{\pi'}}\,\mathbb{E}_{a\sim\pi'(\cdot|s)}\big[A_\pi(s,a)\big]"
    )
    st.markdown(
        """
Read the subscripts, because they are the whole point. The advantage is the OLD
policy's; the state distribution is the NEW policy's. That is what makes the
identity useless as an optimisation target directly — you would have to roll out
π′ to evaluate it — and it is exactly the term every method in this topic
approximates by pretending `d_{π'} ≈ d_π`. That approximation is good only while
the two policies are close, which is *why* a trust region is needed at all. The
clip and the KL constraint are two ways of keeping the approximation honest.
        """
    )

    st.subheader("4. Soft value functions and the soft Bellman equation")
    st.markdown("SAC changes the objective rather than the loss. Entropy enters the **return**:")
    st.latex(
        r"J(\pi) = \mathbb{E}\Big[\textstyle\sum_t \gamma^t\big(r(s_t,a_t)"
        r" + \alpha\,\mathcal{H}(\pi(\cdot|s_t))\big)\Big]"
    )
    st.markdown("Carrying that through the Bellman backup gives the **soft** equations:")
    st.latex(
        r"Q(s,a) = r(s,a) + \gamma\,\mathbb{E}_{s'}\big[V(s')\big],\qquad"
        r"V(s) = \mathbb{E}_{a\sim\pi}\big[Q(s,a) - \alpha\log\pi(a|s)\big]"
    )
    st.markdown(
        "and, when you solve for the policy that maximises the right-hand side, "
        "the **soft value** is a log-sum-exp rather than a max:"
    )
    st.latex(
        r"V_{\text{soft}}(s) = \alpha\log\!\int \exp\!\big(Q(s,a)/\alpha\big)\,da,"
        r"\qquad \pi^\star(a|s) \propto \exp\!\big(Q(s,a)/\alpha\big)"
    )
    st.markdown(
        """
Two limits worth being able to state out loud. As **α → 0** the log-sum-exp
becomes a max and π* becomes greedy — ordinary Q-learning. As **α → ∞** the
reward washes out and π* becomes uniform. α is the exchange rate between reward
and randomness, and the optimal policy of *this* objective is stochastic by
construction rather than by accident.

Note what that implies for the `− α log π` term in the target computed by
`train/sac._update`: it is not a regulariser bolted onto a value function, it
**is** the value function of the entropy-augmented MDP. Delete it and the actor
and the critic are optimising different objectives.
        """
    )

    st.subheader("5. The soft policy gradient theorem")
    st.markdown("The actor maximises the soft objective at each state:")
    st.latex(
        r"J_\pi(\theta) = \mathbb{E}_{s\sim\mathcal{D}}\,\mathbb{E}_{a\sim\pi_\theta}"
        r"\big[Q(s,a) - \alpha\log\pi_\theta(a|s)\big]"
    )
    st.markdown(
        "The action is continuous, so instead of the likelihood-ratio trick we "
        "**reparameterise**: `a = f_θ(s, ε) = tanh(μ_θ(s) + σ_θ(s)·ε)` with "
        "`ε ~ N(0, I)`, and differentiate straight through the sample:"
    )
    st.latex(
        r"\nabla_\theta J_\pi = \mathbb{E}\Big[\nabla_\theta\big(\alpha\log\pi_\theta(f_\theta"
        r"(s,\epsilon)|s)\big) + \big(\nabla_a\alpha\log\pi_\theta(a|s) - \nabla_a Q(s,a)\big)"
        r"\big|_{a=f_\theta}\nabla_\theta f_\theta(s,\epsilon)\Big]"
    )
    st.warning(
        "**The tanh correction.** Squashing changes the density: "
        "`log π(a|s) = log N(u; μ, σ) − Σ_i log(1 − tanh²(u_i))`. Dropping that "
        "second term is the most common SAC bug in the wild — the code still "
        "trains, the entropy term is simply wrong, and α ends up tuned against a "
        "quantity that is not the entropy of anything. See "
        "`train/nets.SquashedGaussianActor.sample`, which is the only place in "
        "this repository that computes it."
    )

    st.subheader("6. Why automatic tuning of α matters")
    st.markdown(
        "Rather than guessing α, constrain the average entropy and solve for the "
        "multiplier:"
    )
    st.latex(
        r"\max_\pi\;\mathbb{E}\Big[\textstyle\sum_t r_t\Big]\;\;\text{s.t.}\;\;"
        r"\mathbb{E}\big[-\log\pi(a_t|s_t)\big]\ge \bar{\mathcal{H}}"
        r"\qquad\Longrightarrow\qquad"
        r"L(\alpha) = -\alpha\big(\log\pi(a|s) + \bar{\mathcal{H}}\big)"
    )
    st.markdown(
        """
Three reasons this is not a convenience:

1. **α trades against a reward scale.** Pendulum's per-step reward reaches −16;
   a task with rewards in [0, 1] would need an α roughly sixteen times smaller
   for the same behaviour. A fixed α is a hyperparameter you must retune for
   every environment, and *nobody tells you when it is wrong* — the run just
   converges somewhere worse.
2. **The right α changes during training.** Early on the agent should be
   uncertain; later it should not. A constant cannot do both, and the automatic
   multiplier visibly falls as the policy sharpens. Watch `alpha` in
   `policy_updates`.
3. **The constraint is interpretable and the coefficient is not.** "Keep at
   least this much entropy" is a statement you can defend to a stakeholder;
   "α = 0.2" is not.

The Entropy Sweep tab is where you check whether that argument survives contact
with your own data. It may not, at the budget you can afford — say so if it does
not.
        """
    )

    st.subheader("7. Entropy and the trust region are two different regularisers")
    st.markdown(
        """
They are easy to confuse because both make a policy less extreme, and the topic
asks you to keep them apart.

| | entropy term (SAC) | trust region (PPO/TRPO) |
|---|---|---|
| **What it constrains** | how random π is, in absolute terms | how far π moved from π_old |
| **Reference point** | the uniform distribution | the previous policy |
| **Where it lives** | inside the reward, so it changes the optimal policy | outside the objective, a constraint on the step |
| **Effect at convergence** | the optimum is stochastic | the optimum is unchanged; only the path to it is |
| **Failure if too strong** | a deliberately random agent | learning stalls; nothing moves |
| **Failure if too weak** | premature determinism, brittle policy | destructive updates, collapse |

The unifying view — and this is the bridge to Topic 6 — is that both are KL
penalties against a reference distribution. Maximising `E[r] + α·H(π)` is
maximising `E[r] − α·KL(π ‖ uniform)` up to a constant, because
`KL(π ‖ uniform) = −H(π) + log|A|`. Maximising `E[r] − β·KL(π ‖ π_ref)` is the
RLHF objective. Same regularised objective, different reference: uniform in one
case, a reference policy in the other. Nothing new is introduced in Topic 6 that
is not already here.
        """
    )


# ===========================================================================
with TABS[1]:
    st.header("Bake-Off — three agents, one budget")
    st.markdown(
        "Learning curves with **± one standard error across seeds**, which is the "
        "only error bar that supports a claim about a configuration. Use the "
        "toggle to switch the x-axis between episodes and environment steps."
    )

    runs_df = load_runs(200)
    if runs_df.empty:
        st.info(
            "No runs logged yet. Run `python -m train.train` for the three "
            "deployable agents, then `python -m train.compare --seeds 3` for the "
            "matched-budget comparison the brief asks for."
        )
    else:
        col_a, col_b, col_c = st.columns([2, 2, 1])
        env_choice = col_a.selectbox(
            "Environment", sorted(runs_df["env_id"].dropna().unique()) or ["CartPole-v1"]
        )
        matched = col_b.toggle(
            "Matched environment-step budget",
            value=True,
            help=(
                "ON: the x-axis is environment steps, which is the only axis on "
                "which a sample-efficiency claim is fair. OFF: episodes — useful "
                "for reading a single run, and misleading across algorithms, "
                "because an agent that learns faster has longer CartPole episodes "
                "and shorter Acrobot ones at the same step count."
            ),
        )
        bin_width = col_c.number_input(
            "Bin width", min_value=1, value=2000 if matched else 20, step=100,
            help="Written on the axis so the reader knows what they are looking at.",
        )

        subset = runs_df[runs_df["env_id"] == env_choice]
        counts = subset.groupby("algorithm")["seed"].nunique()
        thin = [a for a, c in counts.items() if c < 3]
        if thin:
            st.warning(
                "Fewer than three seeds in: " + ", ".join(f"`{a}`" for a in thin) +
                ". A policy gradient on CartPole can reach 500 or plateau at 40 "
                "with nothing changed but the initialisation, so below three seeds "
                "you are measuring the seed rather than the algorithm."
            )

        eps = load_episodes(tuple(subset["experiment_id"]))
        if eps.empty:
            st.warning("Runs exist but no episode rows came back. Is the data tier awake?")
        else:
            eps = eps.join(
                subset.set_index("experiment_id")[["algorithm", "seed"]], on="experiment_id"
            )
            # `return` is a Python keyword, so the Pydantic field is `return_`
            # with an alias. Renaming once here keeps the rest of the tab readable.
            eps = eps.rename(columns={"return": "return_"})
            x_col = "env_steps" if matched else "episode_index"
            if matched and eps["env_steps"].isna().all():
                st.error(
                    "These runs have no `env_steps`. That column was added in "
                    "`002_topic4.sql`; runs logged before it was applied cannot be "
                    "compared at a matched budget. Retrain, or switch the toggle off."
                )
            else:
                plot = eps.dropna(subset=[x_col])
                st.altair_chart(
                    seed_band(
                        plot, x_col, "return_", int(bin_width),
                        f"{'environment step' if matched else 'training episode'} "
                        f"(mean of {int(bin_width)})",
                        "episode return",
                        scale=ALGO_SCALE,
                    ),
                    use_container_width=True,
                )
                st.caption(
                    f"Random-policy return in {env_choice} ≈ "
                    f"**{RANDOM_BASELINE.get(env_choice, float('nan')):.0f}**. On "
                    "Acrobot that number is a FLOOR rather than a mean: the reward "
                    "is −1 per step until the goal is reached and a random policy "
                    "essentially never reaches it, so −450 is real progress even "
                    "though it is only 10% of the way up the axis."
                )

            st.subheader("Final comparison")
            tail_frames = []
            for eid, grp in eps.groupby("experiment_id"):
                tail_frames.append(grp.nlargest(100, "episode_index"))
            tail = pd.concat(tail_frames) if tail_frames else eps
            per_seed = tail.groupby(["algorithm", "seed"], as_index=False)["return_"].mean()
            table = per_seed.groupby("algorithm", as_index=False).agg(
                seeds=("seed", "nunique"), mean_return=("return_", "mean"),
                sd=("return_", "std"),
            )
            table["sd"] = table["sd"].fillna(0.0)
            table["stderr_across_seeds"] = table["sd"] / np.sqrt(table["seeds"].clip(lower=1))
            table["algorithm"] = table["algorithm"].map(lambda a: ALGO_LABELS.get(a, a))
            st.dataframe(
                table.drop(columns=["sd"]).round(2), use_container_width=True, hide_index=True
            )
            st.caption(
                "Mean training return over each run's last 100 episodes. Note that "
                "PPO takes roughly five times as many GRADIENT steps as A2C for the "
                "same environment budget, so this table compares SAMPLE efficiency "
                "and not compute — say which one you are claiming."
            )

        st.subheader("Trust region — the KL PPO does not actually constrain")
        updates = load_policy_updates(tuple(subset["experiment_id"]))
        if updates.empty or updates["kl_divergence"].isna().all():
            st.info(
                "No KL rows for this environment. Only PPO logs a KL — A2C takes one "
                "step per batch and never asks how far it moved, and SAC has no trust "
                "region at all. Those are NULLs, not zeros, on purpose."
            )
        else:
            kl = updates.dropna(subset=["kl_divergence"]).join(
                subset.set_index("experiment_id")[["algorithm", "seed"]], on="experiment_id"
            )
            st.altair_chart(
                seed_band(
                    kl, "env_steps", "kl_divergence", int(max(bin_width, 1000)),
                    "environment step", "mean KL(π_old ‖ π_new) per update",
                    scale=ALGO_SCALE,
                ),
                use_container_width=True,
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Median KL per update", f"{kl['kl_divergence'].median():.4f}")
            c2.metric("Largest KL observed", f"{kl['kl_divergence'].max():.4f}")
            c3.metric("Median clip fraction", f"{kl['clip_fraction'].median():.3f}")
            st.caption(
                "PPO does not bound this quantity — it clips a likelihood ratio and "
                "makes large moves unrewarding, which is not the same thing. A "
                "typical TRPO trust region is δ ≈ 0.01; compare your median against "
                "it and say whether the clip alone was enough on your runs. A clip "
                "fraction near zero means the clip never engaged and you were "
                "running an ordinary surrogate; near one means most of the batch was "
                "outside the region and was wasted."
            )


# ===========================================================================
with TABS[2]:
    st.header("Entropy Sweep — what α actually does")
    st.markdown(
        "SAC retrained under three temperature regimes — α = 0.5, α = 0.01 and "
        "automatic tuning — across at least three seeds each. Convergence speed, "
        "final performance and stability across seeds."
    )

    sweep = load_entropy_sweep()
    if sweep.empty:
        st.info(
            "No sweep rows yet. Run `python -m train.entropy_sweep --seeds 3` "
            "(add `--quick` for a ten-minute smoke test that is not a result)."
        )
    else:
        counts = sweep.groupby("alpha_setting")["seed"].nunique()
        thin = [a for a, c in counts.items() if c < 3]
        if thin:
            st.warning(
                "Fewer than three seeds in: " + ", ".join(f"`{a}`" for a in thin) +
                ". Three seeds is enough to notice a large effect and not enough to "
                "measure a small one."
            )

        summary = sweep.groupby(["alpha_setting", "mode"], as_index=False).agg(
            seeds=("seed", "nunique"),
            mean_final_return=("mean_return_last_100", "mean"),
            across_seed_sd=("mean_return_last_100", "std"),
            mean_policy_entropy=("mean_policy_entropy", "mean"),
            mean_final_alpha=("alpha_value", "mean"),
            mean_eval_return=("eval_mean_return", "mean"),
        )
        summary["across_seed_sd"] = summary["across_seed_sd"].fillna(0.0)
        # Stability is the VARIANCE across seeds, which the brief names
        # explicitly. Reported alongside the standard deviation because a
        # variance is what the SQL view computes and a standard deviation is what
        # a reader can compare against the returns in the column beside it.
        summary["across_seed_variance"] = summary["across_seed_sd"] ** 2
        reached = (
            sweep.assign(_r=sweep["episodes_to_threshold"].notna())
            .groupby("alpha_setting", as_index=False)
            .agg(reached=("_r", "sum"),
                 mean_episodes_to_threshold=("episodes_to_threshold", "mean"))
        )
        summary = summary.merge(reached, on="alpha_setting", how="left")
        st.dataframe(summary.round(3), use_container_width=True, hide_index=True)
        st.caption(
            "`mean_episodes_to_threshold` is averaged over the runs that REACHED "
            f"the threshold ({sweep['threshold'].iloc[0]:.0f}); `reached` says how "
            "many did. A run that never got there stores NULL rather than a large "
            "sentinel, because a sentinel would be averaged in and would invent a "
            "slow convergence where there was none at all. "
            "`mean_policy_entropy` is a DIFFERENTIAL entropy and is negative for "
            "all three arms — that is correct, not a bug: a continuous density can "
            "exceed 1, so it is not bounded below by zero."
        )

        left, right = st.columns(2)
        with left:
            st.subheader("Final return by α")
            st.altair_chart(
                alt.Chart(sweep).mark_circle(size=140, opacity=0.75).encode(
                    x=alt.X("alpha_setting:N", title="temperature regime",
                            sort=ALPHA_ARMS),
                    y=alt.Y("mean_return_last_100:Q", title="mean return, last 100 episodes",
                            scale=alt.Scale(zero=False)),
                    color=alt.Color("alpha_setting:N", scale=ALPHA_SCALE, legend=None),
                    tooltip=["alpha_setting", "seed", "mean_return_last_100",
                             "mean_policy_entropy", "alpha_value"],
                ).properties(height=320),
                use_container_width=True,
            )
            st.caption(
                "One point per SEED, not a bar of the mean. With three seeds a bar "
                "chart hides exactly the thing the stability column is about."
            )
        with right:
            st.subheader("Policy entropy by α")
            st.altair_chart(
                alt.Chart(sweep).mark_circle(size=140, opacity=0.75).encode(
                    x=alt.X("alpha_setting:N", title="temperature regime", sort=ALPHA_ARMS),
                    y=alt.Y("mean_policy_entropy:Q", title="mean differential entropy (nats)",
                            scale=alt.Scale(zero=False)),
                    color=alt.Color("alpha_setting:N", scale=ALPHA_SCALE, legend=None),
                    tooltip=["alpha_setting", "seed", "mean_policy_entropy"],
                ).properties(height=320),
                use_container_width=True,
            )
            st.caption(
                "This is the column that shows the mechanism rather than the "
                "outcome: a higher α should buy a higher entropy. If it does not, "
                "the α is not reaching the objective and the returns below it mean "
                "nothing."
            )

        st.subheader("What to write about this")
        st.info(
            "100–250 words. The honest reading: if two arms' means differ by less "
            "than the spread across their seeds, the correct sentence is *this "
            "study did not separate them*, not *α = 0.5 was slightly better*. State "
            "which of the two your numbers support, and quote the seeds."
        )


# ===========================================================================
with TABS[3]:
    st.header("Play — any of the three agents, live")

    if not health.get("policy_artifact_loaded"):
        st.error(
            "No policy artifact is loaded, so there is nothing to play with. Run "
            "`python -m train.train`, which exports all three, then reload."
        )
    else:
        st.caption(
            "Everything on this tab is served by the NumPy forward pass in "
            "`api/forward.py` — the same archives and the same matrix multiplies "
            "that run in production. No PyTorch is loaded in this process; "
            "`GET /version` reports that honestly."
        )
        try:
            registered = {p["name"]: p for p in service.policies()["policies"]}
        except service.ServiceError as exc:
            registered = {}
            st.error(str(exc))

        names = [n for n in sorted(registered) if registered[n].get("env_id")]
        col_a, col_b, col_c = st.columns([2, 1, 1])
        policy_name = col_a.selectbox("Policy", names or ["default"])
        episodes_n = col_b.slider("Evaluation episodes", 1, 50, 10)
        seed = col_c.number_input("Seed", value=0, step=1,
                                  help="Name the seed or the result is not evidence.")

        meta = registered.get(policy_name, {})
        env_id = meta.get("env_id") or "CartPole-v1"
        st.caption(
            f"`{policy_name}` · {env_id} · {meta.get('obs_dim')}-dimensional "
            f"observation · {meta.get('action_space')} action · "
            f"{meta.get('bytes', 0)} bytes · `{str(meta.get('sha256', ''))[:12]}…`"
        )

        if st.button("Run a deterministic evaluation", type="primary"):
            try:
                r = service.rollout(policy_name, episodes_n, int(seed))
            except service.ServiceError as exc:
                st.error(str(exc))
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Mean return", f"{r['mean_return']:.1f}")
                m2.metric(
                    "Standard error", f"{r['stderr_return']:.2f}",
                    help="s / sqrt(n). Quote this alongside the mean, never the mean alone.",
                )
                m3.metric("Mean episode length", f"{r['mean_length']:.1f}")
                m4.metric(
                    "Random baseline",
                    f"{r.get('random_baseline', float('nan')):.0f}",
                    help="What a uniformly random policy scores here. Without it the "
                         "number to its left is not interpretable.",
                )
                st.caption(
                    f"artifact `{r['policy_sha256'][:12]}…` · {r['episodes']} episodes · "
                    f"seed {r['seed']} · env `{r.get('env_id')}` · **deterministic**, "
                    "which is a different question from the training curve on the "
                    "Bake-Off tab. For SAC it is the MODE of the squashed Gaussian, "
                    "not its mean — that has no closed form."
                )
                if r["trajectory"]:
                    st.subheader("One full trajectory")
                    traj = pd.DataFrame(r["trajectory"])
                    labels = OBS_LABELS.get(env_id)
                    if labels and len(labels) == len(traj["state"].iloc[0]):
                        states = pd.DataFrame(traj["state"].tolist(), columns=labels)
                        st.line_chart(states)
                    st.dataframe(traj, use_container_width=True, height=240)
                st.bar_chart(pd.DataFrame({"return": r["returns"]}))

        st.divider()
        st.subheader("Ask the policy about one state")
        st.caption(
            "Move a slider and watch the decision change. Note that the sliders "
            "below change shape with the policy — this is the dimensionality "
            "hazard the required 422 test exists for, made visible: send this "
            "policy the wrong number of values and the service refuses with a "
            "message naming both widths."
        )
        labels = OBS_LABELS.get(env_id, [f"x{i}" for i in range(int(meta.get("obs_dim") or 1))])
        cols = st.columns(len(labels))
        state = [
            cols[i].slider(lab, -3.0, 3.0, 0.0, 0.05, key=f"{policy_name}-{i}")
            for i, lab in enumerate(labels)
        ]
        try:
            resp = service.act(state, policy_name, deterministic=True)
        except service.ServiceError as exc:
            st.error(str(exc))
        else:
            left, right = st.columns([1, 2])
            if meta.get("action_space") == "continuous":
                left.metric("Torque", f"{resp['action'][0]:+.3f}")
                right.caption(
                    f"mean log σ in this state: **{resp['value_estimate']:+.3f}**. That is "
                    "how uncertain the policy is here — the SAC artifact is the ACTOR "
                    "alone, so there is no Q-value to report. A low log σ means the "
                    "agent has made up its mind about this state."
                )
            else:
                left.metric("Action", str(resp["action"]))
                confidence = float(resp.get("value_estimate") or 0.0)
                right.progress(
                    min(max(confidence, 0.0), 1.0),
                    text=f"probability of the chosen action: {confidence:.3f}",
                )
                if confidence > 0.99:
                    right.caption(
                        "Near-deterministic in this state. That is a finding, not a "
                        "problem — but check the entropy series: a policy that went "
                        "deterministic everywhere and early stopped exploring."
                    )
            left.caption(f"{resp['latency_ms']:.2f} ms · `{resp['policy_sha256'][:12]}…`")


# ===========================================================================
with TABS[4]:
    st.header("Run history")
    st.caption("Read-only. This tab issues no writes and holds no service-role key.")
    try:
        data = service.runs(200)
        if data.get("degraded"):
            st.warning(
                "The data tier did not answer, so this table is empty rather than "
                "complete. A free-tier project pauses after a week idle."
            )
        elif data["runs"]:
            df = pd.DataFrame(data["runs"])
            st.dataframe(df, use_container_width=True, height=420)
            st.caption(
                f"{len(df)} runs · {int(df['episodes_logged'].sum())} episodes logged · "
                f"{df['seed'].nunique()} distinct seeds"
            )
        else:
            st.info("Nothing logged yet.")
    except service.ServiceError as exc:
        st.error(str(exc))

    st.subheader("Registered artifacts")
    st.caption(
        "Size, checksum, environment and action space for every `.npz` the service "
        "can load. The checksum is what `/act` returns and `audit_log` stores, so "
        "six weeks from now \"which artifact produced this action\" is a join "
        "rather than a guess."
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
**What this service does.** _One paragraph, in the stakeholder's language. It
hosts three control agents behind one contract and lets you compare them on
equal terms. Say what a good day looks like._

**What it does not do.** _State the boundary explicitly. These policies were
trained on CartPole-v1, Acrobot-v1 and Pendulum-v1 and have never seen a pole of
a different length, a different gravity, a torque limit other than ±2, or an
observation with sensor noise._

**The three artifacts.**

| agent | environment | architecture | head | artifact |
|---|---|---|---|---|
| A2C | CartPole-v1 | 4 → 64 → 64 → 2 | softmax | ~19 KB `.npz` |
| PPO | Acrobot-v1 | 6 → 64 → 64 → 3 | softmax | ~19 KB `.npz` |
| SAC | Pendulum-v1 | 3 → 256 → 256 → 2 | tanh-squashed Gaussian, scaled to ±2 | ~250 KB `.npz` |

_Keep the sizes and checksums in sync with the Run History tab. The SAC artifact
is the ACTOR only — the twin critics are training-time objects and are not
deployed._

**Training data and environment.** _Environment ids, environment steps per run,
number of seeds, and the exact commands. A result whose command is not written
down is not reproducible._

**Evaluation.** _Deterministic return with a standard error, from `evaluations`,
against the random baseline for each environment. Not the training curve — say
which is which. For SAC the deterministic action is the MODE of the squashed
Gaussian; the mean of that distribution has no closed form and is not what is
served._

**Numerical equivalence.** _Topic-specific and required: the measured maximum
absolute difference between the NumPy and PyTorch forward passes, for each of
the three policies, with the tolerance. `make equivalence` prints all three._

**Trust region and entropy.** _Your median per-update KL for PPO against the
δ ≈ 0.01 a TRPO implementation would enforce, and the α the automatic tuner
converged to against the two fixed values you tried._

**Limitations.** _At least four, each with how you would test whether it binds.
See the README's limitations section — do not paraphrase it here, point at it._

**Foreseeable misuse and reward-specification risk.** _CartPole's reward is +1
per surviving step, Acrobot's is −1 per step until the goal, Pendulum's punishes
angle and torque on a fixed quadratic. None of them encodes smoothness, energy
budget, actuator wear or safety margin. An agent that maximises the wrong reward
competently is more dangerous than one that fails visibly, and nothing in this
product tells you whether the objective was the right one._

**Responsible disclosure.** _If you found a failure mode, who would you tell,
when, and what would you say?_
        """
    )
    st.caption(
        "This section is part of the rubric, not an afterthought. A results table "
        "with no limitations section is an unfinished deliverable."
    )
