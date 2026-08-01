"""
ui/app.py — the presentation tier. The only thing a non-technical stakeholder
ever needs to open.

Three rules this file obeys, and yours must too:

  1. It contains NO policy code and NO training code. Every number comes back
     from the service tier through ui/service.py — including both value
     functions, which is why `GET /value_map` exists rather than this file
     opening an `.npz`.
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

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from shared.config import get_settings          # noqa: E402
from ui import service                          # noqa: E402

settings = get_settings()

st.set_page_config(page_title="Policy Lab", layout="wide")

SOURCES = ("value_iteration", "monte_carlo")
SOURCE_LABELS = {
    "value_iteration": "Planner (value iteration — exact)",
    "monte_carlo": "Learner (Monte Carlo — sampled)",
}


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
            f"policies loaded · data tier reachable"
        )
    else:
        st.warning(
            f"**Running in a degraded state.** {h.get('detail') or ''}\n\n"
            "Results below may be incomplete. This banner is deliberate: an "
            "empty chart and a broken database look identical otherwise."
        )
    return h


health = health_banner()

st.title("Policy Lab — planning versus learning on one problem")
st.caption(
    "Two agents solve the same 5×5 routing grid. One is given the transition "
    "model and computes the optimal policy exactly; the other never sees the "
    "model and estimates it from sampled returns. Same environment, same "
    "schema, same API — so the comparison is between the two methods rather "
    "than between two pieces of software."
)

TABS = st.tabs(["Concepts", "Value Map", "Convergence", "Run History", "Model Card"])


# ---------------------------------------------------------------------------
with TABS[0]:
    st.header("Concepts")
    st.markdown(
        "This tab is the **Theoretical Brief**. It is written for a colleague "
        "who runs the routing operation and has never seen a Bellman equation, "
        "and it is the same text as the corresponding README section."
    )

    st.subheader("1. The Bellman expectation equation, derived")
    st.markdown(
        "Start from the only definition in play: the value of a state under a "
        "policy is the expected discounted return from that state."
    )
    st.latex(r"v_\pi(s) \;=\; \mathbb{E}_\pi\!\left[\, G_t \mid S_t = s \,\right],"
             r"\qquad G_t \;=\; R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots")
    st.markdown(
        "Split the first reward off the rest. The tail is itself a return, one "
        "step later, discounted once:"
    )
    st.latex(r"G_t \;=\; R_{t+1} + \gamma\,G_{t+1}")
    st.markdown(
        "Substitute, then use the tower property of conditional expectation to "
        "average over what happens on the first step — which action the policy "
        "takes, and where the environment puts you:"
    )
    st.latex(
        r"v_\pi(s) \;=\; \sum_{a} \pi(a \mid s) \sum_{s'} p(s' \mid s, a)"
        r"\Big[\, r(s,a,s') + \gamma\, v_\pi(s') \,\Big]"
    )
    st.markdown(
        "Read it from the inside out: **r** is what you are paid for the step, "
        "**γ v(s′)** is what the rest of the job is worth from where you land, "
        "**p(s′ | s, a)** averages over the wind, and **π(a | s)** averages over "
        "your own choices. That the value of a state is defined through the "
        "values of other states is the recursion every algorithm here exploits."
    )
    st.markdown(
        "Replace the average over actions with a maximum and you have the "
        "**Bellman optimality** equation, which is what "
        "`train/value_iteration.py` iterates:"
    )
    st.latex(
        r"v_*(s) \;=\; \max_a \sum_{s'} p(s' \mid s, a)\Big[\, r(s,a,s') "
        r"+ \gamma\, v_*(s') \,\Big]"
    )
    st.info(
        "Why iterating it works: the backup is a γ-contraction in the max "
        "norm, so every sweep shrinks the distance to the true values by at "
        "least a factor of γ. That is also where the stopping rule comes from "
        "— a residual below θ guarantees the answer is within θγ/(1−γ) of "
        "exact. Lower γ therefore converges faster and values the future less; "
        "the two effects are the same knob and cannot be separated."
    )

    st.subheader("2. The policy improvement theorem")
    st.latex(r"\text{if } q_\pi\big(s, \pi'(s)\big) \;\ge\; v_\pi(s)\ \ \forall s"
             r"\quad\Longrightarrow\quad v_{\pi'}(s) \;\ge\; v_\pi(s)\ \ \forall s")
    st.markdown(
        """
**What it says.** If in some state there is an action that looks better than
what your policy currently does — better meaning its action-value exceeds the
state's value — then the policy that takes that action there and behaves as
before everywhere else is at least as good **everywhere**, not only in that
state. Strictly better if the inequality is strict anywhere reachable.

**Why it is true, informally.** Follow the new policy for one step and then
revert to the old one: by assumption that is no worse than the old policy. Now
do it for two steps, then three. Each extra step of the new policy can only
help, and unrolling the argument gives the result in the limit. This is the
engine inside every policy-iteration-like algorithm, Monte Carlo control
included: it is what licenses *"make the policy greedy with respect to your
current value estimate"* as a step that cannot make things worse.

**Why some policies are not comparable.** The ordering is *partial*. Policy A
may be better than B from the depot and worse from the north corner; neither
dominates and the theorem says nothing about them. That is not a technicality
— it is why "which policy is better?" only becomes a well-posed question once a
start-state distribution is fixed, and why the results table below reports the
value at a **named** state rather than one scalar score.
        """
    )

    st.subheader("3. First-visit versus every-visit Monte Carlo")
    st.markdown(
        """
Both estimate v_π(s) by averaging observed returns from s. They differ only in
which visits count when one episode passes through s more than once.

| | first-visit | every-visit |
|---|---|---|
| Samples per episode | the first visit only | every visit |
| Bias at finite n | unbiased | biased, vanishing as n grows |
| Independence | i.i.d. across episodes | correlated within an episode |
| Data efficiency | lower | higher |

Both converge to v_π by the law of large numbers. The practical difference is
that a first-visit standard error computed the usual way is honest, while an
every-visit one understates the uncertainty: the second visit's return is a
*suffix* of the first visit's, so the two are not independent observations.
This repository uses first-visit, and `shared/preprocess.first_visit_indices`
is the single line that decides it.
        """
    )

    st.subheader("4. Exploring starts and soft policies")
    st.markdown(
        """
Monte Carlo can only estimate the value of a state it has actually visited.
Under a deterministic policy from a fixed depot most of this grid is never
seen, so something has to guarantee coverage.

**Exploring starts** begins each episode in a uniformly chosen state — and, for
control, with a uniformly chosen first action. That is the assumption used
here, and it is a real one: it needs an environment you can reset into any
state. A simulator gives you that. A forklift on a warehouse floor does not.

**Soft policies** buy coverage differently: every action keeps positive
probability, as in ε-greedy where π(a|s) ≥ ε/|A| for all a. No special reset is
required, which is why a deployed system uses this instead. The cost is that
you are now evaluating the ε-soft policy rather than the greedy one you intend
to ship. The two differ by O(ε), and quietly reporting the first as though it
were the second is where a good deal of optimistic offline evidence comes from.
        """
    )

    st.subheader("5. Importance sampling for off-policy evaluation")
    st.markdown(
        "Suppose the data was collected under a **behaviour** policy *b* — last "
        "quarter's dispatch rules — and you want the value of a **target** "
        "policy *π* that has never been run. Re-weight each observed return by "
        "how much more likely the target policy was to have produced that exact "
        "trajectory:"
    )
    st.latex(r"\rho_{t:T-1} \;=\; \prod_{k=t}^{T-1} "
             r"\frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}")
    st.latex(r"V_{\text{ordinary}}(s) = \frac{\sum_t \rho_t\,G_t}{|\mathcal{T}(s)|}"
             r"\qquad\qquad"
             r"V_{\text{weighted}}(s) = \frac{\sum_t \rho_t\,G_t}{\sum_t \rho_t}")
    st.markdown(
        """
**Ordinary** importance sampling is unbiased and has unbounded variance: one
long trajectory that the target policy loved and the behaviour policy rarely
produced carries a ratio in the thousands and dominates the average.
**Weighted** importance sampling is biased at finite sample sizes and
consistent, with dramatically lower variance — it is what practitioners
actually use.

Two conditions bind. *Coverage*: b(a|s) > 0 wherever π(a|s) > 0, or the ratio
is undefined. And the behaviour probabilities must have been **recorded at the
time**; they cannot be reconstructed afterwards from a table of chosen actions.
That is a schema requirement rather than an algorithmic one — for an off-policy
estimate to be recomputable by someone else from your data alone, `episodes`
has to carry the per-step behaviour probability, or enough of the behaviour
policy's parameters to recompute it exactly.
        """
    )


# ---------------------------------------------------------------------------
def _as_frame(grid: dict) -> pd.DataFrame:
    """A value grid as a (rows x cols) frame.

    Row-major, matching `shared.preprocess.state_index`. On a square grid a
    transpose is invisible and would silently flip every conclusion drawn from
    the picture, which is why the reshape happens in exactly one place.
    """
    arr = np.asarray(grid["values"], dtype=float).reshape(grid["rows"], grid["cols"])
    return pd.DataFrame(
        arr,
        index=[f"row {r}" for r in range(grid["rows"])],
        columns=[f"col {c}" for c in range(grid["cols"])],
    )


with TABS[1]:
    st.header("Value map")
    st.caption(
        "Both value functions, then the difference. Two heat maps that 'look "
        "similar' is not a measurement; the third panel is where the "
        "disagreement actually shows."
    )

    try:
        vm = service.value_map()
    except service.ServiceError as exc:
        st.error(str(exc))
        vm = None

    if vm is not None and not vm["grids"]:
        st.error(
            "No value functions are available. Run `python -m train.train` to "
            "produce `policies/value_iteration.npz` and `policies/monte_carlo.npz`."
        )
    elif vm is not None:
        if vm["missing"]:
            st.warning(
                "No exported artifact for: " + ", ".join(vm["missing"]) +
                ". This panel shows one agent short rather than a blank grid — "
                "a missing agent and a zero-valued one must not look alike."
            )

        panels = st.columns(len(vm["grids"]) + (1 if vm["difference"] else 0))
        for panel, grid in zip(panels, vm["grids"]):
            with panel:
                st.subheader(SOURCE_LABELS.get(grid["label"], grid["label"]))
                st.dataframe(
                    _as_frame(grid).style.background_gradient(cmap="viridis")
                    .format("{:.3f}"),
                    use_container_width=True,
                )
                if grid["policy_arrows"]:
                    arrows = np.asarray(grid["policy_arrows"], dtype=object)
                    st.text("\n".join(
                        " ".join(row)
                        for row in arrows.reshape(grid["rows"], grid["cols"])
                    ))

        if vm["difference"]:
            with panels[-1]:
                st.subheader("Learned − exact")
                diff = _as_frame(vm["difference"])
                st.dataframe(
                    # Diverging scale centred on zero. On a sequential scale the
                    # sign of a difference map is unreadable, and the sign is
                    # the only thing anyone wants from it.
                    diff.style.background_gradient(cmap="coolwarm", vmin=-0.1, vmax=0.1)
                    .format("{:+.3f}"),
                    use_container_width=True,
                )
                st.metric("Largest disagreement",
                          f"{np.abs(diff.to_numpy()).max():.4f}")

    st.divider()
    st.subheader("Ask both agents the same question")
    st.caption(
        "A live `POST /rollout` against each policy source on the SAME seed, so "
        "both agents face the same wind."
    )
    left, right = st.columns(2)
    episodes = left.slider("Evaluation episodes", 1, 200, 50)
    seed = right.number_input("Seed", value=0, step=1,
                              help="Name the seed or the result is not evidence.")

    if st.button("Run both"):
        rows = []
        for source in SOURCES:
            try:
                r = service.rollout(episodes=int(episodes), seed=int(seed),
                                    policy_source=source)
            except service.ServiceError as exc:
                st.error(f"{SOURCE_LABELS[source]}: {exc}")
                continue
            rows.append(
                {
                    "agent": SOURCE_LABELS[source],
                    "mean return": r["mean_return"],
                    "std error": r["stderr_return"],
                    "mean length": r["mean_length"],
                    "artifact": r["policy_sha256"][:12] + "…",
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.caption(
                f"{episodes} episodes on seed {seed}. One seed is a "
                "demonstration, not evidence — the numbers quoted in the README "
                "come from the ten-seed study on the Convergence tab."
            )


# ---------------------------------------------------------------------------
with TABS[2]:
    st.header("Convergence")
    st.caption(
        "How much experience buys how much accuracy: RMSE between the Monte "
        "Carlo value estimate and the exact solution against episode budget, "
        "with a 95% band across independent seeds."
    )

    try:
        conv = service.convergence()
    except service.ServiceError as exc:
        st.error(str(exc))
        conv = None

    if conv is None or not conv["points"]:
        st.info(
            "No convergence rows yet. Run `python -m train.compare` — it writes "
            "one row per seed per budget to `evaluations`, and a summary to "
            "`reports/convergence.json`."
        )
    else:
        if conv["degraded"]:
            st.warning(
                f"These numbers came from **{conv['source']}**, not from a live "
                "query against the data tier. They are exactly as fresh as the "
                "last committed `python -m train.compare` run."
            )
        df = pd.DataFrame(conv["points"])
        st.line_chart(df.set_index("episodes")[["mean_rmse", "ci95_low", "ci95_high"]])
        st.dataframe(
            df.rename(columns={
                "mean_rmse": "mean RMSE",
                "ci95_low": "95% low",
                "ci95_high": "95% high",
                "equivalent_at_5pct": "indistinguishable at 5%",
            }),
            use_container_width=True,
        )
        if conv.get("delta") is not None:
            st.metric("Equivalence margin δ", f"{conv['delta']:.5f}")
        n = conv.get("episodes_to_indistinguishable")
        if n:
            st.success(
                f"From **{n:,} episodes** onward the Monte Carlo estimate is "
                "statistically indistinguishable from the exact solution at the "
                "5% level."
            )
        else:
            st.info(
                "The estimate has not become statistically indistinguishable "
                "from the exact solution anywhere in the budget grid. That is a "
                "legitimate result — report it rather than extending the grid "
                "until it goes away."
            )
        if conv.get("test"):
            st.caption("Test: " + conv["test"])
        st.caption(
            "The band drawn here is a normal approximation (±1.96 SE) because "
            "the service tier has no t distribution — SciPy is a training-tier "
            "dependency and stays out of the deployed environment. The interval "
            "quoted in the README uses t (2.262 at 9 degrees of freedom) and is "
            "about 15% wider."
        )


# ---------------------------------------------------------------------------
with TABS[3]:
    st.header("Run history")
    st.caption("Read-only. This tab issues no writes and holds no service-role key.")
    try:
        data = service.runs(100)
        if data.get("degraded"):
            st.warning(
                "The data tier did not answer, so this table is empty rather "
                "than complete."
            )
        elif data["runs"]:
            df = pd.DataFrame(data["runs"])
            st.dataframe(df, use_container_width=True, height=380)

            st.subheader("Rows by algorithm")
            st.dataframe(
                df.groupby("algorithm")
                .agg(seeds=("seed", "nunique"), rows=("episodes_logged", "sum"))
                .reset_index(),
                use_container_width=True,
            )
            st.caption(
                "`value_iteration` rows are SWEEPS, not episodes, and its seed "
                "column is a placeholder — a planner has no random stream. "
                "Reading its `seeds = 1` as thin evidence would be a misreading: "
                "there is nothing to average over."
            )
        else:
            st.info("Nothing logged yet. Run `python -m train.train`.")
    except service.ServiceError as exc:
        st.error(str(exc))

    st.subheader("Registered artifacts")
    try:
        st.dataframe(pd.DataFrame(service.policies()["policies"]),
                     use_container_width=True)
    except service.ServiceError as exc:
        st.error(str(exc))


# ---------------------------------------------------------------------------
with TABS[4]:
    st.header("Model card")
    st.markdown(
        """
**What this service does.** It answers one question — *which way should the
vehicle move from this cell?* — in two different ways, and quantifies how far
apart the two answers are. The planner solves the routing problem exactly from
a known model of how the vehicle moves. The learner estimates the same
quantities from simulated experience with no model at all.

**What it does not do.** It does not plan a route through a real warehouse: the
environment is a 5×5 abstraction with a hand-specified 20% slip probability. It
does not adapt online — both policies are frozen artifacts, and nothing in the
serving path learns. It does not handle a grid it was not built for; the
artifacts are 25-state tables and a request for state 25 is a 422, by design.

_Replace the bracketed placeholders with your own numbers before submitting._

**Environment.** `GridWorld5x5-v1` — slip 0.2, step cost −0.02, goal +1, pits
−1, γ = 0.95, 100-step truncation.

**Training.** Planner: [n] sweeps to a Bellman residual of [r], error bound
[b]. Learner: [n] episodes of Monte Carlo control with exploring starts, seed
[s], truncation rate [t]%.

**Evaluation.** Greedy return with a standard error over [n] episodes and [k]
seeds; RMSE against the exact solution at [budgets] with 95% intervals.

**Limitations.** _At least four, each with how you would test whether it binds:_

1. *Exploring starts is an assumption a deployed system cannot make.* Test:
   re-run the learner with an ε-soft policy from the fixed depot and compare
   both the RMSE curve and the visit count of the least-visited cell.
2. *The truncation cap biases returns downward* for any policy that fails to
   terminate. Test: `train/monte_carlo.py` prints the truncation rate; re-run
   at a larger cap and check whether the value function moves.
3. *The reward specification encodes an unexamined trade-off* — see below.
   Test: sweep `pit_penalty` and record where the optimal policy changes.
4. *Every planner number is conditional on the model being correct*, which is
   precisely the second consultant's objection. Test: perturb `slip` by ±0.05,
   re-plan, then evaluate the OLD plan under the NEW dynamics and report the
   loss.

**Foreseeable misuse and reward-specification risk.** A step cost of −0.02
against a pit penalty of −1 says that a fifty-step detour is worth exactly one
accident. Nobody stated that trade-off out loud; it fell out of two numbers
chosen for convenience. Optimise this reward harder — a faster vehicle, a
denser grid — and the agent will find routes that are technically optimal and
operationally unacceptable. Reward specification is a stakeholder conversation,
not a hyperparameter.

**Responsible disclosure.** _If you found a failure mode in this agent, who
would you tell, when, and what exactly would you say?_
        """
    )
    st.caption(
        "This section is part of the rubric, not an afterthought. A results "
        "table with no limitations section is an unfinished deliverable."
    )
