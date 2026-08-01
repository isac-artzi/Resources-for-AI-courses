"""
ui/app.py — the presentation tier for Lake Pilot. The only thing the client
ever needs to open.

Three rules this file obeys, and yours must too:

  1. It contains NO policy code and NO training code. Every decision, every
     rollout and every learning curve comes back from the service tier through
     ui/service.py. The only thing it imports from `envs` is the MAP — which
     squares are ice and which are holes — because painting the board is
     presentation, and reading it from the same source the agent walks is what
     stops the picture and the environment drifting apart.
  2. It never issues SQL that changes state. Read-only views only.
  3. It degrades visibly. A paused database or a missing artifact produces a
     clearly worded panel, never a stack trace. Supabase free-tier projects
     pause after a week idle, so this will happen to you — probably the night
     before a deadline.

Run it:  streamlit run ui/app.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import pandas as pd
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from envs import ACTION_ARROWS, ACTION_NAMES, MAP_NAME, lake_rows  # noqa: E402
from shared.config import get_settings          # noqa: E402
from ui import service                          # noqa: E402

settings = get_settings()

st.set_page_config(page_title="Lake Pilot", layout="wide")

# The name of the all-zero artifact `train/train.py` exports so this tab has an
# untrained agent to contrast against. A constant rather than a string typed in
# three places: renaming it in train/ and not here produces a "Watch" tab that
# quietly serves the trained policy for both sides of the comparison, which
# reads as the trained agent failing rather than as a bug.
UNTRAINED_POLICY = "untrained_policy"


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

st.title("Lake Pilot")
st.caption(
    "An agent that learns to cross a frozen lake it cannot walk across in a "
    "straight line. Trained offline, served as a NumPy artifact behind a typed "
    "HTTP contract, with every training episode kept as a row you can query."
)

TABS = st.tabs(["Concepts", "Watch", "Compare", "Run History", "Model Card"])


# ---------------------------------------------------------------------------
# CONCEPTS — the Theoretical Brief, mirrored from the README.
# ---------------------------------------------------------------------------
with TABS[0]:
    st.header("What the agent is actually doing")

    st.subheader("The agent–environment loop")
    st.markdown(
        """
Everything in reinforcement learning is one loop. At each step the agent
observes a **state** $s_t$, chooses an **action** $a_t$, and the environment
answers with a **reward** $r_{t+1}$ and a new state $s_{t+1}$. There is no
labelled example anywhere: the only feedback is the reward, it arrives late,
and it does not say which of the last thirty decisions earned it.

On this lake the state is *which of the 64 squares you are standing on*, the
action is one of four directions, and the reward is **1.0 for reaching the goal
and 0.0 for everything else** — including falling in a hole. That last detail is
worth sitting with: the environment never punishes the agent for drowning.
Drowning is bad only because it ends the episode before the 1.0 can be
collected.
        """
    )
    st.latex(r"s_t \;\xrightarrow{\;\pi\;}\; a_t \;\xrightarrow{\;P(s'\mid s,a)\;}\; (r_{t+1},\; s_{t+1})")
    st.info(
        "The ice is slippery: the intended move happens with probability 1/3, "
        "and each perpendicular move with probability 1/3. That is why the "
        "answer is a policy — an action for every square — and not a route."
    )

    st.subheader("The Bellman equation")
    st.markdown(
        "The value of a state is the reward you expect now plus the discounted "
        "value of wherever you end up. Written for a policy $\\pi$:"
    )
    st.latex(r"v_\pi(s) = \sum_a \pi(a\mid s) \sum_{s'} P(s'\mid s,a)\big[R(s,a,s') + \gamma\, v_\pi(s')\big]")
    st.markdown(
        "The optimal action-value function satisfies the same identity with a "
        "maximum where the policy average was, and **that** is the equation this "
        "product implements:"
    )
    st.latex(r"q_*(s,a) = \sum_{s'} P(s'\mid s,a)\Big[R + \gamma \max_{a'} q_*(s',a')\Big]")
    st.markdown(
        "Q-learning never sees $P(s' \\mid s,a)$. It replaces the sum over next "
        "states with a single sampled transition and nudges its estimate towards it:"
    )
    st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha\Big[\underbrace{r + \gamma \max_{a'} Q(s',a') - Q(s,a)}_{\text{temporal-difference error}}\Big]")
    st.markdown(
        """
* $\\alpha$ — how much of the new estimate replaces the old. Large, on a
  slippery lake, means chasing individual lucky episodes.
* $\\gamma$ — the discount. Since the only reward is at the goal, $\\gamma$ is
  the entire reason a shorter crossing is worth more than a longer one, and it
  is what carries value backwards from the goal across sixty-four squares.
* The bracket is the **temporal-difference error**: the gap between what the
  agent believed and what one step of experience just suggested. Learning here
  is nothing but repeatedly closing that gap.
        """
    )

    st.subheader("Exploration and exploitation")
    st.markdown(
        """
A greedy agent takes the best action it currently knows about. Early in
training it knows about nothing — it has never reached the goal, every entry of
$Q$ is zero, and its "best" action is an artefact of how ties are broken.
Exploiting that is how an agent spends twenty thousand episodes confidently
walking into the same wall.

$\\varepsilon$-greedy is the cheapest possible fix: with probability
$\\varepsilon$ act at random, otherwise act greedily.
        """
    )
    st.latex(r"\pi(a\mid s) = \begin{cases} 1-\varepsilon + \varepsilon/|A| & a = \arg\max_{a'} Q(s,a') \\ \varepsilon/|A| & \text{otherwise} \end{cases}")
    st.markdown(
        """
$\\varepsilon$ decays across training: high at the start when the estimates are
worthless, low at the end when they are not. **Exploration is not free.** Every
exploratory step is an action the deployed policy would never take, so the
training curve sits *below* the greedy performance for the whole decay. That is
why this product logs $\\varepsilon$ on every episode row and keeps greedy
evaluations in a separate table: the two curves answer different questions, and
reporting the first while claiming the second is the most common error in this
assignment.
        """
    )

    st.subheader("Where Q-learning sits in the taxonomy")
    st.markdown(
        """
| | **Value-based** | **Policy-based** |
|---|---|---|
| **Model-free** — no $P$, no $R$; learn from sampled experience | **Q-learning**, SARSA, DQN: learn $Q$, act greedily with respect to it | REINFORCE, PPO: adjust $\\pi$ directly, no value table required |
| **Model-based** — you have, or learn, $P$ and $R$ | Value iteration, policy iteration, Dyna-Q | Model-predictive control, AlphaZero-style planning |

**Model-free versus model-based.** A model-based method knows, or learns, the
rules — the probability of each outcome — and can plan by imagining. A
model-free method never represents the rules at all; it tries things and
averages. Lake Pilot is model-free *by choice*: Gymnasium would happily hand us
FrozenLake's exact transition table, and refusing to look at it is what makes
this a rehearsal for problems where nobody has one.

**Value-based versus policy-based.** A value-based method learns *how good*
each action is and derives behaviour by taking the maximum. A policy-based
method learns the behaviour directly. Value-based methods are sample-efficient
on small discrete action sets and become awkward when the action is a
continuous torque; policy-based methods handle that naturally and are noisier.

**Where each one earns its keep**

* *Model-free, value-based* — recommendation and ad ranking, elevator dispatch
  and traffic-signal timing, datacentre cooling setpoints. Discrete choices,
  cheap simulation, plenty of logged experience.
* *Model-free, policy-based* — robot locomotion and manipulation, and the RLHF
  stage of a language model, where the "action" is a continuous vector or an
  entire sentence and no maximum can be taken over it.
* *Model-based, value-based* — inventory and supply-chain policies, where the
  dynamics really are known and simulating a year is cheaper than living it.
* *Model-based planning* — board games and chemical process control: an
  accurate model exists, so searching forward beats guessing.

The through-line: **the choice is made by what you have, not by what is
fashionable.** You have no transition model and four discrete actions, so a
model-free value-based method is the right tool and a Q-table is its simplest
form.
        """
    )


# ---------------------------------------------------------------------------
# WATCH — the stakeholder's tab. This is the demo the client asked for.
# ---------------------------------------------------------------------------
def render_grid(rows: list[str], agent_state: int | None, last_action: int | None = None) -> str:
    """The lake as one HTML table, with the agent on it.

    Built as a single HTML string rather than as a Streamlit column layout
    because the animation replaces the whole board on every frame: rerunning a
    layout sixty times produces visible flicker and a growing widget tree, while
    one `st.empty()` holding one string does not.
    """
    ncol = len(rows[0])
    palette = {"S": "#dbeafe", "F": "#f8fafc", "H": "#1e293b", "G": "#bbf7d0"}
    glyph = {"S": "start", "F": "", "H": "hole", "G": "goal"}
    cells = []
    for r, row in enumerate(rows):
        tds = []
        for c, ch in enumerate(row):
            here = agent_state is not None and (r * ncol + c) == agent_state
            body = "🤖" if here else glyph[ch]
            colour = "#fde68a" if here else palette[ch]
            text = "#0f172a" if here or ch != "H" else "#64748b"
            tds.append(
                f'<td style="width:44px;height:44px;text-align:center;'
                f'border:1px solid #cbd5e1;background:{colour};color:{text};'
                f'font-size:12px">{body}</td>'
            )
        cells.append("<tr>" + "".join(tds) + "</tr>")
    arrow = "" if last_action is None else (
        f'<div style="margin-top:6px;font-size:14px">last action: '
        f"{ACTION_ARROWS[last_action]} {ACTION_NAMES[last_action]}</div>"
    )
    return f'<table style="border-collapse:collapse">{"".join(cells)}</table>{arrow}'


with TABS[1]:
    st.header("Watch the agent cross")

    if not health.get("policy_artifact_loaded"):
        st.error(
            "No policy artifact is loaded, so there is nothing to watch. Run "
            "`python -m train.train`, which trains the agent and writes both "
            "`policies/q_table.npz` and `policies/untrained_policy.npz`."
        )
    else:
        try:
            available = [p["name"] for p in service.policies()["policies"]]
        except service.ServiceError as exc:
            available = []
            st.error(str(exc))

        trained = [n for n in available if n != UNTRAINED_POLICY] or ["default"]

        col_a, col_b = st.columns([1, 3])
        with col_a:
            mode = st.radio(
                "Agent",
                ["Trained", "Untrained (random)"],
                help=(
                    "Both sides run through the same POST /rollout endpoint. "
                    "'Trained' evaluates the exported Q-table greedily; "
                    "'Untrained' samples from an all-zero table, which is a "
                    "uniform random policy."
                ),
            )
        with col_b:
            c1, c2, c3 = st.columns(3)
            if mode == "Trained":
                policy_name = c1.selectbox("Policy artifact", trained)
                deterministic = True
            else:
                policy_name = UNTRAINED_POLICY
                deterministic = False
                c1.write("Policy artifact")
                c1.code(UNTRAINED_POLICY)
                if UNTRAINED_POLICY not in available:
                    st.info(
                        f"`{UNTRAINED_POLICY}.npz` has not been exported yet — run "
                        "`python -m train.train`. Falling back to the trained "
                        "artifact sampled non-greedily, which is NOT the same thing: "
                        "a softmax over a trained row is a weak agent, not a random one."
                    )
                    policy_name = trained[0]
            episodes = c2.slider("Evaluation episodes", 1, 100, 20)
            seed = c3.number_input(
                "Seed", value=0, step=1,
                help="Name the seed or the result is not evidence.",
            )

        speed = st.slider("Animation delay (seconds per step)", 0.0, 0.5, 0.12, 0.02)

        if st.button("Run a rollout", type="primary"):
            try:
                r = service.rollout(
                    policy_name=policy_name,
                    episodes=int(episodes),
                    seed=int(seed),
                    deterministic=deterministic,
                )
            except service.ServiceError as exc:
                st.error(str(exc))
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Mean return", f"{r['mean_return']:.3f}",
                    help="Here the return IS the success rate: each episode scores 1.0 or 0.0.",
                )
                m2.metric(
                    "Standard error", f"{r['stderr_return']:.3f}",
                    help="s / sqrt(n). Quote this alongside the mean, never the mean alone.",
                )
                m3.metric("Mean episode length", f"{r['mean_length']:.1f}")
                st.caption(
                    f"artifact `{r['policy_sha256'][:12]}…` · {r['episodes']} episodes · "
                    f"seed {r['seed']} · {'greedy' if deterministic else 'sampled'}"
                )

                rows = lake_rows(MAP_NAME)
                board = st.empty()
                traj = r["trajectory"]
                if not traj:
                    board.warning("The service returned no trajectory to animate.")
                else:
                    # A transition records the state the agent acted FROM, so the
                    # square it finally landed on is not in the list. The last
                    # transition's reward is what says whether that square was the
                    # goal — hence the caption, rather than leaving the viewer to
                    # infer an outcome from a board frozen one step early.
                    for t in traj:
                        board.markdown(
                            render_grid(rows, int(t["state"][0]), int(t["action"])),
                            unsafe_allow_html=True,
                        )
                        time.sleep(speed)
                    outcome = (
                        "reached the goal" if traj[-1]["reward"] > 0
                        else ("fell in a hole" if traj[-1]["terminated"] else "ran out of steps")
                    )
                    st.caption(
                        f"Episode 1 of this rollout: {len(traj)} steps, {outcome}. "
                        "The board shows the square the agent acted from at each step; "
                        "the square it finished on is not a transition and so is not animated."
                    )
                    st.dataframe(pd.DataFrame(traj), use_container_width=True, height=240)

                st.bar_chart(pd.DataFrame({"return": r["returns"]}))
                st.caption(
                    "One bar per episode, each either 0 or 1. On a stochastic "
                    "environment a good policy still loses episodes — which is why "
                    "the number to quote is a mean with a standard error, not the "
                    "run that happened to work."
                )


# ---------------------------------------------------------------------------
# COMPARE — learning curves, a moving average, and a seed selector.
# ---------------------------------------------------------------------------
def curve_for(experiment_id: str) -> pd.DataFrame:
    """Fetch one run's episode rows. Cached per rerun by Streamlit's own cache.

    Cached because the Compare tab reads each selected run twice — once for the
    return curve and once for the epsilon curve — and every slider nudge reruns
    the whole script. Without the cache, dragging the moving-average slider
    issues one query per frame against a free-tier database.
    """
    payload = service.episodes(experiment_id)
    frame = pd.DataFrame(payload["points"])
    frame.attrs["degraded"] = bool(payload.get("degraded"))
    frame.attrs["truncated"] = bool(payload.get("truncated"))
    return frame


curve_for = st.cache_data(ttl=120, show_spinner=False)(curve_for)


with TABS[2]:
    st.header("Compare configurations")
    st.markdown(
        "Every claim of the form *A beats B* must name the number of independent "
        "seeds behind it. The curves below are drawn from the `episodes` table, "
        "one row per episode, so anything shown here is reproducible from a query "
        "rather than from a saved screenshot."
    )

    try:
        data = service.runs(200)
    except service.ServiceError as exc:
        st.error(str(exc))
        data = {"runs": [], "degraded": True}

    if data.get("degraded"):
        st.warning(
            "The data tier did not answer, so this table is empty rather than "
            "complete. Nothing below is missing because you have not trained — it "
            "is missing because the database is unreachable."
        )
    elif not data["runs"]:
        st.info(
            "No runs logged yet. Run `python -m train.train` — every episode of "
            "every run should land in `episodes`."
        )
    else:
        df = pd.DataFrame(data["runs"])

        st.subheader("Configurations on record")
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

        st.subheader("Learning curves")
        # The label carries algorithm, seed AND id, because two runs of
        # "q-learning seed 0" with different learning rates are usually the exact
        # comparison you are trying to make and are indistinguishable otherwise.
        labels = {
            f"{r['algorithm']} · seed {r['seed']} · {str(r['experiment_id'])[:8]}":
                r["experiment_id"]
            for r in data["runs"]
            if r.get("episodes_logged")
        }
        if not labels:
            st.info("Runs exist but none has episode rows. Check for telemetry warnings.")
        else:
            c1, c2 = st.columns([3, 1])
            chosen = c1.multiselect(
                "Runs to overlay — this is the seed selector",
                list(labels),
                default=list(labels)[:3],
                help="Overlay several seeds of one configuration to see how much of a gap is real.",
            )
            window = c2.slider(
                "Moving-average window", 1, 1000, 100, 10,
                help=(
                    "Per-episode return here is 0 or 1, so the unsmoothed curve is a "
                    "barcode. The moving average is what makes it readable — and it "
                    "is also what hides a collapse shorter than the window."
                ),
            )

            returns, epsilons = {}, {}
            for label in chosen:
                try:
                    pts = curve_for(labels[label])
                except service.ServiceError as exc:
                    st.error(f"{label}: {exc}")
                    continue
                if pts.attrs.get("degraded"):
                    st.warning(f"{label}: the data tier did not answer.")
                    continue
                if pts.attrs.get("truncated"):
                    st.warning(
                        f"{label}: more episodes exist than were returned — the curve "
                        "below stops early and is not the whole run."
                    )
                if pts.empty:
                    continue
                pts = pts.sort_values("episode_index").set_index("episode_index")
                returns[label] = pts["return"].rolling(window, min_periods=1).mean()
                if pts["epsilon"].notna().any():
                    epsilons[label] = pts["epsilon"]

            if returns:
                st.line_chart(pd.DataFrame(returns))
                st.caption(
                    f"Moving average over {window} episodes of TRAINING return. This is "
                    "not the greedy score: while epsilon is high the agent throws away a "
                    "share of its actions on purpose, so this curve sits below what the "
                    "deployed policy achieves. The greedy numbers live in `evaluations` "
                    "and in the table above."
                )
                if epsilons:
                    # Same x-axis, different question. Putting exploration directly
                    # under the curve is what turns "it got worse around episode
                    # 8,000" from an impression into an answerable question.
                    st.line_chart(pd.DataFrame(epsilons))
                    st.caption("Epsilon in force, per episode, read back from the same rows.")
            elif chosen:
                st.info("No episode rows came back for the selected runs.")


# ---------------------------------------------------------------------------
# RUN HISTORY
# ---------------------------------------------------------------------------
with TABS[3]:
    st.header("Run history")
    st.caption(
        "Read-only. This tab issues no writes and holds no service-role key — it "
        "reads the `run_summary` view, which is `experiments` joined to its episode "
        "counts and its latest evaluation."
    )
    try:
        data = service.runs(100)
        if data.get("degraded"):
            st.warning("The data tier did not answer. This is an outage, not an empty history.")
        elif data["runs"]:
            frame = pd.DataFrame(data["runs"])
            st.dataframe(frame, use_container_width=True, height=420)
            st.caption(
                f"{len(frame)} runs · {int(frame['episodes_logged'].sum()):,} episodes "
                "logged in total. Sort by any column."
            )
        else:
            st.info("Nothing logged yet.")
    except service.ServiceError as exc:
        st.error(str(exc))

    st.subheader("Registered artifacts")
    st.caption(
        "Size and SHA-256 of every loadable `.npz`. The checksum is what lets you "
        "attribute a decision recorded in `audit_log` to a specific artifact rather "
        "than to 'the model'."
    )
    try:
        st.dataframe(pd.DataFrame(service.policies()["policies"]), use_container_width=True)
    except service.ServiceError as exc:
        st.error(str(exc))


# ---------------------------------------------------------------------------
# MODEL CARD
# ---------------------------------------------------------------------------
with TABS[4]:
    st.header("Model card")
    st.markdown(
        """
**What this agent does.** _One paragraph in the client's language: it crosses a
specific 8×8 frozen lake, on ice that pushes it sideways, more often than not.
Put your measured greedy return here, with its standard error and its seed
count._

**What it does not do.** _It has never seen any other map. The policy is a
lookup table indexed by square number, so change the lake and every entry
becomes meaningless — not degraded, meaningless. It does not plan, it cannot
explain itself, and it has no way to tell you it is out of its depth._

**Training data and environment.** _`FrozenLake-v1`, 8×8, `is_slippery=True`.
State the episode count, the seeds, and the exact epsilon schedule._

**Evaluation.** _Greedy mean return over N episodes with a standard error, and
the number of independent seeds. If you quote the best seed, say so._

**Known statistical flaw.** _Q-learning bootstraps from a `max` over four noisy
estimates, and the maximum of noisy estimates is biased upwards. The value
estimate `/act` returns is therefore optimistic, especially early in training.
Double Q-learning is the standard remedy; this product does not implement it._

**Limitations.** _At least four, each with how you would test whether it binds._

**Foreseeable misuse and reward-specification risk.** _The reward is 1.0 for the
goal and 0.0 for everything else — including drowning. Nothing in the objective
says "avoid holes"; avoiding them is merely instrumental. Write down what the
analogous omission would be in a system that mattered._

**Responsible disclosure.** _If you found a failure mode, who would you tell,
when, and what would you say?_
        """
    )
    st.caption(
        "This section is part of the rubric, not an afterthought. A results table "
        "with no limitations section is an unfinished deliverable."
    )
