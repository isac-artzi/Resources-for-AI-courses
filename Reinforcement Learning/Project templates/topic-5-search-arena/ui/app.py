"""
ui/app.py — the presentation tier. The only thing a non-technical stakeholder
ever needs to open.

Three rules this file obeys, and yours must too:

  1. It contains NO search code and NO training code. Every move comes back
     from the service tier through ui/service.py. The "Play" tab does not
     import `search/`; if it did, the agent a human plays would be the UI's
     copy of the agent rather than the deployed one, and the win rates in the
     "Tournament" tab would describe different software from the game on the
     screen.
  2. It never issues SQL that changes state. Read-only views only, through the
     anon key.
  3. It degrades visibly. A paused database or a missing artifact produces a
     clearly worded panel, never a stack trace. Supabase free-tier projects
     pause after a week idle, so this will happen to you — probably the night
     before a deadline.

One exception to rule 1, and it is worth naming because it looks like a
violation: the board itself lives in `st.session_state` and the human's own move
is applied to it locally. The GAME STATE is presentation state; the DECISIONS
are not. Every column the agent drops is a `/act` response, and whether the game
has ended is also the service's answer — see `drop` and the 422 handling below.

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

ROWS, COLS = 6, 7
EMPTY, YELLOW, RED = 0, 1, -1
GLYPH = {EMPTY: "·", YELLOW: "🟡", RED: "🔴"}

st.set_page_config(page_title="Search Arena", layout="wide")


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

st.title("Search Arena — Connect Four")
st.caption(
    "Three search strategies and a learned one, playing the same game behind "
    "one API, under an enforced per-request node budget. Every number on this "
    "page came back from the service tier; none of it was computed here."
)


# ---------------------------------------------------------------------------
# Board helpers. Presentation only — no game rules beyond gravity.
# ---------------------------------------------------------------------------
#
# The board is held the way the WIRE holds it: a flat 42-list with row 0 at the
# bottom, plus the side to move. Keeping the UI in the same coordinate system as
# the API means there is exactly one flip in this file — in `board_grid`, for
# display — rather than one per function.


def new_board() -> list[float]:
    return [0.0] * (ROWS * COLS) + [float(YELLOW)]


def board_grid(state: list[float]) -> list[list[int]]:
    """Top row first, the way a human reads it."""
    return [[int(state[r * COLS + c]) for c in range(COLS)]
            for r in range(ROWS - 1, -1, -1)]


def drop(state: list[float], col: int) -> list[float] | None:
    """Apply a move locally. Returns None if the column is full.

    This is the ONLY piece of game logic in the presentation tier, and it is
    deliberately the smallest possible piece: gravity. It does not decide who
    won, it does not decide whether the game is over, and it does not choose a
    move. Those all come from the service, which is why a disagreement between
    this file and the engine can only ever be a display bug.
    """
    state = list(state)
    for r in range(ROWS):
        if state[r * COLS + col] == 0.0:
            state[r * COLS + col] = state[-1]
            state[-1] = -state[-1]
            return state
    return None


def render_board(state: list[float]) -> None:
    lines = ["".join(f"{GLYPH[v]} " for v in row) for row in board_grid(state)]
    st.markdown("  \n".join(lines))
    st.caption("columns:  " + "   ".join(str(c) for c in range(COLS)))


TABS = st.tabs(
    ["Concepts", "Play", "Tournament", "Scalability", "Revision",
     "Run History", "Model Card"]
)


# ---------------------------------------------------------------------------
with TABS[0]:
    st.header("Concepts")
    st.markdown(
        """
This tab is the **Theoretical Brief**: 350–600 words explaining the mathematics
in language a non-specialist colleague could follow. Replace this scaffold with
your own words — the headings below are the argument the rest of the app makes,
and they are the ones the rubric is looking for.

### One recursion, two strategies

Exhaustive search and heuristic search are not two algorithms. They are one
algorithm — depth-limited negamax — under two policies for **which child to look
at next**. In this repository that is literally true: `search/minimax.py`
contains a single recursion, and the two baseline agents differ only in the
ordering callback they are handed.
        """
    )
    st.latex(r"v(s) \;=\; \max_{a \in A(s)} \; \bigl(-\,v(\mathrm{push}(s, a))\bigr)")
    st.markdown(
        """
Negamax rather than a `min`/`max` pair, because Connect Four is zero-sum:
min(a, b) = −max(−a, −b). Every value is from the point of view of the side to
move, which halves the code and removes the sign error that a copy-pasted `min`
node otherwise introduces.

### What alpha–beta actually buys

Pruning cannot change the value; it changes how much of the tree has to be
looked at to be sure of it. With perfect move ordering the node count falls from
b^d to about b^(d/2) — the same as **doubling the reachable depth for the same
budget**. Ordering with pruning switched off changes nothing at all, and the
"Scalability" tab shows both curves so you can see that for yourself.

### Upper confidence bounds for trees

MCTS needs no evaluation function. It estimates a position's value by playing it
out, and concentrates its effort with the UCT rule:
        """
    )
    st.latex(
        r"\mathrm{UCT}_i \;=\; \underbrace{\frac{W_i}{N_i}}_{\text{exploit}}"
        r"\;+\; C\,\underbrace{\sqrt{\frac{\ln N}{N_i}}}_{\text{explore}}"
    )
    st.markdown(
        """
The exploration term decays as N_i grows and grows as the parent is visited, so
a rarely-tried move keeps getting a chance until the evidence against it is
strong. C sets the exchange rate between the two terms and is **problem
dependent**: √2 is the value UCB1's regret bound is derived for when rewards lie
in [0, 1], and this game's rewards lie in [−1, 1], so it is a hyperparameter to
sweep rather than a constant to trust.

### And what AlphaZero changes

Two things, neither of which needs a data centre to demonstrate: replace the
random playout with a **learned value**, and replace UCT's uniform exploration
term with a **learned prior** (PUCT). The network is then trained on the
search's own visit distribution — the search plays better than the network
alone, so its output is a training target for free. That loop is the direct
ancestor of search at inference time in today's reasoning systems.
        """
    )
    st.info(
        "Write for the stakeholder in the product brief, not for the grader. "
        "If a sentence would not survive being read aloud in a meeting, cut it."
    )


# ---------------------------------------------------------------------------
with TABS[1]:
    st.header("Play")
    st.caption(
        "You are 🟡 and move first. Every agent reply is a `POST /act` — the "
        "node count and the time below are what the service reported, not what "
        "this page measured."
    )

    try:
        agent_info = service.agents()
        available = [a["name"] for a in agent_info["agents"]]
        descriptions = {a["name"]: a["description"] for a in agent_info["agents"]}
    except service.ServiceError as exc:
        st.error(str(exc))
        available, descriptions = [], {}

    if "board" not in st.session_state:
        st.session_state.board = new_board()
        st.session_state.log = []
        st.session_state.finished = None

    c1, c2, c3 = st.columns([2, 1, 1])
    opponent = c1.selectbox("Opponent", available or ["heuristic"])
    budget = c2.select_slider(
        "Node budget",
        options=[100, 1_000, 10_000, 50_000, 200_000],
        value=10_000,
        help=(
            "The per-request ceiling the service enforces. Turn it down and "
            "watch the agent get visibly worse — that is the deployment "
            "constraint this whole product is built around, made playable."
        ),
    )
    if c3.button("New game", use_container_width=True):
        st.session_state.board = new_board()
        st.session_state.log = []
        st.session_state.finished = None

    if opponent in descriptions:
        st.caption(descriptions[opponent])

    render_board(st.session_state.board)

    if st.session_state.finished:
        st.success(f"Game over — {st.session_state.finished}")
    else:
        cols = st.columns(COLS)
        for c in range(COLS):
            full = st.session_state.board[(ROWS - 1) * COLS + c] != 0.0
            if cols[c].button(f"↓ {c}", key=f"drop{c}", disabled=full,
                              use_container_width=True):
                after_human = drop(st.session_state.board, c)
                if after_human is None:
                    st.warning("That column is full.")
                else:
                    st.session_state.board = after_human
                    try:
                        reply = service.act_agent(after_human, opponent,
                                                  node_budget=budget)
                    except service.ServiceError as exc:
                        # The service returns 422 "already over" when the human's
                        # move ended the game. That is the same 422 any caller
                        # gets, and it is how this tab learns the result — the
                        # rules are not reimplemented here, on purpose. Two
                        # implementations of four-in-a-row eventually disagree,
                        # and the one in the least-tested tier is the one that
                        # will be wrong.
                        if "already over" in str(exc):
                            st.session_state.finished = "you won, or the board filled"
                        else:
                            st.error(str(exc))
                    else:
                        st.session_state.log.append(reply)
                        moved = drop(st.session_state.board, int(reply["action"]))
                        if moved is not None:
                            st.session_state.board = moved
                        if reply.get("budget_exhausted"):
                            st.warning(
                                f"`{opponent}` hit its node budget of "
                                f"{reply['node_budget']:,} and returned a truncated "
                                "answer. It is playing weaker than it can — which "
                                "is exactly what a loaded free-tier instance does "
                                "to a search agent."
                            )
                    st.rerun()

    if st.session_state.log:
        last = st.session_state.log[-1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Nodes expanded", f"{last['nodes_expanded']:,}")
        m2.metric("Time", f"{last['latency_ms']:.0f} ms")
        m3.metric("Search depth", last["search_depth"])
        m4.metric("Position value", f"{last['value_estimate']:.1f}")
        st.caption(
            f"agent config `{last['policy_sha256'][:12]}…` · budget "
            f"{last['node_budget']:,} · legal columns {last['legal_moves']}"
        )
        st.subheader("Every move this agent has made")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "move": i + 1,
                        "column": r["action"],
                        "nodes": r["nodes_expanded"],
                        "ms": round(r["latency_ms"], 1),
                        "depth": r["search_depth"],
                        "value": round(r["value_estimate"], 1),
                        "truncated": r["budget_exhausted"],
                    }
                    for i, r in enumerate(st.session_state.log)
                ]
            ),
            use_container_width=True, hide_index=True,
        )


# ---------------------------------------------------------------------------
with TABS[2]:
    st.header("Tournament")
    st.markdown(
        "Round-robin results from the `games` table, aggregated into `matches`. "
        "**Draws count as half a win** — that definition lives in "
        "`db/migrations/002_topic5.sql`, in `train/benchmark.py` and here, and "
        "the three must not drift."
    )
    try:
        data = service.tournament()
    except service.ServiceError as exc:
        st.error(str(exc))
        data = {"matches": [], "agents": [], "source": "none", "degraded": True,
                "total_games": 0}

    if data["source"] == "none":
        st.info(
            "No tournament has been run yet. `python -m train.benchmark --games 200` "
            "writes the rows this table reads."
        )
    else:
        if data["degraded"]:
            st.warning(
                "These numbers came from the checked-in `reports/benchmark.json`, "
                "not from a live query, and may be stale. A table that silently "
                "falls back to a file is how an old result gets presented as a "
                "current one, so this banner is not optional."
            )
        df = pd.DataFrame(data["matches"])
        if not df.empty:
            st.subheader("Win-rate matrix")
            matrix = df.pivot(index="agent", columns="opponent", values="win_rate")
            st.dataframe(
                matrix.style.format("{:.3f}", na_rep="—")
                .background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
                use_container_width=True,
            )
            st.caption(
                f"{data['total_games']} games total. A win rate over 30 games has "
                "a standard error of about 9 percentage points; say which budget "
                "produced the table you are quoting."
            )

            st.subheader("Cost per decision")
            st.caption(
                "Node counts are machine-independent and transfer to a reviewer's "
                "laptop; milliseconds do not. Quote both."
            )
            cost = (
                df[df["mean_nodes"] > 0]
                .groupby("agent")[["mean_nodes", "mean_ms"]]
                .mean()
                .sort_values("mean_nodes")
            )
            st.dataframe(cost.style.format("{:,.1f}"), use_container_width=True)

            st.subheader("Decision quality against the fixed reference")
            vs_random = df[df["opponent"] == "random"][
                ["agent", "games", "wins", "draws", "losses", "win_rate", "mean_nodes"]
            ].sort_values("win_rate", ascending=False)
            if vs_random.empty:
                st.info("No games against `random` in this run.")
            else:
                st.dataframe(vs_random, use_container_width=True, hide_index=True)
                st.caption(
                    "`random` never improves, so this column is comparable across "
                    "weeks. A win rate against 'my previous agent' is a moving "
                    "target and cannot be plotted against anything."
                )


# ---------------------------------------------------------------------------
with TABS[3]:
    st.header("Scalability")
    st.markdown(
        "Nodes expanded against search depth, per variant. This is the chart "
        "that makes the topic: b^d and b^(d/2) read the same in a sentence and "
        "nothing like each other on a chart."
    )
    try:
        data = service.scalability()
    except service.ServiceError as exc:
        st.error(str(exc))
        data = {"points": [], "source": "none", "degraded": True}

    if not data["points"]:
        st.info("Run `python -m train.benchmark` to produce the sweep.")
    else:
        if data["degraded"]:
            st.warning("Served from the checked-in report rather than a live query.")
        df = pd.DataFrame(data["points"])
        truncated = df[~df["completed"]]
        df = df[df["completed"]]

        st.subheader("Nodes expanded")
        st.line_chart(
            df.pivot_table(index="depth", columns="variant", values="nodes",
                           aggfunc="mean"),
            height=320,
        )
        st.caption(
            "Read the gap between `exhaustive` and `heuristic+ab` at each depth: "
            "that is what ordering plus pruning bought. `beam3` is below both and "
            "is **unsound** — it can drop the winning move, which is why it is "
            "drawn here and not recommended."
        )

        st.subheader("Wall clock (ms)")
        st.line_chart(
            df.pivot_table(index="depth", columns="variant", values="wall_clock_ms",
                           aggfunc="mean"),
            height=280,
        )

        st.subheader("Growth factor per extra ply")
        # The EMPIRICAL branching factor, nodes(d) / nodes(d-1). A full-width
        # search should sit near b = 7; alpha-beta with good ordering should
        # approach sqrt(b) ~ 2.65. That comparison is the entire theoretical
        # claim of this topic, expressed as one column of measured numbers
        # rather than as an asymptotic aside.
        growth = []
        for variant, g in df.groupby("variant"):
            g = g.sort_values("depth")
            ratios = (g["nodes"] / g["nodes"].shift(1)).dropna()
            growth.append({"variant": variant,
                           "mean nodes(d)/nodes(d-1)": round(float(ratios.mean()), 2)})
        st.dataframe(pd.DataFrame(growth), use_container_width=True, hide_index=True)
        st.caption(
            "A full-width search should sit near b = 7. Alpha-beta with good "
            "ordering approaches √b ≈ 2.65 — the classic result, measured rather "
            "than quoted."
        )

        if not truncated.empty:
            st.warning(
                f"{len(truncated)} configurations exceeded the probe's node budget "
                "and are excluded rather than plotted at their truncated count. "
                "Plotting them would put a fake plateau exactly where the "
                "interesting growth is."
            )
            st.dataframe(truncated[["variant", "depth"]], use_container_width=True,
                         hide_index=True)


# ---------------------------------------------------------------------------
with TABS[4]:
    st.header("Revision — original versus revised, side by side")
    st.markdown(
        """
The revised agent is **`mcts_v2`**. Two changes to `mcts`, both proposed from
benchmark evidence rather than from taste:

1. **Tactical playouts.** The original simulates with uniform random moves,
   which walk past a win roughly half the time — so a won position is scored as
   a loss and the estimate is mostly noise. `mcts_v2` takes a win and blocks a
   loss inside the playout.
2. **A lower exploration constant** (1.414 → 0.9). A less noisy value estimate
   needs less exploration before it can be trusted; keeping the original C
   spends simulations re-checking children the better estimate had settled.

**The trade, named.** Each simulation costs roughly 2–3x more, so at a fixed
wall-clock budget the revised agent runs fewer of them. That is a real cost, and
it is why this is a trade rather than a free improvement.

**Why these metrics.** Win rate says whether it plays better. Milliseconds per
decision says what that cost the user. Nodes per decision says whether the
improvement came from thinking *better* or merely from thinking *more* — without
that third column, any revision that quietly raised the compute budget would
look like a better algorithm.
        """
    )
    try:
        data = service.tournament()
        df = pd.DataFrame(data["matches"])
    except service.ServiceError as exc:
        st.error(str(exc))
        df = pd.DataFrame()

    if df.empty:
        st.info("Run `python -m train.benchmark --agents all` to populate this tab.")
    else:
        pair = df[df["agent"].isin(["mcts", "mcts_v2"])]
        head_to_head = pair[(pair["agent"] == "mcts_v2") & (pair["opponent"] == "mcts")]
        if not head_to_head.empty:
            row = head_to_head.iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("Revised vs original", f"{row['win_rate']:.3f}",
                      help="Win rate of mcts_v2 against mcts, head to head.")
            m2.metric("Games", int(row["games"]))
            m3.metric("Mean nodes / decision", f"{row['mean_nodes']:,.0f}")

        st.subheader("Both agents, against every opponent")
        st.dataframe(
            pair[["agent", "opponent", "games", "win_rate", "mean_nodes", "mean_ms"]]
            .sort_values(["opponent", "agent"]),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "A revision that improves one metric at the cost of another is "
            "acceptable and often more interesting — but the trade has to be "
            "named and defended, which is what the paragraph above is for."
        )


# ---------------------------------------------------------------------------
with TABS[5]:
    st.header("Run history")
    st.caption("Read-only. This tab issues no writes and holds no service-role key.")
    try:
        data = service.runs(100)
        if data.get("degraded"):
            st.warning(
                "The data tier did not answer, so this table is empty rather than "
                "complete."
            )
        elif data["runs"]:
            st.dataframe(pd.DataFrame(data["runs"]), use_container_width=True,
                         height=380)
        else:
            st.info("Nothing logged yet. `python -m train.train` writes the first row.")
    except service.ServiceError as exc:
        st.error(str(exc))

    st.subheader("Registered artifacts")
    try:
        st.dataframe(pd.DataFrame(service.policies()["policies"]),
                     use_container_width=True)
    except service.ServiceError as exc:
        st.error(str(exc))

    st.subheader("Agents on this service")
    try:
        info = service.agents()
        st.dataframe(
            pd.DataFrame([{"name": a["name"], "description": a["description"],
                           "config": str(a["config"])} for a in info["agents"]]),
            use_container_width=True, hide_index=True,
        )
        for note in info.get("unavailable", []):
            st.warning(note)
    except service.ServiceError as exc:
        st.error(str(exc))


# ---------------------------------------------------------------------------
with TABS[6]:
    st.header("Model card")
    st.markdown(
        """
**What this agent does.** _One paragraph, in the stakeholder's language._

**What it does not do.** _State the boundary explicitly. This one is easy to get
wrong: none of these agents plays Connect Four perfectly. The game is solved — a
first-player win with correct play, starting in the centre column — and nothing
here searches 42 plies._

**Training data and environment.** _Environment id, self-play game count, seeds,
and the simulation budget the network was trained under._

**Evaluation.** _Win rate against the fixed reference with a standard error, the
head-to-head matrix, and the number of games behind each cell._

**Limitations.** _At least four, each with how you would test whether it binds.
Starting points: the static evaluation function is hand-written and was never
tuned; the node budget makes every agent weaker under load in a way the
tournament did not measure; the learned network saw far fewer positions than its
architecture can absorb; the reported win rates have standard errors wider than
several of the differences between agents._

**Foreseeable misuse and reward-specification risk.** _What goes wrong if someone
optimises this harder than you did? For a game agent the honest answer is
usually "not much" — which makes it worth thinking about the DECISION-SUPPORT
version of the same system, where a search that is confidently wrong past its
horizon looks exactly like one that is right._

**Responsible disclosure.** _If you found a failure mode, who would you tell,
when, and what would you say?_
        """
    )
    st.caption(
        "This section is part of the rubric, not an afterthought. A results "
        "table with no limitations section is an unfinished deliverable."
    )
