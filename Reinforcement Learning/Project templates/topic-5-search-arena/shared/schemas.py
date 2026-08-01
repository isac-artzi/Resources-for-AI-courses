"""
shared/schemas.py — the contract between every pair of tiers.

Nothing crosses a tier boundary in this repository except an instance of a
model defined in this file or a row defined by a migration in db/migrations/.
That is the whole point of the file: the Streamlit tier, the service tier and
the test suite all import the SAME class, so a contract change that breaks a
caller breaks it at import time and in CI, not in front of a stakeholder.

Two conventions worth copying into your own work:

1.  Requests and responses are separate types. It is tempting to reuse one
    model for both; do not. The response almost always grows fields the
    request must never accept (`policy_sha256`, `latency_ms`), and a shared
    model quietly makes those settable by the caller.

2.  Every numeric field that has a legal range says so with `Field(...)`.
    FastAPI turns those into a 422 with a readable message. A stack trace is
    not an error contract.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Topic 5: the node budget, which is a contract term and not an implementation
# detail.
# ---------------------------------------------------------------------------
#
# A search agent's cost is not bounded by its inputs. Every other product in
# this course serves a fixed-size forward pass: an observation arrives, a
# handful of matrix multiplies happen, an action leaves, and the work is the
# same whatever the observation was. A tree search is different — the same
# 43-number board with `depth: 9` instead of `depth: 4` is roughly 2,000 times
# the work — so the bound has to be part of the request contract.
#
# MAX_NODE_BUDGET is the ceiling a caller may ASK for; Pydantic rejects anything
# above it with a 422 before a single node is expanded. DEFAULT_NODE_BUDGET is
# what an unqualified request gets. The number is calibrated in
# db/migrations/002_topic5.sql's terms: at roughly 40,000 nodes per second in
# pure Python, 200,000 nodes is about five seconds — long for a web request,
# short enough not to be killed by a free-tier proxy timeout of thirty.
#
# The budget also comes BACK in the response (`nodes_expanded`), because a limit
# the caller cannot observe is a limit the caller cannot plan around.

DEFAULT_NODE_BUDGET = 200_000
MAX_NODE_BUDGET = 2_000_000


# ---------------------------------------------------------------------------
# POST /act  — a state in, an action out. The RL analogue of /predict.
# ---------------------------------------------------------------------------


class ActRequest(BaseModel):
    """A single decision request.

    `state` is a list of floats even for the tabular topics, where it is a
    one-element list holding the discrete state index. Keeping one shape
    across all six products means the Streamlit "Watch" tab and the test
    suite do not change when the policy stops being a table.
    """

    state: list[float] = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Observation vector. For tabular environments, [state_index].",
    )
    policy_name: str = Field(
        default="default",
        max_length=64,
        description="Which registered artifact to evaluate. See GET /policies.",
    )
    deterministic: bool = Field(
        default=True,
        description="Greedy action if true; sample from the policy if false.",
    )
    # -- this topic's extension ------------------------------------------------
    # One optional field, not a second endpoint. `/act` still means exactly
    # "state in, action out"; naming an agent selects a SEARCH PROCEDURE instead
    # of an artifact, and the state is then interpreted as a board rather than
    # as an observation vector. Forking this into `/search` would have doubled
    # the surface that every later change has to be applied to twice, and would
    # have given the Streamlit "Play" tab two clients to keep in step.
    agent: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Name of a search agent from GET /agents (e.g. 'heuristic', 'mcts'). "
            "When set it OVERRIDES policy_name, and `state` is read as a "
            "Connect Four board: 42 cells then the side to move."
        ),
    )
    node_budget: int = Field(
        default=DEFAULT_NODE_BUDGET,
        ge=1,
        le=MAX_NODE_BUDGET,
        description=(
            "Hard ceiling on nodes expanded for this request. Enforced by a "
            "counter inside the recursion, not by a depth cap — alpha-beta at a "
            "fixed depth can vary 30x in node count depending on move ordering, "
            "and a bound that varies 30x is not a bound. The search degrades to "
            "a static evaluation rather than raising when it runs out."
        ),
    )
    depth: int | None = Field(
        default=None,
        ge=1,
        le=12,
        description=(
            "Override the agent's configured search depth. Ignored by agents "
            "that are not depth-limited (MCTS, random). The upper bound is here "
            "because 12 is already far past what the node budget will fund; it "
            "turns a typo into a 422 instead of a five-second request."
        ),
    )


class ActResponse(BaseModel):
    action: int | list[float] = Field(
        ..., description="Discrete action index, or a continuous action vector."
    )
    policy_name: str
    policy_sha256: str = Field(
        ...,
        description=(
            "Checksum of the artifact that produced this action. Logged to "
            "audit_log so a past decision can be attributed to a specific "
            "artifact rather than to 'the model'. For a SEARCH agent there is no "
            "artifact, so this is the SHA-256 of the agent's configuration JSON "
            "— depth, ordering, exploration constant. A search agent's identity "
            "is its configuration in exactly the way a network's is its weights, "
            "and the audit log has to be able to tell a depth-4 answer from a "
            "depth-6 one six weeks later."
        ),
    )
    value_estimate: float | None = Field(
        default=None, description="max_a Q(s,a) where the policy exposes one."
    )
    latency_ms: float
    # -- search telemetry ------------------------------------------------------
    # Null for the artifact path, populated for the agent path. These are the
    # numbers the "Play" tab shows a human next to the board, and they are the
    # same numbers that land in `games`; they come from the same counter, so the
    # demo and the report cannot disagree.
    agent: str | None = None
    nodes_expanded: int | None = Field(
        default=None, description="Nodes actually expanded. Always <= node_budget."
    )
    node_budget: int | None = None
    budget_exhausted: bool = Field(
        default=False,
        description=(
            "True when the search stopped early because the budget ran out. The "
            "move returned is still legal but is a truncated answer, and the UI "
            "must say so — an agent that is silently weaker under load is worse "
            "than one that says it is."
        ),
    )
    search_depth: int | None = Field(
        default=None, description="Deepest ply actually reached."
    )
    legal_moves: list[int] = Field(
        default_factory=list,
        description="The legal-move mask, echoed so a caller can validate our answer.",
    )


# ---------------------------------------------------------------------------
# POST /rollout — bounded server-side evaluation. This is where the greedy
# numbers in your report come from; the training curve is a different thing.
# ---------------------------------------------------------------------------


class RolloutRequest(BaseModel):
    policy_name: str = Field(default="default", max_length=64)
    episodes: int = Field(
        default=20,
        ge=1,
        le=200,
        description=(
            "Hard-capped. An unbounded rollout endpoint is how a free-tier "
            "instance gets killed by its own users."
        ),
    )
    max_steps: int = Field(default=200, ge=1, le=2000)
    seed: int | None = Field(default=None, description="Name the seed or the result is not evidence.")
    record_trajectory: bool = Field(
        default=True, description="Return one full trajectory for the UI to animate."
    )


class Transition(BaseModel):
    step: int
    state: list[float]
    action: int | list[float]
    reward: float
    terminated: bool
    truncated: bool


class RolloutResponse(BaseModel):
    returns: list[float]
    mean_return: float
    std_return: float
    stderr_return: float = Field(
        ..., description="s / sqrt(n). Report this, not the mean alone."
    )
    mean_length: float
    episodes: int
    seed: int | None
    trajectory: list[Transition] = Field(default_factory=list)
    policy_name: str
    policy_sha256: str


# ---------------------------------------------------------------------------
# POST /game — a full, bounded game between two named agents, played server-side.
# ---------------------------------------------------------------------------
#
# This is the endpoint the product brief is really about: "play these two agents
# against each other and tell me who won and what it cost". It exists as a
# server-side endpoint rather than as a loop in Streamlit for the reason that
# governs every tier boundary in this repository — the presentation tier does
# not own decisions. A tournament run from the browser would produce numbers
# that describe the browser's copy of the agents.
#
# EVERY bound here is required, and each one closes a specific way to hang the
# instance: `max_moves` bounds the game, `node_budget` bounds each decision, and
# the two together bound the request. A game that cannot exceed 42 moves and 42
# budgeted searches has a computable worst case; the same endpoint without
# `node_budget` does not.


class GameRequest(BaseModel):
    agent_a: str = Field(..., max_length=64, description="Plays first (yellow).")
    agent_b: str = Field(..., max_length=64, description="Plays second (red).")
    node_budget: int = Field(
        default=DEFAULT_NODE_BUDGET,
        ge=1,
        le=MAX_NODE_BUDGET,
        description="Per-DECISION budget, applied to both agents identically.",
    )
    max_moves: int = Field(
        default=42,
        ge=1,
        le=42,
        description=(
            "Connect Four cannot exceed 42 moves, so this can only ever bind "
            "early. It is here anyway: a bound that is currently unreachable "
            "costs nothing and stops being unreachable the moment someone "
            "changes the board size, which is exactly what the scalability "
            "study invites them to do."
        ),
    )
    seed: int | None = Field(
        default=0, description="Name the seed or the result is not evidence."
    )
    record_moves: bool = True
    log_to_store: bool = Field(
        default=True,
        description="Write a `games` row. Turned off by the tests so a test run "
                    "does not contaminate the tournament table.",
    )


class MoveRecord(BaseModel):
    ply: int
    agent: str
    column: int = Field(..., ge=0)
    nodes_expanded: int
    wall_clock_ms: float
    search_depth: int
    value: float
    budget_exhausted: bool = False


class GameResponse(BaseModel):
    game_id: str
    agent_a: str
    agent_b: str
    result: Literal["agent_a", "agent_b", "draw"]
    winner: str | None = Field(
        default=None, description="Agent name, or null for a draw."
    )
    moves: int
    move_records: list[MoveRecord] = Field(default_factory=list)
    final_board: list[list[int]] = Field(
        default_factory=list, description="Top row first, the way a human reads it."
    )
    # Per-agent totals. Reported separately rather than summed, because the
    # whole comparison is between two agents whose costs differ by orders of
    # magnitude, and a combined total hides exactly the thing being measured.
    nodes_expanded: dict[str, int] = Field(default_factory=dict)
    wall_clock_ms: dict[str, float] = Field(default_factory=dict)
    mean_nodes_per_move: dict[str, float] = Field(default_factory=dict)
    max_search_depth: dict[str, int] = Field(default_factory=dict)
    node_budget: int = Field(
        ..., description="The per-decision ceiling this game was played under."
    )
    budget_exhausted_moves: int = Field(
        default=0,
        description=(
            "How many decisions in this game were truncated by the budget. A "
            "tournament where this is non-zero is a tournament of budget-limited "
            "agents, which is a different experiment from the one you meant to "
            "run — so it is reported per game rather than aggregated away."
        ),
    )
    seed: int | None = None
    logged: bool = False


# ---------------------------------------------------------------------------
# GET /agents — what the UI's picker and the benchmark harness both read.
# ---------------------------------------------------------------------------


class AgentInfo(BaseModel):
    name: str
    description: str
    config: dict[str, Any] = Field(default_factory=dict)
    config_sha256: str = Field(
        ...,
        description="Identity of this agent's configuration. See ActResponse.policy_sha256.",
    )


class AgentsResponse(BaseModel):
    agents: list[AgentInfo]
    count: int
    unavailable: list[str] = Field(
        default_factory=list,
        description=(
            "Agents that could not be constructed — in practice, the learned one "
            "before `python -m train.train` has been run. Named rather than "
            "silently omitted: 'the agent is missing' and 'the agent does not "
            "exist' are different problems with different fixes."
        ),
    )


# ---------------------------------------------------------------------------
# GET /tournament, GET /scalability — the evidence this product is judged on.
# ---------------------------------------------------------------------------


class MatchRow(BaseModel):
    agent: str
    opponent: str
    games: int
    wins: int
    draws: int
    losses: int
    win_rate: float
    mean_nodes: float
    mean_ms: float
    mean_peak_kib: float | None = None


class TournamentResponse(BaseModel):
    matches: list[MatchRow] = Field(default_factory=list)
    agents: list[str] = Field(default_factory=list)
    total_games: int = 0
    source: Literal["data-tier", "checked-in-report", "none"] = "none"
    degraded: bool = Field(
        default=False,
        description=(
            "True when these rows came from reports/benchmark.json rather than "
            "from a live query. The UI must say which — a table that silently "
            "falls back to a checked-in file is how a stale result gets "
            "presented as a current one."
        ),
    )


class ScalabilityPoint(BaseModel):
    variant: str
    depth: int
    nodes: int
    wall_clock_ms: float
    peak_kib: float | None = None
    completed: bool = Field(
        default=True,
        description="False when the configuration exceeded its node budget.",
    )


class ScalabilityResponse(BaseModel):
    points: list[ScalabilityPoint] = Field(default_factory=list)
    source: Literal["data-tier", "checked-in-report", "none"] = "none"
    degraded: bool = False


# ---------------------------------------------------------------------------
# GET /runs, GET /policies — the evidence tier, surfaced read-only.
# ---------------------------------------------------------------------------


class RunSummary(BaseModel):
    experiment_id: str
    algorithm: str
    env_id: str
    seed: int
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    git_sha: str | None = None
    created_at: str | None = None
    episodes_logged: int = 0
    mean_return_last_100: float | None = None
    eval_mean_return: float | None = None
    eval_stderr: float | None = None


class RunsResponse(BaseModel):
    runs: list[RunSummary]
    count: int
    degraded: bool = Field(
        default=False,
        description=(
            "True when the data tier could not be reached and this response is "
            "served from the local fallback. The UI must render this as a "
            "visible warning, never as an empty table."
        ),
    )


class PolicyArtifact(BaseModel):
    name: str
    format: Literal["npz", "npy", "json"] = "npz"
    bytes: int
    sha256: str
    experiment_id: str | None = None
    kind: Literal["tabular", "mlp", "linear-head", "value-net"] = "tabular"
    obs_dim: int | None = Field(
        default=None,
        description=(
            "Expected observation dimensionality. /act compares the request "
            "against this and returns 422 on a mismatch — see api/main.py."
        ),
    )
    n_actions: int | None = None


class PoliciesResponse(BaseModel):
    policies: list[PolicyArtifact]
    count: int


# ---------------------------------------------------------------------------
# GET /healthz, GET /version
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """200 only when BOTH dependencies are reachable.

    A health check that returns 200 because the process is running tells you
    nothing you did not already know from the fact that it answered.
    """

    status: Literal["ok", "degraded"]
    policy_artifact_loaded: bool
    data_tier_reachable: bool
    detail: str | None = None


class VersionResponse(BaseModel):
    app_version: str
    git_sha: str
    torch_imported: bool = Field(
        ...,
        description=(
            "Reported honestly at runtime, not just asserted in a test. If this "
            "is ever true in a deployed process, the build should not have shipped."
        ),
    )


# ---------------------------------------------------------------------------
# Rows written by the training tier. These mirror db/migrations/001_init.sql;
# if you change one, change the other in the same commit.
# ---------------------------------------------------------------------------


class EpisodeRow(BaseModel):
    experiment_id: str
    episode_index: int = Field(..., ge=0)
    return_: float = Field(..., alias="return")
    length: int = Field(..., ge=0)
    epsilon: float | None = Field(
        default=None,
        description=(
            "The exploration parameter IN FORCE at this episode. Logging it is "
            "what makes the exploration tax recomputable from your data later."
        ),
    )

    model_config = {"populate_by_name": True}


class EvaluationRow(BaseModel):
    experiment_id: str
    episodes: int
    mean_return: float
    std_return: float
    stderr_return: float
    at_training_episode: int


# ---------------------------------------------------------------------------
# Rows written by the benchmark harness. These mirror db/migrations/002_topic5.sql;
# if you change one, change the other in the same commit.
# ---------------------------------------------------------------------------


class GameRow(BaseModel):
    """One game. The atom of every claim in this product's write-up.

    Note that a game is logged ONCE, from `agent`'s point of view, with
    `opponent` naming the other side — not twice, once per side. Logging both
    directions doubles the row count and makes every aggregate quietly count
    each game twice unless the query remembers to halve it. The win-rate matrix
    view in the migration reads the single row from both directions instead,
    which puts that subtlety in one place where it can be reviewed.
    """

    game_id: str
    experiment_id: str | None = None
    agent: str
    opponent: str
    result: Literal["win", "loss", "draw"] = Field(
        ..., description="From `agent`'s point of view. Always. See the class docstring."
    )
    agent_played_first: bool = Field(
        ...,
        description=(
            "Who moved first, which in Connect Four is worth roughly a 5-point "
            "swing in win rate against a comparable opponent — the game is a "
            "first-player win under perfect play. A tournament that does not "
            "record this cannot tell a stronger agent from a luckier draw."
        ),
    )
    moves: int
    nodes_expanded: int = Field(..., description="Total across `agent`'s decisions only.")
    wall_clock_ms: float
    search_depth: int = Field(..., description="Deepest ply `agent` reached in this game.")
    peak_kib: float | None = Field(
        default=None,
        description=(
            "Peak tracemalloc allocation during a SINGLE decision by `agent`, "
            "maximised over this game, in KiB. Measured rather than asserted: "
            "'MCTS uses more memory' is a claim with a number attached only if "
            "something recorded the number. Per decision rather than per game, "
            "because a game-wide bracket attributes an expensive opponent's "
            "search tree to a cheap agent."
        ),
    )
    # The opponent's half of the same four measurements. Stored because the
    # win-rate matrix reads each game row from BOTH directions; without them
    # every cost cell in one triangle of the matrix is empty. See
    # db/migrations/002_topic5.sql.
    opponent_nodes_expanded: int | None = None
    opponent_wall_clock_ms: float | None = None
    opponent_search_depth: int | None = None
    opponent_peak_kib: float | None = None
    node_budget: int | None = None
    budget_exhausted_moves: int = 0
    seed: int | None = None


class MatchAggregateRow(BaseModel):
    """One ordered pairing, aggregated. Written by train/benchmark.py."""

    experiment_id: str | None = None
    agent: str
    opponent: str
    games: int
    wins: int
    draws: int
    losses: int
    win_rate: float
    mean_nodes: float
    mean_ms: float
    mean_peak_kib: float | None = None


class SearchProbeRow(BaseModel):
    """One scalability measurement: a single search at a single depth.

    A separate table from `games` rather than a game with null columns. A probe
    has no opponent, no result and no move count, so storing it in `games` would
    mean half the columns are NULL for half the rows and every query against
    that table has to know which kind of row it is looking at. Two small tables
    beat one table with a discriminator, when the two things genuinely are
    different things.
    """

    experiment_id: str | None = None
    variant: str = Field(..., description="e.g. 'exhaustive', 'heuristic+ab', 'beam3'")
    depth: int
    nodes: int
    leaves: int = 0
    cutoffs: int = 0
    wall_clock_ms: float
    peak_kib: float | None = None
    completed: bool = True
    position_label: str = Field(
        default="empty",
        description=(
            "Which position was searched. Node counts are meaningless without "
            "it: the empty board and a crowded midgame differ by an order of "
            "magnitude at the same depth."
        ),
    )
