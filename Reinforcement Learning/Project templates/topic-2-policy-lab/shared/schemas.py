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
# This topic's one contract extension: a request may name a POLICY SOURCE
# instead of an artifact.
# ---------------------------------------------------------------------------
#
# The product ships two agents that solve the same problem by different routes,
# and the client thinks in those terms — "what does the planner say, and what
# does the learner say?" — not in terms of artifact filenames. So `/act`
# accepts `policy_source`, and the service maps it onto the artifact name the
# base contract already understands.
#
# Note what this deliberately is NOT: a second endpoint, a second request model,
# or a `policy_name` field whose meaning changes depending on another field.
# The base contract is extended by one optional field with a closed set of
# values; every existing caller keeps working, `policy_name` still means
# exactly "which registered artifact", and the mapping between the two lives
# in ONE dictionary that both tiers import.
#
# `Literal` rather than `str` because an unknown source should be a 422 from
# FastAPI's validator with the legal values listed, not a 404 from the artifact
# store fifteen lines later.

PolicySource = Literal["value_iteration", "monte_carlo"]

POLICY_SOURCE_ARTIFACTS: dict[str, str] = {
    "value_iteration": "value_iteration",
    "monte_carlo": "monte_carlo",
}


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
    policy_source: PolicySource | None = Field(
        default=None,
        description=(
            "'value_iteration' (the exact plan) or 'monte_carlo' (the learned "
            "policy). When set it OVERRIDES policy_name — the caller has named "
            "the agent rather than the file, which is the more stable of the "
            "two references and the one the UI uses."
        ),
    )
    deterministic: bool = Field(
        default=True,
        description="Greedy action if true; sample from the policy if false.",
    )


class ActResponse(BaseModel):
    action: int | list[float] = Field(
        ..., description="Discrete action index, or a continuous action vector."
    )
    policy_name: str = Field(
        ...,
        description=(
            "The artifact that was actually used, AFTER policy_source was "
            "resolved. Echoing the resolved name rather than the requested one "
            "is what lets a caller confirm the routing did what they meant."
        ),
    )
    policy_source: PolicySource | None = None
    policy_sha256: str = Field(
        ...,
        description=(
            "Checksum of the artifact that produced this action. Logged to "
            "audit_log so a past decision can be attributed to a specific "
            "artifact rather than to 'the model'."
        ),
    )
    value_estimate: float | None = Field(
        default=None, description="max_a Q(s,a) where the policy exposes one."
    )
    latency_ms: float


# ---------------------------------------------------------------------------
# POST /rollout — bounded server-side evaluation. This is where the greedy
# numbers in your report come from; the training curve is a different thing.
# ---------------------------------------------------------------------------


class RolloutRequest(BaseModel):
    policy_name: str = Field(default="default", max_length=64)
    policy_source: PolicySource | None = Field(
        default=None,
        description=(
            "Same override as on /act. Both endpoints accept it so the UI can "
            "hold one 'which agent?' control and pass it everywhere; an "
            "endpoint that took only one of the two references would force the "
            "presentation tier to know the mapping."
        ),
    )
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


# ---------------------------------------------------------------------------
# GET /value_map, GET /convergence — the two views this topic's UI needs.
# ---------------------------------------------------------------------------
#
# They are endpoints rather than something the Streamlit app computes because
# of the rule that survives every topic: the presentation tier never opens a
# policy artifact and never re-derives a number the service already owns. A
# heat map drawn from a value function the UI recomputed is a heat map of the
# UI's arithmetic, which is not what anyone is being asked to review.


class ValueGrid(BaseModel):
    """One value function, shaped for a heat map.

    `values` is row-major and `rows * cols == len(values)`; that invariant is
    stated here because the difference between a value function and its
    transpose is invisible on a square grid and fatal to the conclusion.
    """

    label: str
    rows: int
    cols: int
    values: list[float]
    policy_arrows: list[str] = Field(
        default_factory=list,
        description="Greedy action per cell as an arrow, or '*' for a terminal cell.",
    )


class ValueMapResponse(BaseModel):
    grids: list[ValueGrid]
    difference: ValueGrid | None = Field(
        default=None,
        description=(
            "learned minus exact, per cell. Shipped as its own grid because "
            "'these two heat maps look similar' is not a measurement, and the "
            "difference map is where the disagreement actually shows."
        ),
    )
    gamma: float
    missing: list[str] = Field(
        default_factory=list,
        description="Policy sources with no exported artifact yet. Named, not silently omitted.",
    )


class ConvergencePoint(BaseModel):
    episodes: int
    seeds: int
    mean_rmse: float
    ci95_low: float
    ci95_high: float
    equivalent_at_5pct: bool = False


class ConvergenceResponse(BaseModel):
    points: list[ConvergencePoint]
    delta: float | None = None
    episodes_to_indistinguishable: int | None = None
    test: str | None = None
    source: Literal["data-tier", "checked-in-report", "none"] = "none"
    degraded: bool = Field(
        default=False,
        description=(
            "True when these numbers came from the checked-in report rather "
            "than from a live query. The UI must say which — a chart that "
            "silently falls back to a file is how a stale result gets "
            "presented as a current one."
        ),
    )


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
            "what makes the exploration tax recomputable from your data later. "
            "NULL for both agents in this topic: the planner does not explore "
            "at all, and exploring starts is not an epsilon-greedy schedule. "
            "0.0 would read as 'greedy', which is a different claim."
        ),
    )
    bellman_residual: float | None = Field(
        default=None,
        description=(
            "max_s |V_{k+1}(s) - V_k(s)| for a value-iteration SWEEP. Added by "
            "db/migrations/002_topic2.sql. NULL for a sampled episode, which is "
            "how a reader tells a planner row from a learner row without "
            "joining back to experiments."
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
    metric: Literal["return", "value_rmse"] = "return"
    rmse: float | None = Field(
        default=None,
        description="Set when metric == 'value_rmse'. Distance from the exact solution.",
    )
    policy_source: PolicySource | None = None
