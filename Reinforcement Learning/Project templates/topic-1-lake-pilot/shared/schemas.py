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
    deterministic: bool = Field(
        default=True,
        description=(
            "Greedy (argmax) when true; sample from the policy when false. The "
            "'Watch' tab needs both: the trained agent is evaluated greedily, "
            "and the untrained agent is an all-zero Q-table sampled from — "
            "equal values make the softmax uniform, so the same contract "
            "serves a genuinely random policy without a special case in the "
            "service tier. NOTE that the sampled path draws from NumPy's "
            "global RNG, so `seed` reproduces the environment's slips but not "
            "the agent's choices. The greedy path is reproducible end to end; "
            "quote random-baseline numbers from `episodes`, not from here."
        ),
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


class EpisodePoint(BaseModel):
    """One point on a learning curve.

    Deliberately the same four fields as `EpisodeRow` below, because it IS a
    row of `episodes` on its way back out. A separate, narrower response type
    for the curve would be one more place for the column names to drift.
    """

    episode_index: int
    return_: float = Field(..., alias="return")
    length: int
    epsilon: float | None = None

    model_config = {"populate_by_name": True}


class EpisodesResponse(BaseModel):
    """The learning curve for ONE experiment, which is what the Compare tab plots.

    One experiment, not many: a curve averaged across seeds before it leaves
    the service tier cannot be un-averaged by the caller, and the seed selector
    in the UI is the whole point. The UI asks for each seed's curve separately
    and overlays them.
    """

    experiment_id: str
    points: list[EpisodePoint]
    count: int
    truncated: bool = Field(
        default=False,
        description=(
            "True when the run has more episodes than `limit` returned. A "
            "curve that silently stops at episode 5,000 of a 20,000-episode "
            "run looks like a run that stopped learning."
        ),
    )
    degraded: bool = Field(
        default=False,
        description="True when the data tier could not be reached. Render it, do not hide it.",
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
