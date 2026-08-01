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
# Topic 3 adds one row type. `episodes` answers "did it learn"; this answers
# "why was it noisy", which is the question the product brief actually asks.
# Mirrors db/migrations/002_gradient_stats.sql — change both in one commit.
# ---------------------------------------------------------------------------


class GradientStatRow(BaseModel):
    """One row per POLICY UPDATE, not per episode.

    The distinction matters and is the first thing to get right. A batch of ten
    episodes produces ten `episodes` rows and exactly one `gradient_stats` row,
    because the gradient estimate whose variance you are measuring is the
    batch's, not any single episode's. Logging it per episode would give you a
    number that cannot be compared across arms with different batch sizes.
    """

    experiment_id: str
    update_index: int = Field(..., ge=0, description="0-based index of the policy update.")
    episode_index: int = Field(
        ...,
        ge=0,
        description=(
            "Training episodes consumed BEFORE this update. Lets the gradient "
            "variance chart share an x-axis with the learning curve, which is "
            "the comparison the headline chart of this product is making."
        ),
    )
    gradient_norm: float = Field(
        ...,
        description="L2 norm of the BATCH-MEAN gradient — the step actually taken.",
    )
    gradient_variance: float = Field(
        ...,
        ge=0.0,
        description=(
            "Trace of the covariance of the per-episode gradient estimates: "
            "sum over parameters of Var_i(g_i). This is the quantity every "
            "technique in this topic exists to reduce, so it is stored rather "
            "than recomputed — a plot you cannot regenerate from the database "
            "is not evidence."
        ),
    )
    policy_entropy: float = Field(
        ...,
        description=(
            "Mean H(pi(.|s)) over the batch's states, in nats. Bounded above by "
            "log(n_actions) = 0.693 for CartPole. A run whose entropy collapses "
            "to zero early has stopped exploring, and its flat learning curve "
            "has a cause you can point at."
        ),
    )
    off_policy: bool = Field(
        default=False,
        description="True when this update reused an older batch through importance sampling.",
    )
    is_weight_mean: float | None = None
    is_weight_max: float | None = None
    is_weight_p95: float | None = None
    is_weight_ess: float | None = Field(
        default=None,
        description=(
            "Effective sample size (sum w)^2 / sum w^2, as a FRACTION of the "
            "number of weighted samples. 1.0 means the old and new policies "
            "agree; 0.05 means twenty samples are doing the work of one and "
            "the update is nearly worthless however good its mean looks."
        ),
    )
    is_weight_histogram: dict[str, Any] | None = Field(
        default=None,
        description=(
            "{'edges': [...], 'counts': [...]} — the weight distribution the "
            "assignment asks you to plot. A histogram rather than the raw "
            "weights because a 200-step episode times 10 episodes times 100 "
            "updates is 200,000 floats per run, and the free tier has 500 MB."
        ),
    )


class GradientStatsResponse(BaseModel):
    stats: list[GradientStatRow]
    count: int
    degraded: bool = False


class EpisodesResponse(BaseModel):
    """Per-episode returns for one experiment, for the learning-curve tabs."""

    experiment_id: str
    episodes: list[EpisodeRow]
    count: int
    degraded: bool = False
