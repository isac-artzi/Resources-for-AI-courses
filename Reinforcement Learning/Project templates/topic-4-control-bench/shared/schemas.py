"""
shared/schemas.py — the contract between every pair of tiers.

Nothing crosses a tier boundary in this repository except an instance of a model
defined in this file or a row defined by a migration in db/migrations/. That is
the whole point of the file: the Streamlit tier, the service tier and the test
suite all import the SAME class, so a contract change that breaks a caller
breaks it at import time and in CI, not in front of a stakeholder.

Two conventions worth copying into your own work:

1.  Requests and responses are separate types. It is tempting to reuse one model
    for both; do not. The response almost always grows fields the request must
    never accept (`policy_sha256`, `latency_ms`), and a shared model quietly
    makes those settable by the caller.

2.  Every numeric field that has a legal range says so with `Field(...)`.
    FastAPI turns those into a 422 with a readable message. A stack trace is not
    an error contract.
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
    one-element list holding the discrete state index. Keeping one shape across
    all six products means the Streamlit "Play" tab and the test suite do not
    change when the policy stops being a table.

    In THIS product the same request type serves three policies whose
    observations are 4, 6 and 3 numbers wide. Pydantic cannot check that for
    you — it does not know which artifact `policy_name` will resolve to — so the
    check lives in `api/main.act` and produces a 422 naming both numbers. That
    hazard is the reason the dimensionality test is required here rather than
    merely encouraged.
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
        description="Greedy/modal action if true; sample from the policy if false.",
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
        default=None,
        description=(
            "max_a Q(s,a) for a table, the chosen action's probability for a "
            "categorical policy, and the mean log σ for the SAC actor — which is "
            "how uncertain it is here, not a value. Different quantities behind "
            "one field name is a compromise; the model card must say which is "
            "which for each policy."
        ),
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
    max_steps: int = Field(default=500, ge=1, le=2000)
    seed: int | None = Field(
        default=None, description="Name the seed or the result is not evidence."
    )
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
    env_id: str | None = Field(
        default=None,
        description=(
            "Which environment the rollout actually ran in, read from the "
            "artifact rather than from the request. With three agents behind one "
            "contract, a response that does not say what it ran cannot be "
            "checked by the person reading it."
        ),
    )
    random_baseline: float | None = Field(
        default=None,
        description=(
            "The return a uniformly random policy scores in this environment. "
            "Returned alongside the result because '-450' means nothing until "
            "the reader knows the floor is -500."
        ),
    )


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
    action_space: Literal["discrete", "continuous"] = Field(
        default="discrete",
        description=(
            "Added in Topic 4, because one contract now serves both. The UI "
            "needs it to decide whether to render an action as a label or as a "
            "torque, and a caller needs it to know whether `action` will come "
            "back as an int or as a list."
        ),
    )
    env_id: str | None = Field(
        default=None,
        description=(
            "Which environment this artifact was trained on, read out of the "
            "archive. /rollout uses it to construct the right environment; "
            "without it the service would have to guess from obs_dim, and two "
            "environments with the same width would collide silently."
        ),
    )


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
            "The exploration parameter IN FORCE at this episode. Null for every "
            "algorithm in this product: A2C, PPO and SAC all explore through the "
            "entropy of their own policy rather than through an epsilon, which "
            "is why `policy_updates.policy_entropy` is the column that replaces "
            "it. Null rather than 0.0 — a zero here would say 'fully greedy', "
            "which is a different and false claim."
        ),
    )
    env_steps: int | None = Field(
        default=None,
        description=(
            "Cumulative environment steps consumed at the END of this episode. "
            "Topic 4's headline comparison is at MATCHED STEP BUDGETS, and "
            "episode index is not a step budget: an A2C run whose episodes last "
            "20 steps and a PPO run whose episodes last 500 have consumed "
            "twenty-five times more experience at the same episode number. "
            "Without this column the bake-off chart cannot be drawn honestly."
        ),
    )


class EpisodesResponse(BaseModel):
    """Per-episode returns for one experiment — the raw material of a curve."""

    experiment_id: str
    episodes: list[EpisodeRow]
    count: int
    degraded: bool = False


class EvaluationRow(BaseModel):
    experiment_id: str
    episodes: int
    mean_return: float
    std_return: float
    stderr_return: float
    at_training_episode: int


# ---------------------------------------------------------------------------
# Topic 4 adds two row types.
#
#   `policy_updates` answers "how big was each update, and how far did the
#   policy move" — one row per gradient update, for all three algorithms.
#   `entropy_sweep`  answers "what did the temperature do" — one row per SAC
#   run in the sweep.
#
# Both mirror db/migrations/002_topic4.sql. Change the pair in one commit.
# ---------------------------------------------------------------------------


class PolicyUpdateRow(BaseModel):
    """One row per POLICY UPDATE, not per episode.

    The grain is the first thing to get right. A PPO iteration that collects
    1,024 steps and then takes ten epochs of minibatch updates produces roughly
    two `episodes` rows and exactly ONE row here: the KL divergence being
    reported is the distance between the policy that COLLECTED the batch and the
    policy that exists after the whole iteration, which is the quantity PPO's
    clip is supposed to bound. Logging a KL per minibatch would give you forty
    numbers per iteration that no longer answer that question.

    One table for three algorithms, with nullable columns, rather than three
    tables. The comparison the product exists to make is across algorithms, and
    a comparison across three tables is three queries and a join you will get
    wrong at least once. Nullable is the honest encoding of "this algorithm does
    not have that quantity": A2C has no KL because it takes one step per batch
    and never asks how far it moved, and filling that with 0.0 would make it
    look like the most conservative method in the study.
    """

    experiment_id: str
    update_index: int = Field(..., ge=0, description="0-based index of the policy update.")
    env_steps: int = Field(
        ...,
        ge=0,
        description=(
            "Environment steps consumed BEFORE this update. The x-axis of every "
            "matched-budget comparison in this product."
        ),
    )
    episode_index: int = Field(
        ..., ge=0, description="Episodes completed before this update."
    )

    policy_loss: float | None = None
    value_loss: float | None = Field(
        default=None,
        description=(
            "Critic loss. Required for SAC by the product brief, and worth "
            "logging for all three: a critic whose loss is falling while the "
            "return is flat is learning to predict a policy that is not "
            "improving, which is a different problem from a critic that is not "
            "learning at all."
        ),
    )
    policy_entropy: float | None = Field(
        default=None,
        description=(
            "Mean H(π(·|s)) over the batch, in nats. Bounded above by "
            "ln(n_actions) for a categorical policy — 0.693 for CartPole, 1.099 "
            "for Acrobot — and UNBOUNDED for the continuous SAC actor, whose "
            "differential entropy is negative when σ is small. Do not put the "
            "two on one axis without saying so."
        ),
    )
    kl_divergence: float | None = Field(
        default=None,
        description=(
            "Mean KL(π_old ‖ π_new) over the batch's states, for the update just "
            "taken. PPO only; null elsewhere. This is the column that makes the "
            "trust region visible in DATA rather than in prose: PPO does not "
            "constrain the KL directly the way TRPO does, it clips a ratio and "
            "HOPES the KL stays small — and whether that hope held is an "
            "empirical question your own run can answer."
        ),
    )
    clip_fraction: float | None = Field(
        default=None,
        description=(
            "Fraction of sampled steps whose likelihood ratio left [1−ε, 1+ε] "
            "and was therefore clipped. PPO only. Near zero means the clip never "
            "engaged and you were running an ordinary surrogate; near one means "
            "almost every sample was outside the trust region and the batch was "
            "mostly wasted."
        ),
    )
    alpha: float | None = Field(
        default=None,
        description=(
            "The entropy temperature in force at this update. SAC only. Logged "
            "per update rather than per run because under automatic tuning it "
            "MOVES, and its trajectory is the evidence for the claim that "
            "automatic tuning does something a fixed value cannot."
        ),
    )


class PolicyUpdatesResponse(BaseModel):
    stats: list[PolicyUpdateRow]
    count: int
    degraded: bool = False


class EntropySweepRow(BaseModel):
    """One row per SAC run in the temperature sweep — three regimes × ≥3 seeds.

    Written at the END of a run rather than during it, because every field here
    is a summary of the whole run. That is a different grain from
    `policy_updates` and it deserves its own table: mixing a per-update series
    and a per-run summary in one table forces every query to carry a filter that
    a reader has to notice in order to trust the number.
    """

    experiment_id: str
    mode: Literal["fixed", "auto"] = Field(
        ...,
        description="Whether α was held fixed or tuned automatically against a target entropy.",
    )
    alpha_setting: str = Field(
        ...,
        description=(
            "The human label of this arm — 'alpha=0.5', 'alpha=0.01', 'auto'. "
            "Kept as text so 'auto' does not have to be encoded as a magic "
            "number, and so the Streamlit table can group on it directly."
        ),
    )
    alpha_value: float | None = Field(
        default=None,
        description=(
            "The fixed α, or the FINAL α under automatic tuning. Null would lose "
            "the most interesting number in the sweep: where the tuner ended up "
            "relative to the two fixed values you chose by hand."
        ),
    )
    seed: int
    episodes: int
    env_steps: int

    episodes_to_threshold: int | None = Field(
        default=None,
        description=(
            "Convergence speed: the first episode at which the trailing "
            "10-episode mean return reached `threshold`. NULL means the run "
            "never got there, which is a result and must not be encoded as a "
            "large number — an 'infinity' of 9999 would be averaged into a mean "
            "and silently ruin the comparison."
        ),
    )
    threshold: float = Field(
        ...,
        description="The stated bar. A convergence speed without its threshold is not a number.",
    )
    mean_return_last_100: float = Field(
        ...,
        description=(
            "Final performance: mean training return over the last 100 episodes "
            "of the run, or over all of them if the run was shorter — in which "
            "case `episodes` tells the reader that, and the caption must too."
        ),
    )
    return_std_last_100: float = Field(
        ..., description="Within-run spread over the same window."
    )
    mean_policy_entropy: float = Field(
        ...,
        description=(
            "Mean differential entropy of the squashed Gaussian actor over the "
            "run's updates, in nats. NEGATIVE is normal and correct here: a "
            "continuous density can exceed 1, so differential entropy is not "
            "bounded below by zero. This is the column that shows α = 0.5 "
            "exploring more than α = 0.01."
        ),
    )
    eval_mean_return: float | None = Field(
        default=None,
        description="Deterministic (modal) evaluation of the finished policy.",
    )


class EntropySweepResponse(BaseModel):
    rows: list[EntropySweepRow]
    count: int
    degraded: bool = False
