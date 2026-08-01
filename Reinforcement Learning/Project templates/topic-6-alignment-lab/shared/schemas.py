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
    # "reward-head" is this topic's addition: a scorer, not a controller. It
    # takes text and returns one number, so `n_actions` is meaningless for it
    # and `obs_dim` is the size of the feature vector (the TF-IDF vocabulary,
    # or the sentence-encoder dimension). Kept in the SAME enum as the control
    # policies deliberately — /policies, the checksum, the audit join and the
    # registry row are identical, which is the point the architecture note is
    # making. A reward model is a policy artifact like any other.
    kind: Literal["tabular", "mlp", "linear-head", "value-net", "reward-head"] = "tabular"
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
# TOPIC 6 — the scoring contract.
#
# Note what is NOT here: a generation request. The deployed service scores
# text; it never produces any. That is stated in the architecture note in the
# README and enforced here, in the type system, where a future contributor
# will actually trip over it.
# ---------------------------------------------------------------------------


class ScoreRequest(BaseModel):
    """Text in, one scalar out.

    `max_length` is 8,000 characters, and it is a security control rather than
    a nicety. The TF-IDF transform is linear in the input length and this
    endpoint is public; without a cap, one caller pasting a novel occupies a
    worker for as long as they like. Pick a number, state it in the model card,
    and return 422 above it — which FastAPI does for you because the bound is
    declared here.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="The text to score. Logged BY SHA-256 DIGEST ONLY — see api/main.py.",
    )
    policy_name: str = Field(
        default="reward_tfidf",
        max_length=64,
        description=(
            "Which registered reward head to use. Defaults to the TF-IDF head, "
            "which is the only one that is self-contained enough to deploy."
        ),
    )


class ScoreResponse(BaseModel):
    reward: float = Field(
        ...,
        description=(
            "The reward model's score. UNCALIBRATED AND UNBOUNDED: a "
            "Bradley-Terry model is identified only up to an additive constant "
            "per prompt, so this number is meaningful in COMPARISON and "
            "meaningless in isolation. The UI is required to say so; POST "
            "/compare exists so that callers have a correct alternative."
        ),
    )
    policy_name: str
    policy_sha256: str
    tokens: int = Field(
        ..., description="Tokens the head actually saw, after shared.preprocess.tokenise."
    )
    oov_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of tokens absent from the head's vocabulary. A score "
            "computed from text that is 90% out-of-vocabulary is a score of an "
            "empty vector, and the caller deserves to know that rather than to "
            "receive a confident number."
        ),
    )
    text_sha256: str = Field(
        ...,
        description=(
            "The digest written to audit_log. Returned so a caller can prove "
            "which request a logged row corresponds to WITHOUT the service "
            "having retained their text."
        ),
    )
    latency_ms: float


class CompareRequest(BaseModel):
    """Two texts in, the preferred one and the margin out.

    This is the endpoint that matches what the model was actually trained to
    do. `/score` is the convenient one; this is the correct one.
    """

    text_a: str = Field(..., min_length=1, max_length=8000)
    text_b: str = Field(..., min_length=1, max_length=8000)
    policy_name: str = Field(default="reward_tfidf", max_length=64)


class CompareResponse(BaseModel):
    preferred: Literal["a", "b"]
    reward_a: float
    reward_b: float
    margin: float = Field(
        ...,
        description="reward(preferred) - reward(other). Always >= 0 by construction.",
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "sigmoid(margin) — the Bradley-Terry probability that a labeller "
            "prefers the winner. This is the number to quote, because it is "
            "the quantity the training loss was a likelihood of."
        ),
    )
    policy_name: str
    policy_sha256: str
    latency_ms: float


# ---------------------------------------------------------------------------
# GET /completions — the offline generation library, read-only.
# ---------------------------------------------------------------------------


class CompletionRow(BaseModel):
    prompt_id: str
    prompt: str
    model_variant: str
    beta: float | None = Field(
        default=None,
        description="Null for the base model, which IS the reference and has no KL term.",
    )
    text: str
    reward_score: float | None = None
    true_quality: float | None = Field(
        default=None,
        description=(
            "A quality signal the reward model never saw. On the synthetic path "
            "this is exact; on the real path it is your hand rating. Without it "
            "there is no reward-hacking evidence, only a rising proxy."
        ),
    )
    tokens: int | None = None


class CompletionsResponse(BaseModel):
    prompt_id: str | None
    completions: list[CompletionRow]
    count: int
    degraded: bool = False


# ---------------------------------------------------------------------------
# GET /alignment_runs — one row per beta.
# ---------------------------------------------------------------------------


class AlignmentRunRow(BaseModel):
    beta: float
    final_loss: float | None = None
    implicit_reward_margin: float | None = None
    implicit_reward_accuracy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of held-out pairs the implicit reward ranks correctly.",
    )
    kl_from_reference: float | None = None
    mean_reward_model_score: float | None = None
    mean_true_quality: float | None = None
    steps: int | None = None
    seed: int | None = None
    experiment_id: str | None = None
    created_at: str | None = None


class AlignmentRunsResponse(BaseModel):
    runs: list[AlignmentRunRow]
    count: int
    degraded: bool = False


# ---------------------------------------------------------------------------
# Rows written by the training tier. These mirror db/migrations/002_topic6.sql;
# if you change one, change the other in the same commit.
# ---------------------------------------------------------------------------


class PreferenceRow(BaseModel):
    prompt_id: str
    prompt: str
    chosen: str
    rejected: str
    split: Literal["train", "test"]
    chosen_len: int | None = None
    rejected_len: int | None = None
    source: str = Field(
        ...,
        description=(
            "Required, never defaulted. A preferences table that mixes real and "
            "fallback data with no way to tell them apart is worse than no table."
        ),
    )
