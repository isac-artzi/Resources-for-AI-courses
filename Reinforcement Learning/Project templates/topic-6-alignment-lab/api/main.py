"""
api/main.py — the service tier. It owns the policy and nothing else.

Run it the way you will demonstrate it:

    uvicorn api.main:app --reload --port 8000
    open http://127.0.0.1:8000/docs

TOPIC 6: THIS SERVICE SCORES TEXT. IT DOES NOT GENERATE ANY.
------------------------------------------------------------
There is no `/generate`, and its absence is the architecture rather than a gap
in the build. Generation happens once, offline, in the training tier, and the
completions are persisted; `GET /completions` reads them back. `POST /score`
and `POST /compare` evaluate a reward head that is four NumPy operations over
a TF-IDF vector. Nothing in this process has ever imported a transformer, and
`GET /version` reports that honestly rather than asserting it.

The reason is the same one that has governed every topic: `import torch` alone
costs ~490 MB against a 690 MB guarantee, and a 124M-parameter language model
costs another ~500 MB on top of that, before the first token is sampled. A
service that scores is deployable on the free tier; a service that generates is
not. That is also the shape of a real inference budget — the expensive model
runs where you can afford it and the cheap one runs where the traffic is.

ONE MORE THING WORTH READING BEFORE THE CODE
--------------------------------------------
`/score` receives USER TEXT. It writes a SHA-256 digest to `audit_log` and
never the text itself. The comment about hashing in `db/migrations/001_init.sql`
has been there since Topic 1 as a hypothetical; this is the topic where it
stops being one, and `tests/test_audit_hashing.py` asserts it rather than
trusting it.

Every endpoint below is required by the course quality bar. Two of them are
worth reading closely even if you skim the rest:

  * `/act` validates the observation against the artifact's declared
    dimensionality and returns **422 with a readable message** on a mismatch.
    A stack trace is not an error contract, and "it returned 500" is not a
    description a caller can act on.

  * `/healthz` returns 200 only when the policy artifact is loaded AND the
    data tier answers. A health check that reports success because the process
    is running tells you nothing the response itself did not already prove.

The Streamlit tier imports the handler functions from this module directly
when SERVICE_MODE=inprocess, and calls them over HTTP when SERVICE_MODE=http.
The contract is identical either way — that is the whole design.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Response, status

from api.policy import PolicyArtifactStore
from api.reward import RewardHead, sigmoid
from shared.config import get_settings
from shared.schemas import (
    ActRequest,
    ActResponse,
    AlignmentRunRow,
    AlignmentRunsResponse,
    CompareRequest,
    CompareResponse,
    CompletionRow,
    CompletionsResponse,
    HealthResponse,
    PoliciesResponse,
    PolicyArtifact,
    RolloutRequest,
    RolloutResponse,
    RunSummary,
    RunsResponse,
    ScoreRequest,
    ScoreResponse,
    Transition,
    VersionResponse,
)
from shared.store import get_store

settings = get_settings()

app = FastAPI(
    title="Agent Service",
    version=settings.app_version,
    description=(
        "Owns the learned policy. Serves NumPy artifacts. Never imports a "
        "training framework — see GET /version, which reports that honestly."
    ),
)

# One artifact store per process. Reloaded by POST /reload during development.
POLICIES = PolicyArtifactStore(settings.policy_dir)

# The environment factory is supplied by the topic layer. The base template
# ships a stub so that a fresh clone starts and /healthz answers before you
# have written anything. Replace it in envs/ and import it here.
try:  # pragma: no cover - the topic layer provides this
    from envs import make_env  # type: ignore
except Exception:  # pragma: no cover
    make_env = None  # type: ignore


# ---------------------------------------------------------------------------


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    return VersionResponse(
        app_version=settings.app_version,
        git_sha=settings.git_sha,
        # Reported, not assumed. If this is ever True in the deployed process,
        # the memory budget has already been spent.
        torch_imported="torch" in sys.modules,
    )


@app.get("/healthz", response_model=HealthResponse)
def healthz(response: Response) -> HealthResponse:
    artifact_ok = POLICIES.loaded()
    try:
        data_ok = get_store().reachable()
    except Exception:
        data_ok = False

    ok = artifact_ok and data_ok
    if not ok:
        # 503, not 200-with-a-sad-field. Orchestrators read status codes.
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    detail = None
    if not artifact_ok:
        detail = f"no loadable .npz artifact in {settings.policy_dir}/ — run train/export.py"
    elif not data_ok:
        detail = "data tier unreachable — a free-tier project pauses after a week idle"

    return HealthResponse(
        status="ok" if ok else "degraded",
        policy_artifact_loaded=artifact_ok,
        data_tier_reachable=data_ok,
        detail=detail,
    )


@app.get("/policies", response_model=PoliciesResponse)
def policies() -> PoliciesResponse:
    items = [PolicyArtifact(**m) for m in POLICIES.metadata() if "error" not in m]
    return PoliciesResponse(policies=items, count=len(items))


@app.post("/act", response_model=ActResponse)
def act(req: ActRequest) -> ActResponse:
    t0 = time.perf_counter()
    try:
        policy, meta = POLICIES.get(req.policy_name)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"no policy named '{req.policy_name}'. "
                f"Registered: {POLICIES.names() or '(none — run train/export.py)'}"
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    expected = meta.get("obs_dim")
    if expected is not None and len(req.state) != expected:
        # 422 with the numbers in it. This exact case is a required test.
        raise HTTPException(
            status_code=422,
            detail=(
                f"observation has dimension {len(req.state)}, but policy "
                f"'{req.policy_name}' expects {expected}"
            ),
        )

    try:
        action, value = policy.act(np.asarray(req.state, dtype=np.float64), req.deterministic)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Audit by state HASH, not by raw state. In Topic 6 the "state" is user
    # text, and logging it raw would put user content in your database.
    try:
        get_store().log_audit(
            {
                "endpoint": "/act",
                "policy_sha256": meta["sha256"],
                "state_hash": _hash_state(req.state),
                "action": str(action),
            }
        )
    except Exception:
        pass  # telemetry must never take down the request path

    return ActResponse(
        action=action,
        policy_name=req.policy_name,
        policy_sha256=meta["sha256"],
        value_estimate=value,
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )


@app.post("/rollout", response_model=RolloutResponse)
def rollout(req: RolloutRequest) -> RolloutResponse:
    if make_env is None:
        raise HTTPException(
            status_code=501,
            detail="envs/make_env() is not implemented yet — see envs/__init__.py",
        )
    try:
        policy, meta = POLICIES.get(req.policy_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no policy named '{req.policy_name}'")

    env = make_env()
    returns: list[float] = []
    lengths: list[int] = []
    trajectory: list[Transition] = []

    for ep in range(req.episodes):
        seed = None if req.seed is None else req.seed + ep
        obs, _ = env.reset(seed=seed)
        total, steps = 0.0, 0
        for step in range(req.max_steps):
            state_vec = _as_vector(obs)
            action, _ = policy.act(state_vec, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
            if ep == 0 and req.record_trajectory:
                trajectory.append(
                    Transition(
                        step=step,
                        state=[float(v) for v in state_vec],
                        action=action,
                        reward=float(reward),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                    )
                )
            if terminated or truncated:
                break
        returns.append(total)
        lengths.append(steps)
    env.close()

    arr = np.asarray(returns, dtype=np.float64)
    n = max(len(arr), 1)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return RolloutResponse(
        returns=returns,
        mean_return=float(arr.mean()),
        std_return=std,
        stderr_return=std / np.sqrt(n),
        mean_length=float(np.mean(lengths)),
        episodes=req.episodes,
        seed=req.seed,
        trajectory=trajectory,
        policy_name=req.policy_name,
        policy_sha256=meta["sha256"],
    )


@app.get("/runs", response_model=RunsResponse)
def runs(limit: int = 50) -> RunsResponse:
    limit = max(1, min(limit, 200))
    try:
        rows = get_store().recent_runs(limit)
        degraded = False
    except Exception:
        rows, degraded = [], True

    out: list[RunSummary] = []
    for r in rows:
        out.append(
            RunSummary(
                experiment_id=str(r.get("experiment_id") or r.get("id") or ""),
                algorithm=str(r.get("algorithm", "")),
                env_id=str(r.get("env_id", "")),
                seed=int(r.get("seed", 0)),
                hyperparameters=r.get("hyperparameters") or {},
                git_sha=r.get("git_sha"),
                created_at=str(r.get("created_at")) if r.get("created_at") else None,
                episodes_logged=int(r.get("episodes_logged", 0)),
                mean_return_last_100=r.get("mean_return_last_100"),
                eval_mean_return=r.get("eval_mean_return"),
                eval_stderr=r.get("eval_stderr"),
            )
        )
    return RunsResponse(runs=out, count=len(out), degraded=degraded)


@app.post("/reload")
def reload_artifacts() -> dict[str, object]:
    """Development convenience: pick up a newly exported artifact without a restart."""
    POLICIES.reload()
    return {"loaded": POLICIES.names()}


# ===========================================================================
# TOPIC 6 — the scoring endpoints.
# ===========================================================================


def _reward_head(name: str) -> tuple[RewardHead, dict]:
    """Resolve a name to a reward head, or raise the right HTTP error.

    Factored out of the three handlers below because the failure modes are
    identical and because getting them wrong in one of three places is how a
    service ends up returning 404 from one endpoint and 500 from another for
    the same missing artifact.
    """
    try:
        artifact, meta = POLICIES.get(name)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"no policy named '{name}'. Registered: "
                f"{POLICIES.names() or '(none — run python -m train.train)'}"
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if not isinstance(artifact, RewardHead):
        # 422, not 500. The caller asked a well-formed question of the wrong
        # object, and the answer they need is which object to ask instead.
        raise HTTPException(
            status_code=422,
            detail=(
                f"'{name}' is a {meta.get('kind')} artifact, not a reward head. "
                "Reward heads are the ones with a 'vocab' array — see "
                "GET /policies, where kind == 'reward-head'."
            ),
        )
    return artifact, meta


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    """Text in, one reward-model score out.

    The score is UNCALIBRATED. A Bradley-Terry reward model is identified only
    up to an additive constant per prompt — adding 5 to every reward leaves the
    likelihood of every comparison unchanged — so `reward = 2.7` means nothing
    at all on its own. It is comparable across two texts and not across two
    artifacts, two vocabularies, or two training runs. `/compare` exists so
    that there is a correct endpoint to reach for; this one exists because
    stakeholders will ask for a number and it is better to give them one with a
    caveat attached than to have them compute a worse one themselves.
    """
    t0 = time.perf_counter()
    head, meta = _reward_head(req.policy_name)

    reward, n_tokens, oov = head.score(req.text)
    digest = _hash_text(req.text)

    # THE HASH IS THE POINT. `req.text` is user content and it does not go into
    # the database. Note also that the digest is over the raw text, so the same
    # submission produces the same row and you can count repeats — which is the
    # legitimate operational reason to log anything here at all — without ever
    # being able to reconstruct what was said.
    try:
        get_store().log_audit(
            {
                "endpoint": "/score",
                "policy_sha256": meta["sha256"],
                "state_hash": digest,
                "action": f"{reward:.6f}",
            }
        )
    except Exception:
        pass  # telemetry must never take down the request path

    return ScoreResponse(
        reward=reward,
        policy_name=req.policy_name,
        policy_sha256=meta["sha256"],
        tokens=n_tokens,
        oov_rate=oov,
        text_sha256=digest,
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest) -> CompareResponse:
    """Two texts in, the preferred one and the margin out.

    This is what the model was trained to do. The Bradley-Terry likelihood the
    head was fitted with is exactly

        P(a preferred to b) = sigmoid(r(a) - r(b))

    so `probability` below is not a post-hoc calibration story — it is the same
    functional form as the training objective, evaluated on new text. Quote it
    in preference to the raw rewards.

    Both texts are hashed into the audit log, joined by the digest of the pair,
    for the same reason `/score` hashes one.
    """
    t0 = time.perf_counter()
    head, meta = _reward_head(req.policy_name)

    ra, _, _ = head.score(req.text_a)
    rb, _, _ = head.score(req.text_b)
    margin = abs(ra - rb)

    # Ties go to "a". A comparison endpoint that returns None on an exact tie
    # forces every caller to handle a case that occurs about once in 10^9
    # requests, and they will handle it by crashing. Document the convention
    # instead: exact float equality here means the two texts produced the same
    # tf-idf vector, which almost always means they are the same text.
    preferred = "a" if ra >= rb else "b"

    try:
        get_store().log_audit(
            {
                "endpoint": "/compare",
                "policy_sha256": meta["sha256"],
                "state_hash": _hash_text(req.text_a + "\x00" + req.text_b),
                "action": preferred,
            }
        )
    except Exception:
        pass

    return CompareResponse(
        preferred=preferred,
        reward_a=ra,
        reward_b=rb,
        margin=margin,
        probability=float(sigmoid(margin)),
        policy_name=req.policy_name,
        policy_sha256=meta["sha256"],
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )


@app.get("/completions", response_model=CompletionsResponse)
def completions(prompt_id: str | None = None, limit: int = 200) -> CompletionsResponse:
    """Base and aligned completions for a prompt, read from the data tier.

    Read-only, and deliberately a READ. The service does not generate these; it
    serves rows that `train/dpo.py` wrote after generating them offline. If this
    endpoint is empty, the fix is to run the training tier, not to add a model
    to the service.

    `limit` is capped for the same reason `/rollout`'s episode count is: an
    unbounded list endpoint on a free-tier instance is a denial of service with
    a REST interface.
    """
    limit = max(1, min(limit, 500))
    try:
        rows = get_store().completions(prompt_id=prompt_id, limit=limit)
        degraded = False
    except Exception:
        rows, degraded = [], True

    out = [
        CompletionRow(
            prompt_id=str(r.get("prompt_id", "")),
            prompt=str(r.get("prompt", "")),
            model_variant=str(r.get("model_variant", "")),
            beta=r.get("beta"),
            text=str(r.get("text", "")),
            reward_score=r.get("reward_score"),
            true_quality=r.get("true_quality"),
            tokens=r.get("tokens"),
        )
        for r in rows
    ]
    return CompletionsResponse(
        prompt_id=prompt_id, completions=out, count=len(out), degraded=degraded
    )


@app.get("/alignment_runs", response_model=AlignmentRunsResponse)
def alignment_runs(limit: int = 50) -> AlignmentRunsResponse:
    """One row per beta: the loss, the two implicit-reward diagnostics, and the KL.

    This is the table the "Reward Hacking" tab plots. It is a separate endpoint
    from `/runs` because an alignment run is not an episodic training run and
    forcing it through `RunSummary` would mean reporting a KL divergence in a
    column called `mean_return_last_100`.
    """
    limit = max(1, min(limit, 200))
    try:
        rows = get_store().alignment_runs(limit)
        degraded = False
    except Exception:
        rows, degraded = [], True

    out = [
        AlignmentRunRow(
            beta=float(r.get("beta", 0.0)),
            final_loss=r.get("final_loss"),
            implicit_reward_margin=r.get("implicit_reward_margin"),
            implicit_reward_accuracy=r.get("implicit_reward_accuracy"),
            kl_from_reference=r.get("kl_from_reference"),
            mean_reward_model_score=r.get("mean_reward_model_score"),
            mean_true_quality=r.get("mean_true_quality"),
            steps=r.get("steps"),
            seed=r.get("seed"),
            experiment_id=str(r["experiment_id"]) if r.get("experiment_id") else None,
            created_at=str(r.get("created_at")) if r.get("created_at") else None,
        )
        for r in rows
    ]
    return AlignmentRunsResponse(runs=out, count=len(out), degraded=degraded)


# ---------------------------------------------------------------------------


def _as_vector(obs) -> np.ndarray:
    """Normalise a Gymnasium observation into the vector shape /act expects."""
    if isinstance(obs, (int, np.integer)):
        return np.asarray([float(obs)], dtype=np.float64)
    return np.asarray(obs, dtype=np.float64).ravel()


def _hash_state(state: list[float]) -> str:
    import hashlib

    return hashlib.sha256(",".join(f"{v:.6g}" for v in state).encode()).hexdigest()[:16]


def _hash_text(text: str) -> str:
    """SHA-256 of the UTF-8 bytes, FULL 64 hex characters.

    Not truncated to 16 the way `_hash_state` is, and the difference is
    deliberate. A truncated 64-bit digest is fine for a four-float observation
    vector, where a collision costs you one mislabelled audit row. It is not
    fine for user text: a 64-bit space collides by birthday at about four
    billion entries, and an audit log is exactly the artifact you do not want
    to have to explain a collision in. The extra 48 characters cost nothing.

    Note that a hash is pseudonymisation, not anonymisation. Anyone holding a
    candidate text can confirm it was submitted by hashing it themselves.
    That is a FEATURE for deduplication and abuse investigation and a RISK for
    a short, guessable input space — "yes" hashes to the same digest for
    everyone. Say so in the model card rather than claiming the log is
    anonymous.
    """
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
