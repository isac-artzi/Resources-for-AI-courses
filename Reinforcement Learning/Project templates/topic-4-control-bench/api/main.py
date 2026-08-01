"""
api/main.py — the service tier. It owns the policies and nothing else.

Run it the way you will demonstrate it:

    uvicorn api.main:app --reload --port 8000
    open http://127.0.0.1:8000/docs

Topic 4 is the first product where "one contract, several agents" is the whole
brief, so two endpoints are worth reading closely even if you skim the rest:

  * `/act` validates the observation against the artifact's declared
    dimensionality and returns **422 with a readable message** on a mismatch.
    Three agents are registered here whose observations are 4, 6 and 3 numbers
    wide, so sending a CartPole state to the Acrobot policy is not a contrived
    test case — it is the mistake a caller makes in their first hour. A stack
    trace is not an error contract, and "it returned 500" is not a description a
    caller can act on.

  * `/rollout` builds the environment named INSIDE the artifact, not one named
    in the request. The caller asks for a policy; which world that policy lives
    in is a property of the policy. Letting the caller choose would make it
    possible to evaluate the Pendulum actor on CartPole, which would fail deep
    inside Gymnasium with a shape error rather than at the boundary.

  * `/healthz` returns 200 only when a policy artifact is loaded AND the data
    tier answers. A health check that reports success because the process is
    running tells you nothing the response itself did not already prove.

The Streamlit tier imports the handler functions from this module directly when
SERVICE_MODE=inprocess, and calls them over HTTP when SERVICE_MODE=http. The
contract is identical either way — that is the whole design.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Response, status

from api.policy import PolicyArtifactStore
from shared.config import get_settings
from shared.schemas import (
    ActRequest,
    ActResponse,
    EntropySweepResponse,
    EntropySweepRow,
    EpisodeRow,
    EpisodesResponse,
    HealthResponse,
    PoliciesResponse,
    PolicyArtifact,
    PolicyUpdateRow,
    PolicyUpdatesResponse,
    RolloutRequest,
    RolloutResponse,
    RunSummary,
    RunsResponse,
    Transition,
    VersionResponse,
)
from shared.store import get_store

settings = get_settings()

app = FastAPI(
    title="Control Bench",
    version=settings.app_version,
    description=(
        "Hosts three trained control agents — A2C, PPO and SAC — behind one "
        "contract. Serves NumPy artifacts. Never imports a training framework; "
        "see GET /version, which reports that honestly."
    ),
)

# One artifact store per process. Reloaded by POST /reload during development.
POLICIES = PolicyArtifactStore(settings.policy_dir, default_name=settings.default_policy)

# The environment factory is supplied by the topic layer. Imported defensively
# so that a host without the classic-control extra still starts and answers
# /healthz — only the endpoints that genuinely need an environment fail.
try:  # pragma: no cover - the topic layer provides this
    from envs import ENV_SPECS, make_env  # type: ignore
except Exception:  # pragma: no cover
    make_env = None  # type: ignore
    ENV_SPECS = {}  # type: ignore


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
        detail = f"no loadable .npz artifact in {settings.policy_dir}/ — run python -m train.train"
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
                f"Registered: {POLICIES.names() or '(none — run python -m train.train)'}"
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    expected = meta.get("obs_dim")
    if expected is not None and len(req.state) != expected:
        # 422 with BOTH numbers and the environment in it. This exact case is a
        # required test, and in this product it is the mistake a caller actually
        # makes: CartPole is 4-dimensional, Acrobot 6 and Pendulum 3, and all
        # three answer at the same URL. Naming the environment as well as the
        # widths turns "422: dimension mismatch" into a message the caller can
        # act on without opening your source.
        raise HTTPException(
            status_code=422,
            detail=(
                f"observation has dimension {len(req.state)}, but policy "
                f"'{req.policy_name}' expects {expected}"
                + (f" (trained on {meta['env_id']})" if meta.get("env_id") else "")
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
        raise HTTPException(
            status_code=404,
            detail=(
                f"no policy named '{req.policy_name}'. Registered: {POLICIES.names()}"
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # The environment comes from the ARTIFACT, never from the request. See the
    # module docstring: which world a policy lives in is a property of the
    # policy, and a caller who could choose would sooner or later evaluate the
    # Pendulum actor on CartPole.
    env_id = meta.get("env_id")
    if not env_id:
        raise HTTPException(
            status_code=503,
            detail=(
                f"artifact '{req.policy_name}' does not record which environment it "
                "was trained on. Re-export it with train/export.py, which writes "
                "env_id into the archive."
            ),
        )
    spec = ENV_SPECS.get(env_id)

    try:
        env = make_env(env_id)
    except Exception as exc:
        # The environment is a SERVING dependency in this product, because
        # /rollout runs episodes server-side. If gymnasium is missing from the
        # deployed environment this is where you find out, and a 503 naming the
        # package is a better answer than a 500 naming a traceback.
        raise HTTPException(status_code=503, detail=f"could not construct {env_id}: {exc}")

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
            obs, reward, terminated, truncated, _ = env.step(_as_env_action(action))
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
        env_id=env_id,
        # Returned with the result rather than left to the reader. "-450" means
        # nothing until you know the random floor is -500.
        random_baseline=float(spec.random_return) if spec else None,
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


# ---------------------------------------------------------------------------
# Topic 4 adds three read endpoints. They exist so the Streamlit tier can draw
# the Bake-Off and Entropy Sweep tabs WITHOUT holding a database handle: rule 1
# of ui/app.py is that every number comes back through the service tier. It
# would be two lines shorter to query Supabase from the UI with the anon key,
# and doing so would put an aggregation the service owns into the presentation
# tier, where the test suite cannot reach it.
# ---------------------------------------------------------------------------


@app.get("/episodes", response_model=EpisodesResponse)
def episodes(experiment_id: str, limit: int = 20_000) -> EpisodesResponse:
    """Per-episode returns for one run — the raw material of a learning curve.

    `experiment_id` is REQUIRED and there is no unfiltered form. An endpoint
    that dumps every row of the biggest table in the schema when called with no
    arguments is one bookmark away from being the reason your free-tier project
    is over quota.

    Returned raw rather than smoothed. Smoothing is a presentation decision and
    belongs in the tab that draws the curve, where the window is visible next to
    the chart; a service that silently returns a 20-episode moving average makes
    every downstream claim about seed variance quietly wrong.
    """
    limit = max(1, min(limit, 50_000))
    try:
        rows = get_store().episodes_for(experiment_id)
        degraded = False
    except Exception:
        rows, degraded = [], True

    out: list[EpisodeRow] = []
    for r in rows[:limit]:
        try:
            out.append(
                EpisodeRow(
                    experiment_id=str(r.get("experiment_id", experiment_id)),
                    episode_index=int(r["episode_index"]),
                    **{"return": float(r["return"])},
                    length=int(r.get("length", 0)),
                    epsilon=r.get("epsilon"),
                    env_steps=r.get("env_steps"),
                )
            )
        except Exception:
            # A single malformed row must not empty the whole chart. Skip it and
            # let the count disagree with the row count in the database — a
            # visibly short series is a better failure than a 500.
            continue
    return EpisodesResponse(
        experiment_id=experiment_id, episodes=out, count=len(out), degraded=degraded
    )


@app.get("/policy_updates", response_model=PolicyUpdatesResponse)
def policy_updates(experiment_id: str, limit: int = 5000) -> PolicyUpdatesResponse:
    """Per-update statistics for one run: losses, entropy, and PPO's KL.

    This is the endpoint behind the trust-region evidence. PPO does not
    constrain the KL the way TRPO does — it clips a likelihood ratio and hopes
    the KL stays small — so whether the hope held on YOUR run is an empirical
    question, and this is the series that answers it.
    """
    limit = max(1, min(limit, 50_000))
    try:
        rows = get_store().policy_updates_for(experiment_id)
        degraded = False
    except Exception:
        rows, degraded = [], True

    out: list[PolicyUpdateRow] = []
    for r in rows[:limit]:
        try:
            out.append(
                PolicyUpdateRow(
                    **{k: v for k, v in r.items() if k in PolicyUpdateRow.model_fields}
                )
            )
        except Exception:
            continue
    return PolicyUpdatesResponse(stats=out, count=len(out), degraded=degraded)


@app.get("/entropy_sweep", response_model=EntropySweepResponse)
def entropy_sweep(limit: int = 200) -> EntropySweepResponse:
    """One row per SAC run in the temperature study.

    Unfiltered, unlike the two above, and that is safe for one reason worth
    stating: this table has one row per RUN, so the sweep the brief asks for is
    nine rows and a generous re-run history is a few hundred. The size argument
    that forces `experiment_id` on `/episodes` simply does not apply.
    """
    limit = max(1, min(limit, 1000))
    try:
        rows = get_store().entropy_sweep_rows(limit)
        degraded = False
    except Exception:
        rows, degraded = [], True

    out: list[EntropySweepRow] = []
    for r in rows:
        try:
            out.append(
                EntropySweepRow(
                    **{k: v for k, v in r.items() if k in EntropySweepRow.model_fields}
                )
            )
        except Exception:
            continue
    return EntropySweepResponse(rows=out, count=len(out), degraded=degraded)


@app.post("/reload")
def reload_artifacts() -> dict[str, object]:
    """Development convenience: pick up a newly exported artifact without a restart."""
    POLICIES.reload()
    return {"loaded": POLICIES.names()}


# ---------------------------------------------------------------------------


def _as_vector(obs) -> np.ndarray:
    """Normalise a Gymnasium observation into the vector shape /act expects."""
    if isinstance(obs, (int, np.integer)):
        return np.asarray([float(obs)], dtype=np.float64)
    return np.asarray(obs, dtype=np.float64).ravel()


def _as_env_action(action):
    """Turn the action `policy.act()` returned into the shape `env.step()` wants.

    A discrete environment wants a plain Python int. A continuous one wants an
    ARRAY, even when the action is one number: Gymnasium checks the action
    against `Box(shape=(1,))` and a bare float fails that check with a message
    about `np.ndarray` that reads like an internal error. One line here, in the
    one place actions cross into the environment, rather than the same
    conversion repeated in the training loop and the rollout handler and
    eventually disagreeing between them.
    """
    if isinstance(action, list):
        return np.asarray(action, dtype=np.float32)
    return int(action)


def _hash_state(state: list[float]) -> str:
    import hashlib

    return hashlib.sha256(",".join(f"{v:.6g}" for v in state).encode()).hexdigest()[:16]
