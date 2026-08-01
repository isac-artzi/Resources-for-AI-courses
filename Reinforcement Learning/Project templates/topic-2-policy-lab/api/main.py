"""
api/main.py — the service tier. It owns the policy and nothing else.

Run it the way you will demonstrate it:

    uvicorn api.main:app --reload --port 8000
    open http://127.0.0.1:8000/docs

This topic serves TWO agents from one service: an exact plan and a learned
policy. `POST /act` and `POST /rollout` therefore accept a `policy_source` of
"value_iteration" or "monte_carlo", which is resolved to an artifact name
through `POLICY_SOURCE_ARTIFACTS` in shared/schemas.py. That is the whole
extension — one optional field and one dictionary. Forking the endpoint into
`/act_vi` and `/act_mc` would have been faster to write and would have doubled
the surface every later change has to be applied to twice.

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
from shared.config import get_settings
from shared.schemas import (
    POLICY_SOURCE_ARTIFACTS,
    ActRequest,
    ActResponse,
    ConvergencePoint,
    ConvergenceResponse,
    HealthResponse,
    PoliciesResponse,
    PolicyArtifact,
    RolloutRequest,
    RolloutResponse,
    RunSummary,
    RunsResponse,
    Transition,
    ValueGrid,
    ValueMapResponse,
    VersionResponse,
)
from shared.store import get_store

settings = get_settings()

app = FastAPI(
    title="Policy Lab Service",
    version=settings.app_version,
    description=(
        "Owns two policies for one problem — an exact plan and a learned "
        "approximation — and serves both from one contract. NumPy artifacts "
        "only; never imports a training framework, and GET /version reports "
        "that honestly rather than asserting it."
    ),
)

# One artifact store per process. Reloaded by POST /reload during development.
POLICIES = PolicyArtifactStore(settings.policy_dir, default_name=settings.default_policy)

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


def _resolve_policy_name(req: ActRequest | RolloutRequest) -> str:
    """policy_source -> artifact name, with policy_name as the fallback.

    One function, called by both endpoints, so the two can never drift. The
    lookup cannot raise: `policy_source` is a Literal, so FastAPI has already
    rejected anything outside the mapping with a 422 that lists the legal
    values — which is a better error than a 404 for a name the caller never
    typed.
    """
    if req.policy_source is not None:
        return POLICY_SOURCE_ARTIFACTS[req.policy_source]
    return req.policy_name


@app.post("/act", response_model=ActResponse)
def act(req: ActRequest) -> ActResponse:
    t0 = time.perf_counter()
    name = _resolve_policy_name(req)
    try:
        policy, meta = POLICIES.get(name)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"no policy named '{name}'"
                + (f" (requested as policy_source='{req.policy_source}')"
                   if req.policy_source else "")
                + f". Registered: {POLICIES.names() or '(none — run train/export.py)'}"
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
                f"'{name}' expects {expected}"
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
        # The RESOLVED name, not the requested one — see ActResponse.
        policy_name=name,
        policy_source=req.policy_source,
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
    name = _resolve_policy_name(req)
    try:
        policy, meta = POLICIES.get(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no policy named '{name}'")

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
        policy_name=name,
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


@app.get("/value_map", response_model=ValueMapResponse)
def value_map() -> ValueMapResponse:
    """Both value functions and their difference, shaped for a heat map.

    Served rather than computed in Streamlit because the presentation tier is
    not allowed to open a policy artifact. The values come straight out of the
    `.npz` files the training tier wrote — no arithmetic happens here beyond
    the subtraction, and even that is done once, in one place, so the "learned
    minus exact" sign convention cannot differ between the chart and its
    caption.
    """
    if make_env is None:  # pragma: no cover - the topic layer always provides it
        raise HTTPException(status_code=501, detail="envs/make_env() is not implemented")

    core = make_env().unwrapped
    arrows = _arrow_table(core)
    grids: list[ValueGrid] = []
    missing: list[str] = []
    by_source: dict[str, np.ndarray] = {}

    for source, artifact in POLICY_SOURCE_ARTIFACTS.items():
        try:
            policy, _ = POLICIES.get(artifact)
        except (KeyError, ValueError):
            missing.append(source)
            continue
        V = getattr(policy, "V", None)
        if V is None:
            # A tabular artifact with no V is legal (Topic 1 exports one) but
            # it cannot be plotted here. Name it rather than fall back to
            # Q.max(axis=1), which would put an optimised maximum on a chart
            # labelled as a policy evaluation.
            missing.append(source)
            continue
        by_source[source] = np.asarray(V, dtype=np.float64)
        pi = getattr(policy, "pi", None)
        grids.append(
            ValueGrid(
                label=source,
                rows=core.rows,
                cols=core.cols,
                values=[float(v) for v in by_source[source]],
                policy_arrows=arrows if pi is None else _arrow_table(core, pi),
            )
        )

    difference = None
    if "monte_carlo" in by_source and "value_iteration" in by_source:
        diff = by_source["monte_carlo"] - by_source["value_iteration"]
        difference = ValueGrid(
            label="monte_carlo minus value_iteration",
            rows=core.rows,
            cols=core.cols,
            values=[float(v) for v in diff],
        )

    return ValueMapResponse(
        grids=grids,
        difference=difference,
        gamma=float(getattr(core, "gamma", 0.95)),
        missing=missing,
    )


@app.get("/convergence", response_model=ConvergenceResponse)
def convergence() -> ConvergenceResponse:
    """The RMSE-versus-episodes curve with its confidence bands.

    Prefers the data tier and falls back to `reports/convergence.json`, which
    `train/compare.py` writes alongside the rows. The fallback exists so that a
    reviewer who clones this repository with no credentials still sees the
    chart instead of an empty panel — but it sets `degraded=True` and names its
    source, because a chart that silently serves a checked-in file is how a
    stale result gets presented as a current one.
    """
    try:
        rows = get_store().evaluations_for_metric("value_rmse", limit=2000)
    except Exception:
        rows = []

    if rows:
        by_budget: dict[int, list[float]] = {}
        for r in rows:
            if r.get("rmse") is None:
                continue
            by_budget.setdefault(int(r["at_training_episode"]), []).append(float(r["rmse"]))
        points = [
            _point_from_seeds(budget, values)
            for budget, values in sorted(by_budget.items())
            if len(values) >= 2
        ]
        if points:
            return ConvergenceResponse(points=points, source="data-tier", degraded=False)

    report = _read_report()
    if report is None:
        return ConvergenceResponse(points=[], source="none", degraded=True)

    return ConvergenceResponse(
        points=[
            ConvergencePoint(
                episodes=int(p["episodes"]),
                seeds=int(p["seeds"]),
                mean_rmse=float(p["mean_rmse"]),
                ci95_low=float(p["ci95_low"]),
                ci95_high=float(p["ci95_high"]),
                equivalent_at_5pct=bool(p["equivalent_at_5pct"]),
            )
            for p in report.get("per_budget", [])
        ],
        delta=report.get("delta"),
        episodes_to_indistinguishable=report.get("episodes_to_indistinguishable"),
        test=report.get("test"),
        source="checked-in-report",
        degraded=True,
    )


@app.post("/reload")
def reload_artifacts() -> dict[str, object]:
    """Development convenience: pick up a newly exported artifact without a restart."""
    POLICIES.reload()
    return {"loaded": POLICIES.names()}


# ---------------------------------------------------------------------------


def _arrow_table(core, pi=None) -> list[str]:
    """Greedy action per cell as an arrow, '*' for terminal cells."""
    from envs import ACTION_ARROWS

    out = []
    for s in range(core.n_states):
        if s in core.terminal_states:
            out.append("*")
        elif pi is None:
            out.append("")
        else:
            out.append(ACTION_ARROWS[int(pi[s])])
    return out


def _point_from_seeds(budget: int, values: list[float]) -> ConvergencePoint:
    """Mean RMSE with a normal-approximation band, for the CHART only.

    The chart is allowed a 1.96 here; the REPORT is not. A t quantile needs a
    distribution function the serving tier does not have — SciPy is a
    training-tier dependency and stays out of requirements-serve.txt — so the
    service draws the approximate band and `train/compare.py` computes the
    interval that gets quoted. With 10 seeds the difference is about 15%, which
    is invisible on a log-scaled chart and material in a sentence.
    """
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    stderr = float(arr.std(ddof=1) / np.sqrt(arr.size))
    return ConvergencePoint(
        episodes=budget,
        seeds=int(arr.size),
        mean_rmse=mean,
        ci95_low=mean - 1.96 * stderr,
        ci95_high=mean + 1.96 * stderr,
    )


def _read_report() -> dict | None:
    import json
    import pathlib

    path = pathlib.Path("reports/convergence.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        # A malformed report must not take down an endpoint the health banner
        # does not cover. Return "none" and let the UI say so.
        return None


def _as_vector(obs) -> np.ndarray:
    """Normalise a Gymnasium observation into the vector shape /act expects."""
    if isinstance(obs, (int, np.integer)):
        return np.asarray([float(obs)], dtype=np.float64)
    return np.asarray(obs, dtype=np.float64).ravel()


def _hash_state(state: list[float]) -> str:
    import hashlib

    return hashlib.sha256(",".join(f"{v:.6g}" for v in state).encode()).hexdigest()[:16]
