"""
api/main.py — the service tier. It owns the policy and nothing else.

Run it the way you will demonstrate it:

    uvicorn api.main:app --reload --port 8000
    open http://127.0.0.1:8000/docs

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

TOPIC 5 additions, and why each is here rather than somewhere easier:

  * `/act` gains an optional `agent` field. Naming an agent routes to a SEARCH
    PROCEDURE instead of an artifact and reads `state` as a board. One optional
    field, not a second endpoint — see shared/schemas.py.

  * `POST /game` plays a whole bounded game between two named agents on the
    server. It is an endpoint rather than a loop in Streamlit because the
    presentation tier does not own decisions: a tournament run in the browser
    produces numbers describing the browser's copy of the agents.

  * **The node budget is enforced here and reported back.** This is the one
    piece of this service that exists purely because of the deployment target.
    Every other product in this course serves a fixed-cost forward pass; a tree
    search's cost is set by the request, and an unbounded search endpoint is how
    a free-tier instance gets killed by its own users. `tests/test_node_budget.py`
    is a required test.

  * NOTE what is NOT imported: gymnasium, torch, or anything from train/. The
    search agents are pure Python and NumPy, and the learned evaluator is four
    matrix multiplies read out of a `.npz`. `GET /version` reports that honestly.
"""

from __future__ import annotations

import sys
import time
import uuid

import numpy as np
from fastapi import FastAPI, HTTPException, Response, status

from api.policy import PolicyArtifactStore
from envs.connect_four import IllegalMoveError, Position, decode_state
from search.agents import Decision, describe_agents, get_agent, get_registry
from shared.config import get_settings
from shared.schemas import (
    ActRequest,
    ActResponse,
    AgentInfo,
    AgentsResponse,
    GameRequest,
    GameResponse,
    HealthResponse,
    MatchRow,
    MoveRecord,
    PoliciesResponse,
    PolicyArtifact,
    RolloutRequest,
    RolloutResponse,
    RunSummary,
    RunsResponse,
    ScalabilityPoint,
    ScalabilityResponse,
    TournamentResponse,
    Transition,
    VersionResponse,
)
from shared.store import get_store

settings = get_settings()

app = FastAPI(
    title="Search Arena Service",
    version=settings.app_version,
    description=(
        "Owns six Connect Four agents — exhaustive search, heuristic-ordered "
        "alpha-beta, forward-pruned beam, UCT, a revised UCT, and a PUCT agent "
        "guided by a self-play network — and serves all of them from one "
        "contract under an enforced per-request node budget. NumPy artifacts "
        "only; never imports a training framework, and GET /version reports "
        "that honestly rather than asserting it."
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


@app.get("/agents", response_model=AgentsResponse)
def agents() -> AgentsResponse:
    """Every agent the service can play, with its configuration and identity.

    The UI's picker and `train/benchmark.py` both read this, so the list a human
    can play against and the list the tournament benchmarked are the same list
    by construction rather than by discipline.
    """
    rows = describe_agents()
    # The learned agent is absent until `python -m train.train` has produced its
    # archive. Naming it as unavailable is a better answer than omitting it:
    # "the agent is missing" and "the agent does not exist" are different
    # problems with different fixes, and only one of them is the student's.
    unavailable = []
    if not any(r["name"] == "alphazero" for r in rows):
        unavailable.append(
            "alphazero — no policies/alphazero_c4.npz yet; run `python -m train.train`"
        )
    return AgentsResponse(
        agents=[AgentInfo(**r) for r in rows], count=len(rows), unavailable=unavailable
    )


def _decide(req_agent: str, state: list[float], node_budget: int,
            depth: int | None) -> tuple[Decision, Position, object]:
    """Shared by `/act` and the "Play" tab: board in, `Decision` out.

    Every failure mode below is converted to a 4xx with the numbers in it,
    because this endpoint is called by a browser with a human on the other end
    of it. The three that actually happen:

      * an unknown agent name           -> 404 listing what exists
      * a board that is not a board     -> 422 saying what is wrong with it
      * a board whose game is over      -> 422, not a 500 from `search_root`

    A 500 here means the service tier has a bug. Any 500 you can predict should
    have been one of these.
    """
    try:
        agent = get_agent(req_agent)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"no agent named '{req_agent}'. "
                f"Available: {sorted(get_registry().keys())}"
            ),
        )

    try:
        position = decode_state(state)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if position.is_terminal():
        raise HTTPException(
            status_code=422,
            detail=(
                "this game is already over "
                f"(winner={position.winner}); there is no move to choose"
            ),
        )

    # The budget and the depth are per-REQUEST overrides applied to a copy of
    # the registered agent's configuration, never by mutating the registry. The
    # registry is process-global and shared across concurrent requests; mutating
    # it here would mean one caller's `depth: 8` silently became every other
    # caller's depth until the next request changed it back. That class of bug
    # only shows up under concurrency, which is to say in production.
    scoped = _scoped_agent(agent, node_budget=node_budget, depth=depth)
    return scoped.choose(position), position, scoped


def _scoped_agent(agent, *, node_budget: int, depth: int | None):
    """A shallow copy of `agent` with per-request overrides applied.

    `copy.copy` rather than a constructor call because the agents in the
    registry are of several different classes with different constructor
    signatures, and a factory that had to know all of them would be a second
    registry. The copy is shallow, so the MCTS instance is shared — which is
    fine for `node_budget` (read per search) and NOT fine for `iterations`,
    which is why iterations is not an overridable field.
    """
    import copy

    scoped = copy.copy(agent)
    if hasattr(scoped, "node_budget"):
        scoped.node_budget = int(node_budget)
    if depth is not None and hasattr(scoped, "depth"):
        scoped.depth = int(depth)
    if hasattr(scoped, "_mcts"):
        # The MCTS object holds its own budget. Copy it too, or the override
        # leaks into every other request that shares the registry instance.
        scoped._mcts = copy.copy(scoped._mcts)
        scoped._mcts.node_budget = int(node_budget)
    return scoped


@app.post("/act", response_model=ActResponse)
def act(req: ActRequest) -> ActResponse:
    t0 = time.perf_counter()

    # -- the search path ----------------------------------------------------
    # Taken only when the caller named an agent. Everything below this block is
    # the base template's artifact path, unchanged, and it still works: the
    # standing /act tests in tests/test_act_schema.py exercise it.
    if req.agent is not None:
        decision, position, scoped = _decide(
            req.agent, req.state, req.node_budget, req.depth
        )
        # The invariant the required test asserts. Checked here rather than only
        # in the test, because the budget is a promise to the caller and a
        # promise the service does not verify is a comment. `raise` and not
        # `assert`: assertions are stripped by `python -O`, and this is the one
        # check that must survive into a production interpreter.
        if decision.nodes_expanded > req.node_budget:
            raise RuntimeError(
                f"node budget violated: expanded {decision.nodes_expanded} with a "
                f"budget of {req.node_budget} — see search/budget.py"
            )
        try:
            get_store().log_audit(
                {
                    "endpoint": "/act",
                    "policy_sha256": scoped.identity(),
                    "state_hash": _hash_state(req.state),
                    "action": str(decision.move),
                }
            )
        except Exception:
            pass  # telemetry must never take down the request path

        return ActResponse(
            action=decision.move,
            policy_name=req.agent,
            agent=req.agent,
            policy_sha256=scoped.identity(),
            value_estimate=decision.value,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            nodes_expanded=decision.nodes_expanded,
            node_budget=req.node_budget,
            budget_exhausted=decision.budget_exhausted,
            search_depth=decision.search_depth,
            legal_moves=position.legal_moves(),
        )

    # -- the artifact path (base template, unchanged) -----------------------
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
            try:
                action, _ = policy.act(state_vec, deterministic=True)
            except ValueError as exc:
                # A shape mismatch between the environment's observation and the
                # artifact's expected input. This is a REAL case in this topic
                # and not a hypothetical: the environment emits the 43-float
                # board encoding, while `alphazero_c4.npz` consumes the 84-float
                # canonical planes. It was a 500 until someone tried it.
                #
                # 422 with both numbers and a pointer to the endpoint that does
                # what the caller meant. `/rollout` evaluates an ARTIFACT against
                # the environment; `/game` plays named AGENTS against each other,
                # and the learned agent is a search procedure wrapped around this
                # artifact rather than the artifact itself.
                env.close()
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"policy '{req.policy_name}' cannot consume this "
                        f"environment's observation: {exc}. If you meant to "
                        f"evaluate the learned AGENT rather than the raw "
                        f"network, use POST /game with agent_a='alphazero'."
                    ),
                )
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


@app.post("/game", response_model=GameResponse)
def game(req: GameRequest) -> GameResponse:
    """Play one complete, bounded game between two named agents, server-side.

    The bounds, and what each one stops:

      * `max_moves`     — the game cannot run forever. Currently unreachable in
                          Connect Four (42 cells), and here anyway, because it
                          stops being unreachable the moment someone changes the
                          board size, which is exactly what the scalability
                          study invites them to do.
      * `node_budget`   — no single decision can run away. Applied identically
                          to both agents; a tournament where one side had a
                          larger budget is a tournament about budgets.
      * both together   — the request has a computable worst case, which is what
                          makes it safe to expose to the internet at all.

    Returns the budget ACTUALLY CONSUMED per agent, plus how many decisions were
    truncated. A game with truncated decisions is a game between weaker agents
    than the ones you named, and the caller has to be able to see that.
    """
    for name in (req.agent_a, req.agent_b):
        if name not in get_registry():
            raise HTTPException(
                status_code=404,
                detail=f"no agent named '{name}'. Available: {sorted(get_registry())}",
            )

    # Fresh copies with the request's budget. Note that both agents are scoped
    # with the SAME budget — see the docstring.
    a = _scoped_agent(get_agent(req.agent_a), node_budget=req.node_budget, depth=None)
    b = _scoped_agent(get_agent(req.agent_b), node_budget=req.node_budget, depth=None)

    # Reseed the stochastic agents from the request seed so that a `/game` call
    # is reproducible from the two names and one integer. Without this, the
    # process-global registry's RNG state makes the Nth call to /game depend on
    # the N-1 calls before it — which is a delightful bug to debug from a
    # tournament table.
    _reseed(a, req.seed)
    _reseed(b, req.seed if req.seed is None else req.seed + 1)

    position = Position()
    records: list[MoveRecord] = []
    totals = {req.agent_a: 0, req.agent_b: 0}
    times = {req.agent_a: 0.0, req.agent_b: 0.0}
    depths = {req.agent_a: 0, req.agent_b: 0}
    counts = {req.agent_a: 0, req.agent_b: 0}
    truncated = 0

    for ply in range(req.max_moves):
        if position.is_terminal():
            break
        mover, name = (a, req.agent_a) if position.player == 1 else (b, req.agent_b)
        decision = mover.choose(position)
        try:
            position.push(decision.move)
        except IllegalMoveError as exc:
            # An agent proposing an illegal column is a defect in that agent,
            # not a bad request from the caller. 500 with the agent named, so
            # the traceback points at the right file.
            raise HTTPException(
                status_code=500,
                detail=f"agent '{name}' proposed illegal column {decision.move}: {exc}",
            )
        totals[name] += decision.nodes_expanded
        times[name] += decision.wall_clock_ms
        depths[name] = max(depths[name], decision.search_depth)
        counts[name] += 1
        truncated += int(decision.budget_exhausted)
        if req.record_moves:
            records.append(
                MoveRecord(
                    ply=ply,
                    agent=name,
                    column=decision.move,
                    nodes_expanded=decision.nodes_expanded,
                    wall_clock_ms=decision.wall_clock_ms,
                    search_depth=decision.search_depth,
                    value=decision.value,
                    budget_exhausted=decision.budget_exhausted,
                )
            )

    winner_piece = position.winner
    if winner_piece is None:
        # Only reachable if `max_moves` bound before the board filled. Scored as
        # a draw and NOT silently: an unfinished game recorded as a draw would
        # bias every aggregate towards the agent that was losing when the clock
        # stopped.
        result, winner = "draw", None
    elif winner_piece == 0:
        result, winner = "draw", None
    elif winner_piece == 1:
        result, winner = "agent_a", req.agent_a
    else:
        result, winner = "agent_b", req.agent_b

    game_id = str(uuid.uuid4())
    logged = False
    if req.log_to_store:
        # Logged from agent_a's point of view, once. See shared/schemas.py's
        # GameRow docstring for why this is one row and not two.
        row = {
            "game_id": game_id,
            "agent": req.agent_a,
            "opponent": req.agent_b,
            "result": {"agent_a": "win", "agent_b": "loss", "draw": "draw"}[result],
            "agent_played_first": True,
            "moves": position.n_pieces,
            "nodes_expanded": totals[req.agent_a],
            "wall_clock_ms": times[req.agent_a],
            "search_depth": depths[req.agent_a],
            # Both halves, matching what train/benchmark.py writes. A row from
            # /game and a row from the harness must be indistinguishable to a
            # query, or the `games` table means two different things depending
            # on which process wrote it.
            "opponent_nodes_expanded": totals[req.agent_b],
            "opponent_wall_clock_ms": times[req.agent_b],
            "opponent_search_depth": depths[req.agent_b],
            "node_budget": req.node_budget,
            "budget_exhausted_moves": truncated,
            "seed": req.seed,
        }
        try:
            get_store().insert_games([row])
            logged = True
        except Exception:
            # A failed write must not fail the game. The response says
            # `logged: false` and the caller can decide what that means.
            logged = False

    return GameResponse(
        game_id=game_id,
        agent_a=req.agent_a,
        agent_b=req.agent_b,
        result=result,  # type: ignore[arg-type]
        winner=winner,
        moves=position.n_pieces,
        move_records=records,
        final_board=position.to_grid(),
        nodes_expanded=totals,
        wall_clock_ms={k: round(v, 3) for k, v in times.items()},
        mean_nodes_per_move={
            k: (totals[k] / counts[k] if counts[k] else 0.0) for k in totals
        },
        max_search_depth=depths,
        node_budget=req.node_budget,
        budget_exhausted_moves=truncated,
        seed=req.seed,
        logged=logged,
    )


@app.get("/tournament", response_model=TournamentResponse)
def tournament() -> TournamentResponse:
    """The round-robin win-rate matrix the "Tournament" tab renders.

    Prefers the data tier and falls back to `reports/benchmark.json`, which
    `train/benchmark.py` writes alongside the rows. The fallback exists so a
    reviewer who clones this repository with no credentials still sees the
    table instead of an empty panel — but it sets `degraded=True` and names its
    source, because a table that silently serves a checked-in file is how a
    stale result gets presented as a current one.
    """
    try:
        rows = get_store().match_rows(limit=500)
    except Exception:
        rows = []

    source = "data-tier"
    if not rows:
        report = _read_report()
        rows = (report or {}).get("matches", [])
        source = "checked-in-report" if rows else "none"

    matches = [
        MatchRow(
            agent=str(r["agent"]),
            opponent=str(r["opponent"]),
            games=int(r["games"]),
            wins=int(r["wins"]),
            draws=int(r["draws"]),
            losses=int(r["losses"]),
            win_rate=float(r["win_rate"]),
            mean_nodes=float(r.get("mean_nodes", 0.0)),
            mean_ms=float(r.get("mean_ms", 0.0)),
            mean_peak_kib=r.get("mean_peak_kib"),
        )
        for r in rows
    ]
    names = sorted({m.agent for m in matches} | {m.opponent for m in matches})
    return TournamentResponse(
        matches=matches,
        agents=names,
        total_games=sum(m.games for m in matches) // 2 if matches else 0,
        source=source,  # type: ignore[arg-type]
        degraded=(source != "data-tier"),
    )


@app.get("/scalability", response_model=ScalabilityResponse)
def scalability() -> ScalabilityResponse:
    """Node count and wall clock against depth, per search variant.

    This is the measurement that justifies the whole topic: the exhaustive
    curve is b^d and the alpha-beta curve is closer to b^(d/2), and seeing the
    two on one axis is more convincing than any amount of asymptotic notation.
    """
    try:
        rows = get_store().probe_rows(limit=500)
    except Exception:
        rows = []
    source = "data-tier"
    if not rows:
        report = _read_report()
        rows = (report or {}).get("scalability", [])
        source = "checked-in-report" if rows else "none"

    return ScalabilityResponse(
        points=[
            ScalabilityPoint(
                variant=str(r["variant"]),
                depth=int(r["depth"]),
                nodes=int(r["nodes"]),
                wall_clock_ms=float(r["wall_clock_ms"]),
                peak_kib=r.get("peak_kib"),
                completed=bool(r.get("completed", True)),
            )
            for r in rows
        ],
        source=source,  # type: ignore[arg-type]
        degraded=(source != "data-tier"),
    )


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


def _hash_state(state: list[float]) -> str:
    import hashlib

    return hashlib.sha256(",".join(f"{v:.6g}" for v in state).encode()).hexdigest()[:16]


def _reseed(agent, seed: int | None) -> None:
    """Point a stochastic agent's RNG at a fresh generator for this request.

    Only touches agents that actually have one. A `hasattr` chain rather than
    an `isinstance` chain because the registry holds several classes and one
    wrapper, and the wrapper's `.rng` is on its inner agent — asking "does it
    have an rng" is the question we actually mean.
    """
    if seed is None:
        return
    if hasattr(agent, "rng"):
        agent.rng = np.random.default_rng(seed)
    if hasattr(agent, "_mcts"):
        agent._mcts.rng = np.random.default_rng(seed)
    if hasattr(agent, "inner"):
        _reseed(agent.inner, seed)


def _read_report() -> dict | None:
    """`reports/benchmark.json`, or None.

    A malformed report must not take down an endpoint the health banner does not
    cover, so a parse failure is treated as "no report" and the response says
    `source: none`. Silently returning half a table would be worse.
    """
    import json
    import pathlib

    path = pathlib.Path("reports/benchmark.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None
