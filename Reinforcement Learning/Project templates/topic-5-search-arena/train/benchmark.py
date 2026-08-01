"""
train/benchmark.py — the round-robin tournament, the scalability sweep, and the
node-count table. Every number in the README comes from here.

    python -m train.benchmark --games 30                 # the sandbox default
    python -m train.benchmark --games 200                # the submitted budget
    python -m train.benchmark --agents random,heuristic,mcts,mcts_v2
    python -m train.benchmark --scalability-only

TRAINING TIER by convention rather than by dependency: it imports nothing from
torch, but it runs for minutes and writes thousands of rows, which is not a
thing a web request does. Keeping it out of `api/` is what stops someone
"helpfully" exposing it as an endpoint.

WHAT IT MEASURES, and why each one is in the list
--------------------------------------------------
The syllabus asks for four things. They are four because no two of them can be
traded for each other without saying so:

  computation time   wall-clock milliseconds per decision. The number a user
                     feels. Machine-dependent, so it is reported alongside...
  node count         positions expanded per decision. The machine-independent
                     cost, and the one that transfers to someone else's laptop.
  decision quality   win rate against the FIXED reference opponent, plus the
                     head-to-head matrix. Fixed, because a win rate against
                     "my other agent" is a moving target.
  peak memory        peak allocation during the game, via `tracemalloc`. MCTS
                     holds a tree; alpha-beta holds a stack. That is a real
                     difference and it deserves a measured number rather than
                     an assertion.
  scalability        all of the above as depth grows. This is the one that
                     makes the topic: b^d and b^(d/2) look the same in a
                     sentence and nothing like each other on an axis.

THE BUDGET, honestly
---------------------
The syllabus asks for at least 200 games per pairing. The default here is 30,
because a template whose first run takes forty minutes is a template nobody
runs. `--games 200` is the number for the run you submit; on the default agent
set that is 6 pairings x 200 = 1,200 games and roughly 25 minutes. Say which
budget produced the table you are showing — a win rate over 30 games has a
standard error of about 9 percentage points, which is wider than most of the
differences you will be tempted to explain.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import time
import tracemalloc
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from envs.connect_four import Position
from search.agents import Agent, get_agent, get_registry
from search.minimax import search_root
from search.ordering import heuristic_ordering, make_beam_ordering, natural_ordering

DEFAULT_AGENTS = ["random", "heuristic", "mcts", "mcts_v2"]


@dataclass
class GameOutcome:
    row: dict[str, Any]
    result: str


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------


def play_game(
    agent_a: Agent,
    agent_b: Agent,
    *,
    a_moves_first: bool,
    seed: int,
    node_budget: int,
    experiment_id: str | None = None,
    measure_memory: bool = True,
) -> GameOutcome:
    """One game, fully instrumented, logged from `agent_a`'s point of view.

    `a_moves_first` alternates across the pairing. Connect Four is a first-player
    win under perfect play and moving first is worth several points of win rate
    even between ordinary agents, so a pairing played with one agent always
    yellow measures the opening as much as the agent. The `agent_played_first`
    column exists so that this is checkable after the fact rather than trusted.

    THE COSTS OF BOTH SIDES ARE RECORDED. The `result` column is from `agent`'s
    point of view and the game is logged once, but `nodes_expanded` /
    `wall_clock_ms` / `search_depth` / `peak_kib` come in pairs — one set for
    `agent`, one for `opponent`. That is not symmetry for its own sake: without
    the opponent's half, the win-rate matrix can only report costs for the
    direction the pairing happened to be enumerated in, and every cell in the
    other triangle shows a mean node count of zero. (It did, in the first
    version of this harness. The matrix looked fine; the cost table did not.)

    MEMORY IS MEASURED PER DECISION, NOT PER GAME, and this took two attempts:

      * `resource.getrusage(RUSAGE_SELF).ru_maxrss` is a high-water mark for
        the whole PROCESS and never goes down, so after the first MCTS game
        every later game reports the same number — including the alpha-beta
        games, which then appear to use exactly as much memory as MCTS.
      * `tracemalloc` bracketing the whole GAME measures both agents together,
        so a cheap agent playing an expensive one inherits the expensive one's
        footprint. The random agent came out at 258 KiB a game, which is a
        statement about its opponents.

    So: `tracemalloc` is started once per game, and around each decision we take
    `peak_after - current_before`. Both halves of that subtraction matter.
    `reset_peak()` resets the peak to the CURRENT traced total, not to zero, so
    reading the raw peak reports every live allocation the game has accumulated
    — which made the random agent look like it used 235 KiB a move. The
    difference is the allocation the decision itself required, which is the
    question. The recorded value is the maximum across that agent's decisions,
    because what has to fit in the container is the worst move, not the average.

    All of this costs roughly 2x in wall clock, which is why `--no-memory`
    exists and why the timing table in the README says which mode produced it.
    """
    _reseed(agent_a, seed)
    _reseed(agent_b, seed + 7919)   # a different prime, so the two never align

    position = Position()
    nodes = {"a": 0, "b": 0}
    millis = {"a": 0.0, "b": 0.0}
    depth = {"a": 0, "b": 0}
    peak = {"a": 0.0, "b": 0.0}
    truncated = {"a": 0, "b": 0}
    moves = 0

    if measure_memory:
        tracemalloc.start()

    while not position.is_terminal():
        a_to_move = (position.player == 1) == a_moves_first
        key = "a" if a_to_move else "b"
        agent = agent_a if a_to_move else agent_b
        if measure_memory:
            baseline = tracemalloc.get_traced_memory()[0]
            tracemalloc.reset_peak()
        decision = agent.choose(position)
        if measure_memory:
            high_water = tracemalloc.get_traced_memory()[1]
            peak[key] = max(peak[key], (high_water - baseline) / 1024.0)
        position.push(decision.move)
        nodes[key] += decision.nodes_expanded
        millis[key] += decision.wall_clock_ms
        depth[key] = max(depth[key], decision.search_depth)
        truncated[key] += int(decision.budget_exhausted)
        moves += 1

    if measure_memory:
        tracemalloc.stop()

    winner = position.winner
    if winner == 0:
        result = "draw"
    else:
        a_piece = 1 if a_moves_first else -1
        result = "win" if winner == a_piece else "loss"

    row = {
        "game_id": str(uuid.uuid4()),
        "experiment_id": experiment_id,
        "agent": agent_a.name,
        "opponent": agent_b.name,
        "result": result,
        "agent_played_first": a_moves_first,
        "moves": moves,
        # Paired, never summed. A combined total would make every row a
        # statement about the pairing rather than about either agent, and the
        # pairing is what `matches` is for.
        "nodes_expanded": nodes["a"],
        "wall_clock_ms": round(millis["a"], 3),
        "search_depth": depth["a"],
        "peak_kib": None if not measure_memory else round(peak["a"], 1),
        "opponent_nodes_expanded": nodes["b"],
        "opponent_wall_clock_ms": round(millis["b"], 3),
        "opponent_search_depth": depth["b"],
        "opponent_peak_kib": None if not measure_memory else round(peak["b"], 1),
        "node_budget": node_budget,
        "budget_exhausted_moves": truncated["a"] + truncated["b"],
        "seed": seed,
    }
    return GameOutcome(row=row, result=result)


def _reseed(agent: Agent, seed: int) -> None:
    if hasattr(agent, "rng"):
        agent.rng = np.random.default_rng(seed)
    if hasattr(agent, "_mcts"):
        agent._mcts.rng = np.random.default_rng(seed)
    if hasattr(agent, "inner"):
        _reseed(agent.inner, seed)


# ---------------------------------------------------------------------------
# The round robin
# ---------------------------------------------------------------------------


def round_robin(
    names: list[str],
    games_per_pairing: int,
    *,
    node_budget: int,
    seed: int = 0,
    experiment_id: str | None = None,
    measure_memory: bool = True,
    progress: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Every unordered pair, `games_per_pairing` games each. Returns (games, matches).

    UNORDERED pairs, with the first move alternating inside the pairing. The
    alternative — every ordered pair, so that (a, b) and (b, a) are separate
    match-ups — doubles the runtime to measure the same thing twice, because
    the colour advantage is already averaged out by the alternation. What it
    would buy is a per-colour breakdown; if you want that, group by
    `agent_played_first` in SQL, which is exactly why that column exists.
    """
    rows: list[dict] = []
    tallies: dict[tuple[str, str], dict[str, Any]] = {}

    pairings = list(itertools.combinations(names, 2))
    for pi, (a_name, b_name) in enumerate(pairings):
        agent_a, agent_b = get_agent(a_name), get_agent(b_name)
        t0 = time.perf_counter()
        for g in range(games_per_pairing):
            outcome = play_game(
                agent_a,
                agent_b,
                a_moves_first=(g % 2 == 0),
                # Seed derived from the pairing index and the game index, so a
                # re-run reproduces game-for-game and a single game can be
                # replayed from the two names and one integer. A global
                # incrementing counter would make game 400 depend on games
                # 1-399 having been played first.
                seed=seed + 1000 * pi + g,
                node_budget=node_budget,
                experiment_id=experiment_id,
                measure_memory=measure_memory,
            )
            rows.append(outcome.row)
            _tally(tallies, a_name, b_name, outcome)
        if progress:
            print(
                f"  {a_name} vs {b_name}: {games_per_pairing} games in "
                f"{time.perf_counter() - t0:.1f}s"
            )

    matches = [_finish_tally(t, experiment_id) for t in tallies.values()]
    return rows, matches


def _tally(tallies, a_name, b_name, outcome: GameOutcome) -> None:
    """Accumulate BOTH directions of the pairing from one game row.

    This mirrors the `win_rate_matrix` view in db/migrations/002_topic5.sql, and
    the mirroring is on purpose: two independent routes to the same aggregate is
    not duplication, it is the check. If the table this writes and the view
    disagree, one of them has a bug and you would rather find that in a diff.
    """
    forward = tallies.setdefault(
        (a_name, b_name),
        {"agent": a_name, "opponent": b_name, "games": 0, "wins": 0, "draws": 0,
         "losses": 0, "nodes": [], "ms": [], "peak": []},
    )
    reverse = tallies.setdefault(
        (b_name, a_name),
        {"agent": b_name, "opponent": a_name, "games": 0, "wins": 0, "draws": 0,
         "losses": 0, "nodes": [], "ms": [], "peak": []},
    )
    forward["games"] += 1
    reverse["games"] += 1
    if outcome.result == "win":
        forward["wins"] += 1
        reverse["losses"] += 1
    elif outcome.result == "loss":
        forward["losses"] += 1
        reverse["wins"] += 1
    else:
        forward["draws"] += 1
        reverse["draws"] += 1
    # Each direction gets ITS OWN agent's costs. Mixing them up — putting the
    # opponent's node count under the agent's name — produces a table in which
    # the random agent appears to expand 40,000 nodes a move, and leaving the
    # reverse direction empty produces one in which half the cells are zero.
    # Both were mistakes made while writing this file; the pairing is stored
    # once and read twice, so this is the only place the attribution can go
    # wrong.
    # Per DECISION, not per game. A game between a fast agent and a slow one
    # has an unequal number of decisions for each side only when the move count
    # is odd, and whoever moved first made the extra one — which is why
    # `agent_played_first` is needed here and not just for the win-rate split.
    total = outcome.row["moves"]
    first, second = (total + 1) // 2, total // 2
    moves_a, moves_b = (
        (first, second) if outcome.row["agent_played_first"] else (second, first)
    )
    moves_a, moves_b = max(moves_a, 1), max(moves_b, 1)
    forward["nodes"].append(outcome.row["nodes_expanded"] / moves_a)
    forward["ms"].append(outcome.row["wall_clock_ms"] / moves_a)
    reverse["nodes"].append(outcome.row["opponent_nodes_expanded"] / moves_b)
    reverse["ms"].append(outcome.row["opponent_wall_clock_ms"] / moves_b)
    if outcome.row.get("peak_kib") is not None:
        forward["peak"].append(outcome.row["peak_kib"])
        reverse["peak"].append(outcome.row["opponent_peak_kib"])


def _finish_tally(t: dict, experiment_id: str | None) -> dict:
    games = max(t["games"], 1)
    return {
        "experiment_id": experiment_id,
        "agent": t["agent"],
        "opponent": t["opponent"],
        "games": t["games"],
        "wins": t["wins"],
        "draws": t["draws"],
        "losses": t["losses"],
        # Draws count half — the same definition as the SQL view and as
        # train/train.py. One definition, three places that must not drift.
        "win_rate": (t["wins"] + 0.5 * t["draws"]) / games,
        # PER DECISION, not per game. Per-game totals would make an agent that
        # survives longer look more expensive than one that loses quickly, which
        # is the opposite of what the column is meant to say.
        "mean_nodes": float(np.mean(t["nodes"])) if t["nodes"] else 0.0,
        "mean_ms": float(np.mean(t["ms"])) if t["ms"] else 0.0,
        # Peak allocation during a single DECISION, averaged over games. See
        # play_game for the two wrong ways to measure this.
        "mean_peak_kib": float(np.mean(t["peak"])) if t["peak"] else None,
    }


# ---------------------------------------------------------------------------
# The node-count table and the scalability sweep
# ---------------------------------------------------------------------------
#
# The four variants below are the answer to "report the node counts for each".
# Reading them in order is the lesson:
#
#   exhaustive     natural order, NO pruning     — the honest baseline, b^d
#   natural+ab     natural order, pruning        — what pruning alone buys
#   heuristic+ab   heuristic order, pruning      — what ORDERING buys on top
#   beam3          heuristic order, top 3 only   — what forward pruning buys,
#                                                  and it is not sound
#
# The row that surprises people is missing on purpose: "heuristic ordering, no
# pruning" is not listed, because it visits EXACTLY as many nodes as
# `exhaustive`. Ordering with nothing to cut off is a no-op on node count.
# Alpha-beta is what converts good ordering into saved work, and that is the
# whole relationship between the two ideas.

VARIANTS = {
    "exhaustive":   dict(ordering=natural_ordering,   alpha_beta=False),
    "natural+ab":   dict(ordering=natural_ordering,   alpha_beta=True),
    "heuristic+ab": dict(ordering=heuristic_ordering, alpha_beta=True),
    "beam3":        dict(ordering=make_beam_ordering(3), alpha_beta=True),
}

# Two positions, because a node count from the empty board is not a node count
# from a real game. The midgame position is more crowded, so terminal detection
# fires earlier and pruning has more to work with.
PROBE_POSITIONS = {
    "empty": [],
    "midgame": [3, 3, 2, 4, 4, 2, 5, 1],
}


def scalability_sweep(
    depths: list[int],
    *,
    node_budget: int = 3_000_000,
    positions: dict[str, list[int]] | None = None,
    experiment_id: str | None = None,
    progress: bool = True,
) -> list[dict]:
    """Node count and wall clock for every (variant, depth, position).

    `strict_budget=True` here, unlike everywhere else in the product. A probe
    that ran out of budget did not measure the configuration it was asked
    about; recording its truncated node count as though it were the real one
    would put a fake plateau on the scalability chart exactly where the
    interesting growth is. So it is recorded as `completed=False` with the
    budget as its node count, and the chart is required to render that
    differently.
    """
    from search.budget import NodeBudgetExceeded

    rows: list[dict] = []
    for label, setup_moves in (positions or PROBE_POSITIONS).items():
        base = Position()
        for col in setup_moves:
            base.push(col)
        for variant, kwargs in VARIANTS.items():
            for depth in depths:
                position = base.copy()
                tracemalloc.start()
                completed = True
                try:
                    _, _, stats = search_root(
                        position, depth, node_budget=node_budget,
                        strict_budget=True, **kwargs,
                    )
                    nodes, ms = stats.nodes, stats.elapsed_ms
                    leaves, cutoffs = stats.leaves, stats.cutoffs
                except NodeBudgetExceeded:
                    completed = False
                    nodes, ms, leaves, cutoffs = node_budget, float("nan"), 0, 0
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                rows.append({
                    "experiment_id": experiment_id,
                    "variant": variant,
                    "depth": depth,
                    "nodes": nodes,
                    "leaves": leaves,
                    "cutoffs": cutoffs,
                    "wall_clock_ms": 0.0 if ms != ms else round(ms, 3),  # NaN -> 0
                    "peak_kib": round(peak / 1024.0, 1),
                    "completed": completed,
                    "position_label": label,
                })
        if progress:
            print(f"  scalability sweep done for position '{label}'")
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Round-robin tournament and scalability sweep.")
    ap.add_argument("--games", type=int, default=30,
                    help="games per pairing. The syllabus budget is 200; 30 is the "
                         "sandbox default so a first run finishes.")
    ap.add_argument("--agents", default=",".join(DEFAULT_AGENTS),
                    help="comma-separated agent names, or 'all'")
    ap.add_argument("--node-budget", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depths", default="1,2,3,4,5,6",
                    help="depths for the scalability sweep")
    ap.add_argument("--no-memory", action="store_true",
                    help="skip tracemalloc; roughly 2x faster and no peak_kib column")
    ap.add_argument("--scalability-only", action="store_true")
    ap.add_argument("--tournament-only", action="store_true")
    ap.add_argument("--out", default="reports/benchmark.json")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    from shared.store import get_store

    store = get_store()
    registry = get_registry()
    names = (
        list(registry.keys())
        if args.agents == "all"
        else [n.strip() for n in args.agents.split(",") if n.strip()]
    )
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise SystemExit(
            f"unknown agent(s) {unknown}. Available: {sorted(registry)}. "
            "('alphazero' appears only after `python -m train.train`.)"
        )
    if len(names) < 2 and not args.scalability_only:
        raise SystemExit("a round robin needs at least two agents")

    experiment_id = store.insert_experiment({
        "algorithm": "search_tournament",
        "env_id": "ConnectFour-6x7-v1",
        "seed": args.seed,
        "hyperparameters": {
            "agents": names,
            "games_per_pairing": args.games,
            "node_budget": args.node_budget,
            "measure_memory": not args.no_memory,
            "agent_configs": {n: registry[n].config() for n in names},
        },
    })

    summary: dict[str, Any] = {
        "experiment_id": experiment_id,
        "agents": names,
        "games_per_pairing": args.games,
        "node_budget": args.node_budget,
        "seed": args.seed,
    }

    games_rows: list[dict] = []
    matches: list[dict] = []
    if not args.scalability_only:
        if not args.quiet:
            n_pairs = len(names) * (len(names) - 1) // 2
            print(
                f"round robin: {len(names)} agents, {n_pairs} pairings, "
                f"{args.games} games each = {n_pairs * args.games} games"
            )
        t0 = time.perf_counter()
        games_rows, matches = round_robin(
            names,
            args.games,
            node_budget=args.node_budget,
            seed=args.seed,
            experiment_id=experiment_id,
            measure_memory=not args.no_memory,
            progress=not args.quiet,
        )
        store.insert_games(games_rows)
        store.insert_matches(matches)
        summary["matches"] = matches
        summary["total_games"] = len(games_rows)
        summary["tournament_seconds"] = round(time.perf_counter() - t0, 1)
        if not args.quiet:
            _print_matrix(names, matches)

    probes: list[dict] = []
    if not args.tournament_only:
        depths = [int(d) for d in args.depths.split(",") if d.strip()]
        if not args.quiet:
            print(f"scalability sweep: depths {depths}")
        probes = scalability_sweep(
            depths, experiment_id=experiment_id, progress=not args.quiet
        )
        store.insert_probes(probes)
        summary["scalability"] = probes
        if not args.quiet:
            _print_node_counts(probes)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(summary, indent=2))
    if not args.quiet:
        print(f"\nwrote {len(games_rows)} games, {len(matches)} match rows, "
              f"{len(probes)} probes; summary -> {args.out}")
    return summary


def _print_matrix(names: list[str], matches: list[dict]) -> None:
    lookup = {(m["agent"], m["opponent"]): m for m in matches}
    width = max(len(n) for n in names) + 2
    print("\nwin rate (row agent vs column opponent; draws count half)")
    print(" " * width + "".join(f"{n:>12s}" for n in names))
    for a in names:
        cells = []
        for b in names:
            m = lookup.get((a, b))
            cells.append(f"{m['win_rate']:>12.3f}" if m else f"{'-':>12s}")
        print(f"{a:<{width}s}" + "".join(cells))


def _print_node_counts(probes: list[dict]) -> None:
    empty = [p for p in probes if p["position_label"] == "empty"]
    if not empty:
        return
    depths = sorted({p["depth"] for p in empty})
    print("\nnodes expanded from the empty board")
    print(f"{'variant':<14s}" + "".join(f"{'d=' + str(d):>12s}" for d in depths))
    for variant in VARIANTS:
        cells = []
        for d in depths:
            row = next((p for p in empty if p["variant"] == variant and p["depth"] == d), None)
            cells.append(f"{row['nodes']:>12,d}" if row and row["completed"]
                         else f"{'>budget':>12s}")
        print(f"{variant:<14s}" + "".join(cells))


if __name__ == "__main__":
    main()
