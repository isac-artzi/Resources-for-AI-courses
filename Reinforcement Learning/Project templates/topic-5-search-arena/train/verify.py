"""
train/verify.py — the four correctness checks, at the budget you would actually
defend rather than the budget a test suite can afford.

    python -m train.verify                    # all four, ~3 minutes
    python -m train.verify --positions 2000 --reference-games 500
    python -m train.verify --only alpha-beta

This is not a duplicate of `tests/`. The test suite runs on every commit and
must stay fast, so it checks each of these properties at a budget of a hundred
or so trials — enough to catch a broken implementation, not enough to quote.
This script is what produces the numbers in the README, and the difference
between the two is the difference between "the tests pass" and "here is the
evidence".

A game engine is one of the few kinds of software where correctness is
CHECKABLE rather than merely plausible, and that is worth exploiting. Each
check below compares something against something else that must agree with it:

  1. alpha-beta == plain full-width minimax, over N random positions.
     Catches: a wrong cutoff. A wrong cutoff does not crash and does not
     obviously misplay; it silently returns the wrong value in the small
     fraction of positions where it fires.

  2. the strong search never LOSES to random, over N games.
     Catches: a sign error, a broken terminal check, a mask bug. An agent with
     any of those still plays legal moves and still wins sometimes.

  3. MCTS at a high simulation budget beats random by a wide margin.
     Catches: the missing negation in backpropagation, which produces an agent
     that is confidently wrong on half the plies and reads as "MCTS is weak".

  4. heuristic ordering reduces the node count under alpha-beta.
     Catches: an ordering function that is sorting the wrong way round — which
     makes the search SLOWER, and is invisible unless you count.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from envs.connect_four import Position
from search.agents import MCTSAgent, MinimaxAgent, RandomAgent
from search.minimax import plain_minimax_value, search_root
from search.ordering import heuristic_ordering, make_beam_ordering, natural_ordering


def random_positions(n: int, seed: int = 0, max_plies: int = 16) -> list[Position]:
    """`n` reachable, non-terminal positions from random play.

    RANDOM PLAY rather than random board fills, because a randomly filled board
    is usually not reachable — floating pieces, impossible piece counts, two
    completed fours — and a value computed from an unreachable position is a
    value about a game nobody is playing. `Position.from_board` would reject
    most of them, which is the same argument stated by the type system.

    The ply count varies so the sample spans openings and midgames. A check run
    entirely on 4-ply positions verifies alpha-beta on the easiest tenth of the
    state space.
    """
    rng = np.random.default_rng(seed)
    out: list[Position] = []
    guard = 0
    while len(out) < n and guard < n * 20:
        guard += 1
        position = Position()
        plies = int(rng.integers(0, max_plies + 1))
        ok = True
        for _ in range(plies):
            moves = position.legal_moves()
            if not moves:
                ok = False
                break
            position.push(int(moves[rng.integers(len(moves))]))
            if position.is_terminal():
                ok = False
                break
        if ok and not position.is_terminal():
            out.append(position)
    return out


# ---------------------------------------------------------------------------
# 1. alpha-beta equivalence
# ---------------------------------------------------------------------------


def check_alpha_beta(positions: int = 1000, depth: int = 4, seed: int = 0) -> dict:
    """alpha-beta must return the IDENTICAL value to full-width minimax.

    Identical, not close. These are the same arithmetic over the same integers
    in a different order, so `==` on the floats is the right comparison and a
    tolerance would hide exactly the bug being looked for.

    Also checks the heuristic-ordered variant, because ordering is
    value-preserving by construction (it returns a permutation) and a "heuristic
    ordering" that silently dropped a move would break that. The beam variant is
    deliberately NOT checked here — it is unsound by design, and asserting it
    agrees would be asserting the opposite of what it is for.
    """
    sample = random_positions(positions, seed=seed)
    t0 = time.perf_counter()
    mismatches = []
    nodes_full = nodes_ab = nodes_ordered = 0

    for i, position in enumerate(sample):
        reference = plain_minimax_value(position.copy(), depth)
        _, v_ab, s_ab = search_root(
            position.copy(), depth, ordering=natural_ordering,
            alpha_beta=True, node_budget=10**9,
        )
        _, v_ord, s_ord = search_root(
            position.copy(), depth, ordering=heuristic_ordering,
            alpha_beta=True, node_budget=10**9,
        )
        _, _, s_full = search_root(
            position.copy(), depth, ordering=natural_ordering,
            alpha_beta=False, node_budget=10**9,
        )
        nodes_full += s_full.nodes
        nodes_ab += s_ab.nodes
        nodes_ordered += s_ord.nodes
        if v_ab != reference or v_ord != reference:
            mismatches.append(
                {"index": i, "reference": reference, "alpha_beta": v_ab,
                 "ordered": v_ord, "history": list(position.history)}
            )

    return {
        "check": "alpha-beta equivalence",
        "positions": len(sample),
        "depth": depth,
        "mismatches": len(mismatches),
        "examples": mismatches[:3],
        "passed": not mismatches,
        "mean_nodes_full_width": nodes_full / max(len(sample), 1),
        "mean_nodes_alpha_beta": nodes_ab / max(len(sample), 1),
        "mean_nodes_ordered_alpha_beta": nodes_ordered / max(len(sample), 1),
        "seconds": round(time.perf_counter() - t0, 1),
    }


# ---------------------------------------------------------------------------
# 2 & 3. play-strength checks
# ---------------------------------------------------------------------------


def _play_series(agent, opponent, games: int, seed: int = 0) -> dict:
    """`games` games, alternating who moves first. Returns W/D/L from `agent`'s view.

    Alternating is not optional. Connect Four is a first-player win under
    perfect play; a 300-game series in which the agent always moves first
    measures the opening at least as much as the agent.
    """
    wins = draws = losses = 0
    for g in range(games):
        agent_is_yellow = (g % 2 == 0)
        if hasattr(agent, "rng"):
            agent.rng = np.random.default_rng(seed + g)
        if hasattr(agent, "_mcts"):
            agent._mcts.rng = np.random.default_rng(seed + g)
        opponent.rng = np.random.default_rng(seed + 100_000 + g)

        position = Position()
        while not position.is_terminal():
            mine = (position.player == 1) == agent_is_yellow
            position.push((agent if mine else opponent).choose(position).move)
        winner = position.winner
        if winner == 0:
            draws += 1
        elif winner == (1 if agent_is_yellow else -1):
            wins += 1
        else:
            losses += 1
    return {"games": games, "wins": wins, "draws": draws, "losses": losses,
            "win_rate": (wins + 0.5 * draws) / max(games, 1)}


def check_strong_never_loses(games: int = 300, depth: int = 6, seed: int = 0) -> dict:
    """The strong search agent must never LOSE to random.

    Note the claim: never LOSES, not always wins. A draw is possible in
    principle (a full board with no four), and demanding 100% wins would be
    demanding something the game does not guarantee. The property that actually
    indicates correctness is that a 6-ply search never walks into a mate that a
    uniformly random opponent stumbled onto — random needs four in a row and the
    searcher sees every threat three plies before it completes.

    "Perfect play" in the strict sense would be a 42-ply solve; nothing in this
    product claims that, and neither should your README. This is the strongest
    agent here, and the check is that it is not broken.
    """
    agent = MinimaxAgent("heuristic_d6", depth=depth, ordering=heuristic_ordering,
                         alpha_beta=True)
    result = _play_series(agent, RandomAgent(seed=seed + 1), games, seed=seed)
    return {
        "check": "strong search never loses to random",
        "agent": f"heuristic-ordered alpha-beta, depth {depth}",
        **result,
        "passed": result["losses"] == 0,
    }


def check_mcts_beats_random(games: int = 100, simulations: int = 800,
                            seed: int = 0, threshold: float = 0.90) -> dict:
    """MCTS at a high simulation budget must beat random by a wide margin."""
    agent = MCTSAgent("mcts", iterations=simulations, c=1.414, seed=seed)
    result = _play_series(agent, RandomAgent(seed=seed + 1), games, seed=seed)
    return {
        "check": "MCTS beats random at a high iteration budget",
        "simulations": simulations,
        **result,
        "threshold": threshold,
        "passed": result["win_rate"] >= threshold,
    }


# ---------------------------------------------------------------------------
# 4. the node-count table
# ---------------------------------------------------------------------------


def check_ordering_reduces_nodes(depth: int = 6, positions: int = 25,
                                 seed: int = 0) -> dict:
    """Heuristic ordering must reduce the node count UNDER alpha-beta.

    Averaged over positions rather than measured on the empty board alone: the
    empty board is symmetric and unusually kind to any ordering, and a single
    measurement of a ratio is not a measurement of a ratio.

    The `exhaustive` and `heuristic-no-pruning` columns are equal by
    construction — ordering with nothing to cut off changes nothing — and the
    check asserts that too, because if they ever differ, the ordering callback
    has started dropping moves and the "exhaustive" agent is no longer
    exhaustive.
    """
    sample = random_positions(positions, seed=seed, max_plies=10)
    totals = {"exhaustive": 0, "heuristic_no_ab": 0, "natural+ab": 0,
              "heuristic+ab": 0, "beam3": 0}
    variants = {
        "exhaustive": (natural_ordering, False),
        "heuristic_no_ab": (heuristic_ordering, False),
        "natural+ab": (natural_ordering, True),
        "heuristic+ab": (heuristic_ordering, True),
        "beam3": (make_beam_ordering(3), True),
    }
    for position in sample:
        for label, (ordering, ab) in variants.items():
            _, _, stats = search_root(position.copy(), depth, ordering=ordering,
                                      alpha_beta=ab, node_budget=10**9)
            totals[label] += stats.nodes

    means = {k: v / max(len(sample), 1) for k, v in totals.items()}
    return {
        "check": "heuristic ordering reduces nodes under alpha-beta",
        "depth": depth,
        "positions": len(sample),
        "mean_nodes": {k: round(v, 1) for k, v in means.items()},
        "ordering_speedup_vs_natural_ab": round(
            means["natural+ab"] / max(means["heuristic+ab"], 1e-9), 2
        ),
        "alpha_beta_speedup_vs_full_width": round(
            means["exhaustive"] / max(means["natural+ab"], 1e-9), 2
        ),
        "passed": (
            means["heuristic+ab"] < means["natural+ab"] < means["exhaustive"]
            and means["exhaustive"] == means["heuristic_no_ab"]
        ),
    }


# ---------------------------------------------------------------------------


CHECKS = {
    "alpha-beta": check_alpha_beta,
    "strong-vs-random": check_strong_never_loses,
    "mcts-vs-random": check_mcts_beats_random,
    "ordering": check_ordering_reduces_nodes,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="The four substantive correctness checks.")
    ap.add_argument("--positions", type=int, default=1000,
                    help="random positions for the alpha-beta equivalence check")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--reference-games", type=int, default=300,
                    help="games for 'the strong search never loses to random'")
    ap.add_argument("--reference-depth", type=int, default=6)
    ap.add_argument("--mcts-games", type=int, default=100)
    ap.add_argument("--mcts-simulations", type=int, default=800)
    ap.add_argument("--ordering-depth", type=int, default=6)
    ap.add_argument("--ordering-positions", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=sorted(CHECKS), default=None)
    ap.add_argument("--out", default="reports/verification.json")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    selected = [args.only] if args.only else list(CHECKS)
    results = []

    for name in selected:
        print(f"[{name}] running...", flush=True)
        if name == "alpha-beta":
            r = check_alpha_beta(args.positions, args.depth, args.seed)
        elif name == "strong-vs-random":
            r = check_strong_never_loses(args.reference_games, args.reference_depth,
                                         args.seed)
        elif name == "mcts-vs-random":
            r = check_mcts_beats_random(args.mcts_games, args.mcts_simulations, args.seed)
        else:
            r = check_ordering_reduces_nodes(args.ordering_depth,
                                             args.ordering_positions, args.seed)
        results.append(r)
        print(json.dumps(r, indent=2))
        print(f"[{name}] {'PASS' if r['passed'] else 'FAIL'}\n", flush=True)

    summary = {"checks": results, "all_passed": all(r["passed"] for r in results)}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"all_passed={summary['all_passed']} -> {args.out}")
    # Non-zero exit on failure so this can be a CI gate or a make target
    # without anyone having to remember to read the output.
    if not summary["all_passed"]:
        raise SystemExit(1)
    return summary


if __name__ == "__main__":
    main()
