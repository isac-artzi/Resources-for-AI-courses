"""
train/train.py — the one command that produces a deployable learned agent.

TRAINING TIER (imports torch, transitively through train/selfplay.py). Runs on
your laptop; never on the deployed instance.

    python -m train.train                      # the committed artifact, ~90 s
    python -m train.train --iterations 20 --games 200 --simulations 200
    python -m train.train --eval-games 60      # a longer acceptance check

What it does, in order:

    1. Trains a policy-value network by self-play, writing `experiments` and
       `episodes` rows as it goes.
    2. Checks the exported NumPy path reproduces the torch path. This is an
       ACCEPTANCE GATE, not a formality — a transposed weight in the export is
       invisible until the agent quietly plays 200 Elo worse in production.
    3. Evaluates the resulting PUCT agent against the fixed reference opponent
       and writes an `evaluations` row.
    4. Exports `policies/alphazero_c4.npz` and registers it in `policies` with
       its size and checksum.

The ordering of 1 and 4 matters: the artifact is written only after the run it
came from exists in the database, so `policies.experiment_id` points at a real
row and "which run produced the thing we deployed" stays answerable.

For the tournament the product brief asks for, run `python -m train.benchmark`
after this. This script trains ONE agent; that one measures all of them.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train the AlphaZero-inspired agent by self-play and export it."
    )
    ap.add_argument("--iterations", type=int, default=6,
                    help="self-play/train generations (report budget: 20)")
    ap.add_argument("--games", type=int, default=40,
                    help="self-play games per generation (report budget: 200)")
    ap.add_argument("--simulations", type=int, default=60,
                    help="MCTS simulations per self-play move (report budget: 200)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-games", type=int, default=60,
                    help="games against the fixed reference opponent after training")
    ap.add_argument("--eval-simulations", type=int, default=100)
    ap.add_argument("--out", default="policies/alphazero_c4.npz",
                    help="artifact path; the stem is the name /act will accept")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)

    # Imported inside main() so that `--help` works, and so that a reader can
    # see at a glance that nothing at module scope reaches into the training
    # tier. Same reason train/export.py imports torch inside its function.
    from train.export import register
    from train.selfplay import SelfPlayConfig, train_selfplay

    from shared.store import get_store

    store = get_store()
    cfg = SelfPlayConfig(
        iterations=args.iterations,
        games_per_iteration=args.games,
        simulations=args.simulations,
        epochs=args.epochs,
        lr=args.lr,
        c_puct=args.c_puct,
        seed=args.seed,
    )

    if not args.quiet:
        print(
            f"self-play: {cfg.iterations} generations x {cfg.games_per_iteration} games "
            f"x {cfg.simulations} simulations/move (seed {cfg.seed})"
        )
    result = train_selfplay(cfg, store=store, log_every=0 if args.quiet else 1)
    if not args.quiet:
        print(
            f"  {result.games_played} games, {result.positions} positions, "
            f"{result.elapsed_s:.1f}s"
        )

    # --- 2. export, then verify the export ---------------------------------
    from train.export import export_policy_value_net

    row = export_policy_value_net(result.arrays, args.out)
    max_abs_error = _verify_export(result.arrays, args.out, seed=args.seed)
    if max_abs_error > 1e-4:
        raise SystemExit(
            f"EXPORT MISMATCH: the NumPy forward pass differs from torch by "
            f"{max_abs_error:.2e}. The archive is NOT deployable. The usual "
            f"cause is a transposed weight — see train/selfplay.py::_extract_arrays."
        )
    if not args.quiet:
        print(f"  export verified: |numpy - torch| <= {max_abs_error:.2e}")

    # --- 3. acceptance evaluation against the fixed reference --------------
    quality = _evaluate_against_random(
        args.out, games=args.eval_games, simulations=args.eval_simulations, seed=args.seed
    )
    if not args.quiet:
        print(
            f"  vs random over {quality['games']} games: "
            f"{quality['wins']}W {quality['draws']}D {quality['losses']}L "
            f"(win rate {quality['win_rate']:.3f})"
        )
    if result.experiment_id is not None:
        store.insert_evaluation(
            {
                "experiment_id": result.experiment_id,
                "at_training_episode": result.games_played,
                "episodes": quality["games"],
                # A win rate IS the mean return here, because the environment's
                # reward is +1/0/-1 for win/draw/loss. Saying so rather than
                # inventing a `win_rate` column keeps the standing schema
                # meaningful across topics.
                "mean_return": quality["mean_return"],
                "std_return": quality["std_return"],
                "stderr_return": quality["stderr_return"],
            }
        )

    register(row, experiment_id=result.experiment_id)

    summary = {
        "experiment_id": result.experiment_id,
        "artifact": row,
        "games_played": result.games_played,
        "positions": result.positions,
        "policy_loss": result.policy_losses[-1] if result.policy_losses else None,
        "value_loss": result.value_losses[-1] if result.value_losses else None,
        "export_max_abs_error": max_abs_error,
        "vs_random": quality,
        "elapsed_s": round(result.elapsed_s, 1),
    }
    pathlib.Path("reports").mkdir(exist_ok=True)
    pathlib.Path("reports/training.json").write_text(json.dumps(summary, indent=2))
    if not args.quiet:
        print(json.dumps(summary, indent=2))
    return summary


def _verify_export(arrays: dict[str, np.ndarray], path: str, seed: int = 0) -> float:
    """Max absolute difference between the torch and NumPy forward passes.

    Run on random inputs rather than on real boards on purpose: a real board is
    sparse and mostly zeros, which will happily hide a transposed weight in a
    layer whose input happens to be symmetric. Random dense inputs will not.
    """
    import torch

    from search.net import PolicyValueNet
    from train.selfplay import OBS_DIM, build_network

    net = PolicyValueNet.from_npz(path)
    module = build_network(seed=seed)
    # Load the exact arrays we exported, so this compares the EXPORT PATH and
    # not two independently initialised networks.
    module.load_state_dict(
        {
            "trunk.0.weight": torch.tensor(arrays["W0"]),
            "trunk.0.bias": torch.tensor(arrays["b0"]),
            "trunk.2.weight": torch.tensor(arrays["W1"]),
            "trunk.2.bias": torch.tensor(arrays["b1"]),
            "policy_head.weight": torch.tensor(arrays["Wp"]),
            "policy_head.bias": torch.tensor(arrays["bp"]),
            "value_head.weight": torch.tensor(arrays["Wv"]),
            "value_head.bias": torch.tensor(arrays["bv"]),
        }
    )
    module.eval()

    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(32):
        x = rng.normal(size=OBS_DIM).astype(np.float32)
        with torch.no_grad():
            t_logits, t_value = module(torch.from_numpy(x).unsqueeze(0))
        n_logits, n_value = net.forward(x.astype(np.float64))
        worst = max(
            worst,
            float(np.max(np.abs(t_logits.squeeze(0).numpy() - n_logits))),
            float(abs(float(t_value.item()) - n_value)),
        )
    return worst


def _evaluate_against_random(path: str, games: int, simulations: int, seed: int) -> dict:
    """Win rate of the exported agent against the fixed reference opponent.

    Uses the SERVING path — `search/net.py` loading the `.npz` — rather than the
    torch module still in memory. That is the point: this number describes the
    thing that will be deployed, not the thing that was trained.

    Alternates who moves first. Connect Four is a first-player win under perfect
    play, and a win rate measured with the agent always moving first is a win
    rate about the opening, not about the agent.
    """
    from envs.connect_four import Position
    from search.agents import MCTSAgent, RandomAgent
    from search.net import PolicyValueNet

    net = PolicyValueNet.from_npz(path)
    agent = MCTSAgent("alphazero", iterations=simulations, c=1.5,
                      prior_value=net.evaluate, seed=seed)
    opponent = RandomAgent(seed=seed + 1000)

    outcomes: list[float] = []
    for g in range(games):
        agent_is_yellow = (g % 2 == 0)
        position = Position()
        while not position.is_terminal():
            mine = (position.player == 1) == agent_is_yellow
            decision = (agent if mine else opponent).choose(position)
            position.push(decision.move)
        winner = position.winner
        if winner == 0:
            outcomes.append(0.0)
        else:
            agent_piece = 1 if agent_is_yellow else -1
            outcomes.append(1.0 if winner == agent_piece else -1.0)

    arr = np.asarray(outcomes)
    wins = int((arr > 0).sum())
    draws = int((arr == 0).sum())
    losses = int((arr < 0).sum())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        # Draws count half. The definition lives in one place — here and in
        # db/migrations/002_topic5.sql's view — because "win rate" is ambiguous
        # in a game with draws and two charts using two definitions is a
        # reporting error nobody notices.
        "win_rate": (wins + 0.5 * draws) / games if games else 0.0,
        "mean_return": float(arr.mean()),
        "std_return": std,
        "stderr_return": std / np.sqrt(max(arr.size, 1)),
    }


if __name__ == "__main__":
    main()
