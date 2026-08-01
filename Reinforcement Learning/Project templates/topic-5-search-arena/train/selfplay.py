"""
train/selfplay.py — the AlphaZero-inspired agent's training loop.

TRAINING TIER. This is the only module in the repository that imports torch, and
it is never imported by `api/`, `ui/` or `search/`. `tests/test_no_torch.py`
enforces that; if it starts failing, something on the serving path grew an
import of this file.

What AlphaZero actually is, stripped of the compute
---------------------------------------------------
Two ideas, and neither of them requires 5,000 TPUs to demonstrate:

  1. **Replace the rollout with a learned evaluation.** Plain MCTS estimates a
     leaf's value by playing the game out at random. That estimate is unbiased
     and enormously noisy — a uniform playout from a won position loses about
     half the time. A network that has seen a few hundred thousand positions
     gives a lower-variance estimate in one forward pass instead of forty
     random moves.

  2. **Train that network on the search's own output.** The search, run for a
     few hundred simulations, plays better than the network alone; its visit
     distribution is therefore a better policy than the network's prior. Train
     the prior towards the visit distribution, and the value head towards the
     eventual game result, and you have a loop that improves without a single
     human-labelled position. That is the "self-play" part, and it is the part
     that surprises people.

The loss is the AlphaZero loss, which is worth writing out because it is short:

    L = (z - v)^2  -  pi^T log p  +  c ||theta||^2
        ^ value      ^ policy       ^ weight decay (here: torch's `weight_decay`)

where `z` is the game result from the position's mover's point of view, `v` is
the value head, `pi` is the MCTS visit distribution and `p` is the policy head.

What is deliberately NOT here, and why
--------------------------------------
No residual tower, no Dirichlet noise at the root, no temperature schedule past
move 30, no replay buffer across many generations, no evaluator gate between
generations. Every one of those matters at scale and none of them is what the
learning objective is about. Adding them to a run of 240 self-play games would
be cargo cult: they are solutions to problems this budget never reaches. If you
want to add one, add it AND measure it in `train/benchmark.py`, which is the
standard the rest of this product is held to.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from envs.connect_four import COLS, Position
from search.mcts import MCTS
from shared.preprocess import canonical_planes

OBS_DIM = 2 * 42          # two binary planes; see shared/preprocess.canonical_planes
HIDDEN = 128


@dataclass
class SelfPlayConfig:
    """The sandbox defaults, and the numbers you should actually submit with.

    These defaults make `python -m train.train` finish in about 90 seconds on a
    laptop CPU, which is the right trade for a template: a student's first run
    should complete before they lose interest, and every one of these is a flag.
    They are also the EXACT defaults that produced the committed
    `policies/alphazero_c4.npz`, so a bare `python -m train.train` reproduces
    the deployed artifact rather than something adjacent to it.

    The REAL budget, for the run whose numbers go in your report:

        --iterations 20 --games 200 --simulations 200 --epochs 8

    which is about two hours on a CPU. That is not AlphaZero; AlphaZero was
    44 million games. What this budget buys is an agent that beats random
    ~98% of the time and is competitive with the hand-written heuristic at a
    comparable node count — which is the claim the README makes and the
    benchmark checks. Do not claim more than your budget bought.
    """

    iterations: int = 6
    games_per_iteration: int = 40
    simulations: int = 60
    c_puct: float = 1.5
    epochs: int = 4
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 0
    # Moves before play becomes greedy. Before this, the move is SAMPLED from
    # the visit distribution. Without sampling, every self-play game from the
    # empty board is nearly identical and the network sees one line of play a
    # thousand times — the single most common reason a small self-play run
    # learns nothing.
    temperature_moves: int = 12


@dataclass
class SelfPlayResult:
    experiment_id: str | None = None
    losses: list[float] = field(default_factory=list)
    policy_losses: list[float] = field(default_factory=list)
    value_losses: list[float] = field(default_factory=list)
    games_played: int = 0
    positions: int = 0
    elapsed_s: float = 0.0
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    episode_rows: list[dict[str, Any]] = field(default_factory=list)


def build_network(seed: int = 0):
    """Trunk + two heads, in torch. Imported inside the function on purpose.

    The import is here rather than at module scope so that this file can be
    READ — and its constants imported — from a serving environment where torch
    is not installed. `train/export.py` uses the same trick and explains it.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    class PolicyValueModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(OBS_DIM, HIDDEN), nn.ReLU(),
                nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            )
            self.policy_head = nn.Linear(HIDDEN, COLS)
            # The value head ends in tanh so its output is in [-1, 1], the same
            # range as a game result. A linear head trained on targets in
            # {-1, 0, 1} converges to the same place, but nothing STOPS it
            # predicting 4.2 on an unseen position, and PUCT will believe it.
            self.value_head = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            h = self.trunk(x)
            return self.policy_head(h), torch.tanh(self.value_head(h))

    return PolicyValueModule()


def _numpy_prior_value(module):
    """Wrap a torch module as the `prior_value` callback MCTS expects.

    Used DURING training, where torch is present anyway. At serving time the
    same role is played by `search/net.PolicyValueNet.evaluate`, reading the
    exported archive with NumPy. Having two implementations of one function is
    a risk, and `tests/test_net_equivalence.py` is what manages it: it asserts
    the two agree to 1e-5 on random inputs. Without that test, a transposed
    weight in the export is a silent 200-Elo regression.
    """
    import torch

    def prior_value(position: Position) -> tuple[np.ndarray, float]:
        x = canonical_planes(position.board, position.player)
        with torch.no_grad():
            logits, value = module(torch.from_numpy(x).float().unsqueeze(0))
        logits = logits.squeeze(0).numpy().astype(np.float64)
        # Mask to legal columns and renormalise — identical to what
        # search/net.py does at serving time. If you change one, change both.
        z = logits - logits.max()
        p = np.exp(z)
        mask = np.zeros(COLS)
        for col in position.legal_moves():
            mask[col] = 1.0
        p = p * mask
        total = p.sum()
        p = p / total if total > 0 else mask / max(mask.sum(), 1.0)
        return p, float(value.item())

    return prior_value


def play_one_selfplay_game(
    module, cfg: SelfPlayConfig, rng: np.random.Generator
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], int]:
    """One self-play game. Returns (observations, visit targets, value targets, moves).

    The value target is assigned AFTER the game ends, by walking back through
    the positions and giving each one the final result from the point of view of
    whoever was to move there. That alternation is the same one
    `search/mcts.py::_backpropagate` performs, and getting it wrong here trains
    the value head to predict the negative of what it should — producing an
    agent that reliably steers into losing positions and looks, from the
    outside, like an agent that simply has not learned yet.
    """
    position = Position()
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    movers: list[int] = []

    mcts = MCTS(
        iterations=cfg.simulations,
        c=cfg.c_puct,
        prior_value=_numpy_prior_value(module),
        seed=int(rng.integers(2**31)),
    )

    moves = 0
    while not position.is_terminal():
        result = mcts.search(position)
        visits = np.zeros(COLS, dtype=np.float32)
        for col, n in result.root_visits.items():
            visits[col] = n
        total = visits.sum()
        if total <= 0:
            # Can only happen at a simulation count so low that nothing was
            # expanded. Fall back to uniform-over-legal rather than dividing by
            # zero and training on NaNs.
            for col in position.legal_moves():
                visits[col] = 1.0
            total = visits.sum()
        visits /= total

        observations.append(canonical_planes(position.board, position.player))
        targets.append(visits.copy())
        movers.append(position.player)

        if moves < cfg.temperature_moves:
            # Sample from the visit distribution, not argmax. See the config.
            col = int(rng.choice(COLS, p=visits))
        else:
            col = int(np.argmax(visits))
        position.push(col)
        moves += 1

    # Final result, from YELLOW's point of view, then flipped per position.
    winner = position.winner
    outcome_for_yellow = 0.0 if winner == 0 else float(winner)
    values = [outcome_for_yellow * mover for mover in movers]
    return observations, targets, values, moves


def train_selfplay(
    cfg: SelfPlayConfig,
    store=None,
    log_every: int = 1,
) -> SelfPlayResult:
    """The full loop: play, train, repeat. Returns exportable NumPy arrays.

    Telemetry is written to `episodes` as it goes — one row per self-play game,
    with the game's length and the winner as its "return". That is a slightly
    odd fit for a table designed around episodic return, and it is the right
    call anyway: the alternative is a fourth table nobody queries, and "return"
    for a self-play game genuinely is its outcome. What matters is that the
    meaning is written down, which is what the `algorithm` column is for.
    """
    import torch
    import torch.nn.functional as F

    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    module = build_network(seed=cfg.seed)
    opt = torch.optim.Adam(module.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    result = SelfPlayResult()
    experiment_id = None
    if store is not None:
        experiment_id = store.insert_experiment(
            {
                "algorithm": "alphazero_selfplay",
                "env_id": "ConnectFour-6x7-v1",
                "seed": cfg.seed,
                "hyperparameters": {
                    "iterations": cfg.iterations,
                    "games_per_iteration": cfg.games_per_iteration,
                    "simulations": cfg.simulations,
                    "c_puct": cfg.c_puct,
                    "epochs": cfg.epochs,
                    "lr": cfg.lr,
                    "temperature_moves": cfg.temperature_moves,
                },
            }
        )
        result.experiment_id = experiment_id

    game_index = 0
    all_obs: list[np.ndarray] = []
    all_pi: list[np.ndarray] = []
    all_z: list[float] = []

    for iteration in range(cfg.iterations):
        # --- self-play ------------------------------------------------------
        for _ in range(cfg.games_per_iteration):
            obs, pi, z, moves = play_one_selfplay_game(module, cfg, rng)
            all_obs.extend(obs)
            all_pi.extend(pi)
            all_z.extend(z)
            result.episode_rows.append(
                {
                    "experiment_id": experiment_id,
                    "episode_index": game_index,
                    "return": float(z[0]) if z else 0.0,
                    "length": moves,
                    # NULL, not 0.0. This agent has no epsilon-greedy schedule;
                    # its exploration is the PUCT term plus the temperature
                    # sampling, and writing 0.0 here would read as "greedy",
                    # which is a different and false claim.
                    "epsilon": None,
                }
            )
            game_index += 1

        # A sliding window over the two most recent generations' games. A full
        # replay buffer is the textbook answer; with 60 games total there is
        # nothing to slide, and keeping everything would train the newest
        # network mostly on data from the weakest one.
        window = cfg.games_per_iteration * 2 * 25   # ~25 positions per game
        obs_arr = np.asarray(all_obs[-window:], dtype=np.float32)
        pi_arr = np.asarray(all_pi[-window:], dtype=np.float32)
        z_arr = np.asarray(all_z[-window:], dtype=np.float32).reshape(-1, 1)

        # --- fit ------------------------------------------------------------
        x = torch.from_numpy(obs_arr)
        target_pi = torch.from_numpy(pi_arr)
        target_z = torch.from_numpy(z_arr)
        n = x.shape[0]

        for _epoch in range(cfg.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, cfg.batch_size):
                idx = perm[i : i + cfg.batch_size]
                logits, value = module(x[idx])
                # Cross-entropy against a DISTRIBUTION, not against a label.
                # `F.cross_entropy` with an integer target would throw away
                # everything the search learned about the second-best move,
                # which is most of what the search learned.
                logp = F.log_softmax(logits, dim=1)
                policy_loss = -(target_pi[idx] * logp).sum(dim=1).mean()
                value_loss = F.mse_loss(value, target_z[idx])
                loss = policy_loss + value_loss
                opt.zero_grad()
                loss.backward()
                opt.step()

        with torch.no_grad():
            logits, value = module(x)
            logp = torch.nn.functional.log_softmax(logits, dim=1)
            pl = float(-(target_pi * logp).sum(dim=1).mean())
            vl = float(torch.nn.functional.mse_loss(value, target_z))
        result.losses.append(pl + vl)
        result.policy_losses.append(pl)
        result.value_losses.append(vl)
        if log_every and iteration % log_every == 0:
            print(
                f"  iteration {iteration + 1}/{cfg.iterations}: "
                f"{n} positions, policy loss {pl:.4f}, value loss {vl:.4f}"
            )

    result.games_played = game_index
    result.positions = len(all_obs)
    result.elapsed_s = time.perf_counter() - t0
    result.arrays = _extract_arrays(module)

    if store is not None and result.episode_rows:
        store.insert_episodes(result.episode_rows)

    return result


def _extract_arrays(module) -> dict[str, np.ndarray]:
    """torch module -> the exact archive layout `search/net.py` reads.

    The key names are a contract between two files and they are written out
    here explicitly rather than derived from `named_parameters()`. A derived
    naming scheme survives exactly until someone inserts a layer, at which point
    the serving side loads a differently-shaped archive and either crashes or —
    much worse — does not.

    Note the weights are exported WITHOUT transposing. `nn.Linear` stores its
    weight as (out, in), and `search/net.py` computes `W @ x + b`, so the shapes
    line up as they are. If the equivalence test fails, this is the first line
    to check: a transposed matrix accounts for nearly every failure of it.
    """
    sd = module.state_dict()
    return {
        "W0": sd["trunk.0.weight"].cpu().numpy(),
        "b0": sd["trunk.0.bias"].cpu().numpy(),
        "W1": sd["trunk.2.weight"].cpu().numpy(),
        "b1": sd["trunk.2.bias"].cpu().numpy(),
        "Wp": sd["policy_head.weight"].cpu().numpy(),
        "bp": sd["policy_head.bias"].cpu().numpy(),
        "Wv": sd["value_head.weight"].cpu().numpy(),
        "bv": sd["value_head.bias"].cpu().numpy(),
    }
