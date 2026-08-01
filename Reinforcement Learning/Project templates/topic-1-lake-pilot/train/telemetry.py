"""
train/telemetry.py — the plumbing every agent in this repository shares:
open an experiment, stream episode rows to the data tier, record a greedy
evaluation, and compute a mean with its standard error.

It lives in `train/` because it writes. The service tier reads the same tables
through `shared/store.py` and never imports this file.

Two decisions here are worth more than the code:

  * **Episodes are buffered, not inserted one at a time.** A training run in
    this course is 20,000 episodes. Twenty thousand single-row inserts against
    a free-tier Postgres is twenty thousand HTTPS round trips: the run takes
    longer to log than to learn, and you will be tempted to stop logging. The
    buffer flushes in chunks, so the logging cost is a rounding error and the
    temptation never arrives. This is why `episodes` can afford to be complete.

  * **A telemetry failure must not kill a training run.** Losing the last 500
    rows of a run is annoying. Losing four hours of training because the
    database blinked is unacceptable. `flush()` reports the failure on stderr
    and keeps going; the run count in the README is then the number of rows you
    can actually query, which is the number you should have been quoting anyway.
"""

from __future__ import annotations

import math
import sys
from typing import Any

from shared.config import get_settings
from shared.store import get_store


def mean_and_stderr(values) -> tuple[float, float, float]:
    """(mean, sample std, standard error). Report the third alongside the first.

    The sample standard deviation uses n-1 in the denominator, so it is
    undefined for a single observation; we return 0.0 rather than nan so a
    one-episode smoke test does not produce a row Postgres will reject. Do not
    read that 0.0 as certainty — with n = 1 you have no estimate of spread at
    all, which is exactly why /rollout's minimum is 1 but its default is 20.
    """
    xs = [float(v) for v in values]
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    std = math.sqrt(var)
    return mean, std, std / math.sqrt(n)


def new_experiment(
    algorithm: str,
    env_id: str,
    seed: int,
    hyperparameters: dict[str, Any],
) -> str:
    """Insert one `experiments` row and return its id.

    Every episode and every evaluation is keyed to this id, so the row has to
    exist before the first step is taken — not written at the end from whatever
    the variables happened to hold. A run that crashes half way still leaves a
    queryable, obviously-truncated experiment behind, and that is a far better
    artifact than nothing.

    `hyperparameters` goes into a jsonb column rather than into columns of its
    own because the next topic's agent has different knobs. Put everything that
    would change the result in here — including the epsilon schedule string —
    or the comparison table you write in week ten will not be reconstructable.
    """
    return get_store().insert_experiment(
        {
            "algorithm": algorithm,
            "env_id": env_id,
            "seed": seed,
            "hyperparameters": hyperparameters,
            "git_sha": get_settings().git_sha,
        }
    )


class EpisodeBuffer:
    """Accumulates episode rows and flushes them in chunks.

    Usage:

        with EpisodeBuffer(experiment_id) as buf:
            buf.add(episode_index=i, ret=total, length=steps, epsilon=eps)

    The context manager exists so the final partial chunk is written even when
    the loop exits early — a `KeyboardInterrupt` half way through a long run
    should still leave you with the episodes you paid for.
    """

    def __init__(self, experiment_id: str, chunk: int = 500) -> None:
        self.experiment_id = experiment_id
        self.chunk = chunk
        self.rows: list[dict[str, Any]] = []
        self.written = 0
        self.dropped = 0

    def add(self, episode_index: int, ret: float, length: int, epsilon: float | None) -> None:
        self.rows.append(
            {
                "experiment_id": self.experiment_id,
                "episode_index": int(episode_index),
                # The column is called "return" because that is the word for
                # the quantity. It is a Python keyword, which is why the
                # Pydantic model in shared/schemas.py carries an alias.
                "return": float(ret),
                "length": int(length),
                # Not optional bookkeeping: epsilon is what lets you separate
                # "the agent got worse" from "the agent explored more" months
                # later, using only the table.
                "epsilon": None if epsilon is None else float(epsilon),
            }
        )
        if len(self.rows) >= self.chunk:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        batch, self.rows = self.rows, []
        try:
            self.written += get_store().insert_episodes(batch)
        except Exception as exc:  # noqa: BLE001 — see the module docstring
            self.dropped += len(batch)
            print(f"[telemetry] dropped {len(batch)} episode rows: {exc}", file=sys.stderr)

    def __enter__(self) -> EpisodeBuffer:
        return self

    def __exit__(self, *exc_info) -> None:
        self.flush()


def record_evaluation(
    experiment_id: str,
    at_training_episode: int,
    returns,
) -> dict[str, float]:
    """Write one `evaluations` row from a greedy sweep and return its numbers.

    Greedy evaluations live in their own table rather than in `episodes` on
    purpose. Training return is contaminated by exploration — an agent with
    epsilon = 0.3 throws away three actions in ten and its training curve says
    so — while greedy return is what the deployed policy will actually do.
    Storing them in one table is how a report ends up quoting the first number
    while claiming the second.
    """
    mean, std, stderr = mean_and_stderr(returns)
    row = {
        "experiment_id": experiment_id,
        "at_training_episode": int(at_training_episode),
        "episodes": len(list(returns)),
        "mean_return": mean,
        "std_return": std,
        "stderr_return": stderr,
    }
    try:
        get_store().insert_evaluation(row)
    except Exception as exc:  # noqa: BLE001
        print(f"[telemetry] evaluation row not written: {exc}", file=sys.stderr)
    return {"mean_return": mean, "std_return": std, "stderr_return": stderr}


def warn_if_data_tier_is_local() -> None:
    """Say so, loudly, when telemetry is going nowhere.

    The failure this prevents: a student trains overnight with an empty .env,
    sees the run finish, and discovers at submission time that `episodes` is
    empty and the learning curves cannot be redrawn. The fallback store is a
    real code path, not a mock — it is just in memory, and memory ends when the
    process does.
    """
    if not get_settings().data_tier_configured:
        print(
            "[telemetry] SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are unset, so this "
            "run is being logged to the in-process fallback store and will vanish "
            "when the process exits. Fine for a smoke test; not fine for the run "
            "you intend to write about.",
            file=sys.stderr,
        )
