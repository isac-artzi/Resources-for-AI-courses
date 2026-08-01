"""
db/seed.py — write one default experiment configuration so a fresh clone has
something for the run-history view to render.

Run it once after applying the migration:

    python -m db.seed
"""

from __future__ import annotations

import json

from shared.config import get_settings
from shared.store import get_store

# The Topic 1 default configuration, spelled out rather than left as a
# placeholder. It is the row the run-history view renders before you have
# trained anything, and it doubles as documentation of what `python -m
# train.train` with no flags will do — if the two ever disagree, one of them is
# wrong and this file is the cheaper one to check.
DEFAULT = {
    "algorithm": "q-learning",
    "env_id": "FrozenLake-v1-8x8-slippery",
    "seed": 0,
    "hyperparameters": {
        "alpha": 0.1,
        "gamma": 0.99,
        "eps_schedule": "linear:1.0:0.05:0.6",
        "episodes": 20000,
        "note": "seeded configuration — no episodes logged against it yet",
    },
    "git_sha": get_settings().git_sha,
}


def main() -> None:
    store = get_store()
    experiment_id = store.insert_experiment(DEFAULT)
    print(json.dumps({"experiment_id": experiment_id, **DEFAULT}, indent=2))
    if not get_settings().data_tier_configured:
        print(
            "\nNOTE: SUPABASE_URL is unset, so this was written to the in-process "
            "fallback store and will vanish when this process exits. Fill in .env "
            "and re-run to write to Postgres."
        )


if __name__ == "__main__":
    main()
