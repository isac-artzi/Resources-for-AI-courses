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

# Named for this topic rather than left as a placeholder, so that a fresh clone
# has a row whose `env_id` matches the one `train/train.py` and
# `train/benchmark.py` will write. A seed row with `env_id = "none"` sorts
# oddly next to real runs and is the first thing anyone deletes; this one is at
# least honest about which product it belongs to.
DEFAULT = {
    "algorithm": "seed-placeholder",
    "env_id": "ConnectFour-6x7-v1",
    "seed": 0,
    "hyperparameters": {
        "note": "replace with your first real run: python -m train.train",
        "agents": "see GET /agents",
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
