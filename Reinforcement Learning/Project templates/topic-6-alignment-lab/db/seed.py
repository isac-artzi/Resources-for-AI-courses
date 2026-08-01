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

DEFAULT = {
    "algorithm": "seed-placeholder",
    "env_id": "preference-corpus",
    "seed": 0,
    "hyperparameters": {"note": "replace with your first real training run"},
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
