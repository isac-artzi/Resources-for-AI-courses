"""Apply the migration (manually) and seed one default dataset into Supabase.

Usage (from the repo root, with a .env present or the env vars exported):

    python -m db.seed

This uses the SERVICE-ROLE key so it can write. Never commit that key.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from api import db
from shared.data import IMG_SIZE, N_CLASSES

load_dotenv()

MIGRATION = Path(__file__).parent / "migrations" / "001_init.sql"


def main() -> None:
    # NOTE: supabase-py cannot run arbitrary DDL over the REST API. Apply
    # 001_init.sql once in the Supabase SQL Editor (see the main TUTORIAL). This
    # script only seeds data, which the REST API supports.
    if not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY")):
        raise SystemExit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY first.")

    print(f"Migration file: {MIGRATION} (apply it in the SQL Editor if you have not).")

    row = db.insert_dataset(
        name="default-shapes",
        n_rows=800,
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        noise=0.1,
        seed=42,
    )
    print(
        f"Seeded dataset id={row['id']} name={row['name']} "
        f"n_rows={row['n_rows']} img_size={IMG_SIZE} n_classes={N_CLASSES}"
    )


if __name__ == "__main__":
    main()
