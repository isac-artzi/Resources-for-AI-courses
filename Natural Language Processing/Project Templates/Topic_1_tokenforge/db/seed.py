"""Apply the migration and insert a couple of demo rows.

    python db/seed.py

Supabase's client library cannot execute arbitrary DDL, so this script does NOT
create the table for you — it checks whether the table exists and tells you what
to do if it doesn't. Paste db/migrations/001_init.sql into the SQL Editor once;
after that this script is just a way to put a row in the History tab so you can
see it render before your NLP code works.
"""
from __future__ import annotations

import hashlib
import sys

from api import db


def main() -> int:
    if not db.configured():
        print("SUPABASE_URL / SUPABASE_SERVICE_KEY are not set. Copy .env.example "
              "to .env and fill them in.")
        return 1

    if not db.ping():
        print(
            "Could not read the 'runs' table.\n"
            "Open your Supabase project -> SQL Editor -> New query, paste the\n"
            "contents of db/migrations/001_init.sql, and click Run. Then re-run me."
        )
        return 1

    for text, before, after in [
        ("the quick brown fox jumps over the lazy dog", 9, 5),
        ("Tokenization of unhappiness in Zurich costs $3.50", 7, 6),
    ]:
        row = db.insert_run(
            kind="preprocess",
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            config={"seed": True, "lowercase": True, "remove_stopwords": True},
            model_version="seed",
            token_count_before=before,
            token_count_after=after,
        )
        print("inserted run", row["id"] if row else "(failed)")

    print("Done. Open the Run History tab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
