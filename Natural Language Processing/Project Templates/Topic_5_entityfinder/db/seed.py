"""Apply the migration and insert enough demo rows to see the UI work.

    python db/seed.py

Supabase's client library cannot execute arbitrary DDL, so this script does NOT
create the tables for you — it checks whether they exist and tells you what to
do if they don't. Paste db/migrations/001_init.sql into the SQL Editor once;
after that this script is a way to put rows in front of you before your NLP code
works.

It seeds one training run per model and one extraction with three entities, one
of them deliberately low-confidence and mislabelled. That last row is the point:
without it the Review Queue tab is an empty page and you cannot tell a working
queue from a broken one.
"""
from __future__ import annotations

import hashlib
import sys

from api import db

DEMO_TEXT = "Ada Lovelace worked with Charles Babbage in London."


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

    # Two training runs. The numbers are placeholders so the comparison tab has
    # something to draw — replace them with your own the moment you have any,
    # and never quote these in a report.
    for model_type, config in [
        ("crf", {"seed": True, "c1": 0.1, "c2": 0.1, "features": ["suffix3", "pos"]}),
        ("transformer", {"seed": True, "base_checkpoint": "bert-base-cased", "epochs": 3}),
    ]:
        row = db.insert_run(
            model_type=model_type,
            dataset="seed",
            config=config,
            model_version="seed",
            notes="Placeholder row from db/seed.py — replace with a real run.",
        )
        print("inserted run", row["id"] if row else "(failed)")

    extraction = db.insert_extraction(
        text_sha256=hashlib.sha256(DEMO_TEXT.encode()).hexdigest(),
        model="transformer",
        model_version="seed",
        entity_count=3,
        latency_ms=120,
    )
    if not extraction:
        print("Could not insert an extraction. Check the extractions table exists.")
        return 1
    print("inserted extraction", extraction["id"])

    # Offsets are real offsets into DEMO_TEXT. Check them with
    # DEMO_TEXT[start:end] before you trust anything this script tells you —
    # that habit is the whole discipline of this assignment in one line.
    rows = db.insert_entities(
        extraction["id"],
        [
            {
                "text": "Ada Lovelace",
                "start_char": 0,
                "end_char": 12,
                "entity_type": "PER",
                "confidence": 0.98,
            },
            {
                "text": "Charles Babbage",
                "start_char": 25,
                "end_char": 40,
                "entity_type": "PER",
                "confidence": 0.95,
            },
            {
                # Wrong type, low score: this is the row the Review Queue exists for.
                "text": "London",
                "start_char": 44,
                "end_char": 50,
                "entity_type": "ORG",
                "confidence": 0.44,
            },
        ],
    )
    print(f"inserted {len(rows)} entities")

    print(
        "Done. Open the CRF vs. Transformer tab for the runs, and the Review "
        "Queue tab for the low-confidence prediction queued for you."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
