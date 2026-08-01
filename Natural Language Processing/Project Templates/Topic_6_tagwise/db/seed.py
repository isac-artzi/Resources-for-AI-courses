"""Apply the migration and insert a few demo rows.

    python db/seed.py

Supabase's client library cannot execute arbitrary DDL, so this script does NOT
create the tables for you — it checks whether they exist and tells you what to do
if they don't. Paste db/migrations/001_init.sql into the SQL Editor once; after
that this script is a way to put rows in the History and comparison tabs so you
can see them render before your NLP code works.

The numbers below are invented and labelled 'seed'. Delete these rows once
your own build script writes real ones, and never report a seeded number as a
result — a small confusion matrix with round numbers in it is very easy to leave
in a screenshot by accident.
"""
from __future__ import annotations

import hashlib
import sys

from api import db

# A 4x4 slice of a confusion matrix over four tags, so the comparison tab has
# something with the right shape to draw. Rows are gold, columns are predicted.
DEMO_LABELS = ["ADJ", "ADV", "NOUN", "VERB"]
DEMO_BASELINE_MATRIX = [
    [30, 6, 3, 1],
    [5, 28, 1, 1],
    [2, 1, 90, 12],
    [1, 2, 14, 60],
]
DEMO_TRANSFORMER_MATRIX = [
    [36, 3, 1, 0],
    [2, 32, 1, 0],
    [1, 1, 99, 4],
    [0, 1, 5, 71],
]


def main() -> int:
    if not db.configured():
        print("SUPABASE_URL / SUPABASE_SERVICE_KEY are not set. Copy .env.example "
              "to .env and fill them in.")
        return 1

    if not db.ping():
        print(
            "Could not read the 'runs' and 'taggings' tables.\n"
            "Open your Supabase project -> SQL Editor -> New query, paste the\n"
            "contents of db/migrations/001_init.sql, and click Run. Then re-run me.\n"
            "If only one of the two tables exists, run the whole file again."
        )
        return 1

    row = db.insert_run(
        model="baseline",
        tagset="UPOS",
        hyperparameters={
            "seed_row": True,
            "treebank": "replace-with-the-treebank-you-used",
            "lowercase_keys": False,
            "tie_break": "corpus frequency",
            "train_sentences": 0,
        },
        model_version="seed",
        accuracy=0.90,
        macro_f1=0.71,
        metrics={"confusion": {"labels": DEMO_LABELS, "matrix": DEMO_BASELINE_MATRIX}},
        notes="Seeded demo row. Delete once your build script writes a real one.",
    )
    print("inserted baseline run", row["id"] if row else "(failed)")

    row = db.insert_run(
        model="transformer",
        tagset="UPOS",
        hyperparameters={
            "seed_row": True,
            "base_model": "replace-with-your-checkpoint",
            "learning_rate": 5e-5,
            "epochs": 3,
            "batch_size": 16,
            "seed": 42,
        },
        model_version="seed",
        accuracy=0.95,
        macro_f1=0.86,
        metrics={"confusion": {"labels": DEMO_LABELS, "matrix": DEMO_TRANSFORMER_MATRIX}},
        notes="Seeded demo row. Delete once your build script writes a real one.",
    )
    print("inserted transformer run", row["id"] if row else "(failed)")

    for sentence, tags, model, unknown in [
        (
            "They book the flight early .",
            ["PRON", "VERB", "DET", "NOUN", "ADV", "PUNCT"],
            "transformer",
            0,
        ),
        (
            "Read the book on the shelf .",
            ["VERB", "DET", "NOUN", "ADP", "DET", "NOUN", "PUNCT"],
            "baseline",
            1,
        ),
    ]:
        row = db.insert_tagging(
            sentence_sha256=hashlib.sha256(sentence.encode()).hexdigest(),
            token_count=len(tags),
            tag_sequence=tags,
            model=model,
            model_version="seed",
            unknown_count=unknown,
        )
        print("inserted tagging", row["id"] if row else "(failed)")

    print("Done. Open the History and 'Baseline vs. Transformer' tabs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
