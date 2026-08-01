"""Apply the migration and insert a couple of demo rows.

    python db/seed.py

Supabase's client library cannot execute arbitrary DDL, so this script does NOT
create the tables for you — it checks whether they exist and tells you what to do
if they don't. Paste db/migrations/001_init.sql into the SQL Editor once; after
that this script is just a way to put rows in both tabs so you can see them
render before your model code works.

It seeds BOTH tables, because both tabs need a data source: one fake training run
per model kind for the Baseline vs. Transformer tab, and a handful of fake served
predictions for the Recent Predictions tab. The numbers below are placeholders,
deliberately marked model_version='seed' so you can delete them with one query:

    delete from runs        where model_version = 'seed';
    delete from predictions where model_version = 'seed';

Do delete them before you submit. A grader who sees 'seed' in your metrics table
has no way to tell which numbers you actually measured.
"""
from __future__ import annotations

import hashlib
import sys

from api import db

# Two placeholder training runs. The metric values are deliberately zero — these
# rows exist to show you the SHAPE, not to look like results. What to copy: both
# models evaluated on the same held-out split, the same positive label on both,
# all four metrics recorded, and hyperparameters complete enough to re-run the
# training from this row alone.
SEED_RUNS = [
    {
        "model_kind": "baseline",
        "dataset_name": "seed-placeholder",
        "hyperparameters": {
            "vectorizer": "tfidf",
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_features": 50000,
            "classifier": "logistic_regression",
            "C": 1.0,
            "class_weight": "balanced",
            "seed": 42,
            "calibration": "none",
        },
        "metrics": {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "positive_label": "REPLACE_ME",
            "n_examples": 0,
            "support_positive": 0,
        },
        "model_version": "seed",
        "n_train": 0,
        "n_eval": 0,
    },
    {
        "model_kind": "transformer",
        "dataset_name": "seed-placeholder",
        "hyperparameters": {
            "checkpoint": "distilbert-base-uncased",
            "learning_rate": 2e-5,
            "epochs": 3,
            "per_device_train_batch_size": 16,
            "max_length": 256,
            "weight_decay": 0.01,
            "seed": 42,
            "calibration": "none",
        },
        "metrics": {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "positive_label": "REPLACE_ME",
            "n_examples": 0,
            "support_positive": 0,
        },
        "model_version": "seed",
        "n_train": 0,
        "n_eval": 0,
    },
]

# Placeholder served predictions. The text is hashed on the way in, exactly as
# the API does it — this script never writes a message body either.
SEED_PREDICTIONS = [
    ("my order never arrived and nobody has replied in four days", "positive", 0.93),
    ("thanks, that worked", "negative", 0.88),
    ("any update on this?", "positive", 0.61),
]


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

    for run in SEED_RUNS:
        row = db.insert_run(**run)
        print("inserted run", row["id"] if row else "(failed)")

    for text, label, prob in SEED_PREDICTIONS:
        row = db.insert_prediction(
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            predicted_label=label,
            probability=prob,
            model_kind="baseline",
            model_version="seed",
            latency_ms=12.0,
        )
        print("inserted prediction", row["id"] if row else "(failed)")

    print(
        "Done. Open the Baseline vs. Transformer and Recent Predictions tabs, "
        "then delete the seed rows once your own runs are landing."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
