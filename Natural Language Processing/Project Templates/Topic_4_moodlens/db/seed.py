"""Apply the migration, then put one demo training run and two demo predictions
in the database so the UI has something to draw.

    python db/seed.py

Supabase's client library cannot execute arbitrary DDL, so this script does NOT
create the tables for you — it checks whether they exist and tells you what to do
if they don't. Paste db/migrations/001_init.sql into the SQL Editor once; after
that this script is just a way to see the Model Performance and Bias Audit tabs
render before your NLP code works.

READ THIS BEFORE YOU SCREENSHOT ANYTHING
----------------------------------------
Every number below is INVENTED. It exists to show the shape of the metrics
payload — which keys go where, how a confusion matrix is oriented, what a
calibration bin looks like — and for no other reason. The seed run is tagged
"seed-not-a-real-model" and the UI puts a red banner on any run whose version
starts with "seed". Delete the row once you have trained something:

    delete from runs where model_version like 'seed%';

A report that quotes these numbers is reporting a fixture.
"""
from __future__ import annotations

import hashlib
import sys

from api import db

SEED_VERSION = "seed-not-a-real-model"

# --- The shape of runs.metrics ---------------------------------------------
# Three keys: documents (one entry per model — you need BOTH), aspects (one per
# aspect), slices (one per bucket per slice_name). The UI reads exactly this.
SEED_METRICS = {
    "documents": [
        {
            "model_name": "transformer",
            "n": 200,
            "accuracy": 0.5,
            "macro_precision": 0.5,
            "macro_recall": 0.5,
            "macro_f1": 0.5,
            "per_class": {
                "negative": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 100},
                "positive": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 100},
            },
            "labels": ["negative", "positive"],
            # confusion_matrix[gold][predicted]. Row 0 is the reviews that really
            # are negative: 50 of them were called negative, 50 positive.
            "confusion_matrix": [[50, 50], [50, 50]],
            "roc_auc": 0.5,
            "roc_points": [
                {"threshold": 0.0, "fpr": 1.0, "tpr": 1.0},
                {"threshold": 0.5, "fpr": 0.5, "tpr": 0.5},
                {"threshold": 1.0, "fpr": 0.0, "tpr": 0.0},
            ],
            # A coin flip that always says 0.5. Mean predicted 0.5, observed 0.5.
            "calibration_bins": [
                {
                    "bin_lower": 0.4,
                    "bin_upper": 0.6,
                    "mean_predicted": 0.5,
                    "observed_positive_rate": 0.5,
                    "count": 200,
                }
            ],
        },
        {
            "model_name": "tfidf-baseline",
            "n": 200,
            "accuracy": 0.5,
            "macro_precision": 0.5,
            "macro_recall": 0.5,
            "macro_f1": 0.5,
            "per_class": {
                "negative": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 100},
                "positive": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 100},
            },
            "labels": ["negative", "positive"],
            "confusion_matrix": [[50, 50], [50, 50]],
            "roc_auc": 0.5,
            "roc_points": [
                {"threshold": 0.0, "fpr": 1.0, "tpr": 1.0},
                {"threshold": 0.5, "fpr": 0.5, "tpr": 0.5},
                {"threshold": 1.0, "fpr": 0.0, "tpr": 0.0},
            ],
            "calibration_bins": [],
        },
    ],
    "aspects": [
        {
            "aspect": aspect,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "support": 0,
            "n_evaluated": 0,
        }
        for aspect in ("acting", "plot", "production")
    ],
    "slices": [
        # review_length is measured, so observed=true.
        {
            "slice_name": "review_length",
            "bucket": bucket,
            "n": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "observed": True,
        }
        for bucket in ("short", "medium", "long")
    ]
    + [
        # genre is inferred unless your corpus ships it: observed=false, and the
        # Bias Audit tab will say so on the row.
        {
            "slice_name": "genre",
            "bucket": bucket,
            "n": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "observed": False,
        }
        for bucket in ("drama", "comedy")
    ],
}


def main() -> int:
    if not db.configured():
        print("SUPABASE_URL / SUPABASE_SERVICE_KEY are not set. Copy .env.example "
              "to .env and fill them in.")
        return 1

    if not db.ping():
        print(
            "Could not read the 'predictions' table.\n"
            "Open your Supabase project -> SQL Editor -> New query, paste the\n"
            "contents of db/migrations/001_init.sql, and click Run. Then re-run me."
        )
        return 1

    run = db.insert_run(
        model_version=SEED_VERSION,
        base_model="none",
        dataset="none — fixture data",
        config={"seed_row": True},
        metrics=SEED_METRICS,
        n_train=0,
        n_eval=200,
        notes=(
            "SEED ROW. Every number here is invented to show the metrics layout. "
            "Delete this row once you have a real training run."
        ),
    )
    print("inserted run", run["id"] if run else "(failed)")

    for text, label, prob in [
        ("A demo review that the seed script pretends was positive.", "positive", 0.5),
        ("A demo review that the seed script pretends was negative.", "negative", 0.5),
    ]:
        row = db.insert_prediction(
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            char_count=len(text),
            label=label,
            probability_positive=prob,
            confidence=0.5,
            calibrated=False,
            aspects=[
                {"aspect": "acting", "label": "not_mentioned", "score": 0.0, "evidence": []},
                {"aspect": "plot", "label": "not_mentioned", "score": 0.0, "evidence": []},
                {
                    "aspect": "production",
                    "label": "not_mentioned",
                    "score": 0.0,
                    "evidence": [],
                },
            ],
            model_name="seed",
            model_version=SEED_VERSION,
        )
        print("inserted prediction", row["id"] if row else "(failed)")

    print("Done. Open the Model Performance and Bias Audit tabs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
