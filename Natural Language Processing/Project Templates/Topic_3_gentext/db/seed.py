"""Apply the migration and insert a few demo rows.

    python db/seed.py

Supabase's client library cannot execute arbitrary DDL, so this script does NOT
create the tables for you — it checks whether they exist and tells you what to do
if they don't. Paste db/migrations/001_init.sql into the SQL Editor once; after
that this script is a way to put rows in the History tab so you can see it render
before your generation code works.

It seeds BOTH tables, because both are part of the assignment: one training run
and three generations from it, one of them already rated by two raters so the
disagreement column has something in it.
"""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone

from api import db

SEED_PROMPTS = [
    (
        "Write a short product review of a pair of running shoes:",
        "greedy",
        {"strategy": "greedy", "max_new_tokens": 60, "temperature": 1.0,
         "top_k": 0, "top_p": 1.0, "num_beams": 1, "repetition_penalty": 1.0,
         "seed": None},
        "They are comfortable and they are comfortable and they are comfortable "
        "for running. (Seed row: this is what greedy decoding does when the "
        "model is unsure — a loop, not a training failure.)",
        0.31,
    ),
    (
        "Write a short product review of a pair of running shoes:",
        "top_p",
        {"strategy": "top_p", "max_new_tokens": 60, "temperature": 0.9,
         "top_k": 0, "top_p": 0.92, "num_beams": 1, "repetition_penalty": 1.0,
         "seed": 7},
        "Light enough for a morning loop and the laces held after a month of "
        "wet mornings, though the toe box runs narrow. (Seed row: same prompt, "
        "nucleus sampling.)",
        0.94,
    ),
    (
        "Write a short product review of a pair of running shoes:",
        "beam",
        {"strategy": "beam", "max_new_tokens": 60, "temperature": 1.0,
         "top_k": 0, "top_p": 1.0, "num_beams": 5, "repetition_penalty": 1.0,
         "seed": None},
        "A good shoe for the price. I would recommend it to anyone looking for "
        "a good shoe for the price. (Seed row: beam search, safe and dull.)",
        0.52,
    ),
]


def main() -> int:
    if not db.configured():
        print("SUPABASE_URL / SUPABASE_SERVICE_KEY are not set. Copy .env.example "
              "to .env and fill them in.")
        return 1

    if not db.ping():
        print(
            "Could not read the 'generations' table.\n"
            "Open your Supabase project -> SQL Editor -> New query, paste the\n"
            "contents of db/migrations/001_init.sql, and click Run. Then re-run me."
        )
        return 1

    run = db.insert_training_run(
        base_model="gpt2",
        model_version="seed-v0",
        hyperparameters={
            "method": "frozen",
            "epochs": 0,
            "note": "seed row: the base model with no adaptation, which is the "
                    "baseline your fine-tune has to beat",
        },
        corpus_source="seed script — replace with your real corpus",
        corpus_sha256=hashlib.sha256(b"seed-corpus").hexdigest(),
        corpus_sentence_count=0,
        held_out_perplexity=None,
        notes="Delete this row once you have logged a real training run.",
    )
    print("inserted training run", run["id"] if run else "(failed)")

    first_id = None
    for prompt, strategy, params, text, distinct_2 in SEED_PROMPTS:
        row = db.insert_generation(
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            strategy=strategy,
            decoding_params=params,
            generated_text=text,
            model_version="seed-v0",
            prompt_token_count=len(prompt.split()),
            generated_token_count=len(text.split()),
            distinct_2=distinct_2,
        )
        print("inserted generation", row["id"] if row else "(failed)")
        if row and first_id is None:
            first_id = row["id"]

    # Two independent raters on the same output, disagreeing by two points. That
    # disagreement is the interesting case in your report, not the average.
    if first_id is not None:
        now = datetime.now(timezone.utc).isoformat()
        for rater, score, note in [
            ("rater-a", 2, "Fluent but it loops. Unusable as a review."),
            ("rater-b", 4, "Grammatical and on topic; I scored fluency, not content."),
        ]:
            db.append_rating(
                first_id,
                {
                    "rater_id": rater,
                    "rating": score,
                    "dimensions": {"fluency": score, "coherence": score},
                    "notes": note,
                    "recorded_at": now,
                },
            )
        print(f"rated generation {first_id} twice (scores 2 and 4 — read the notes)")

    print("Done. Open the History tab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
