-- 001_init.sql — Classify-It schema (Cloud #3).
--
-- Apply it in the Supabase dashboard: SQL Editor → New query → paste → Run.
-- It is idempotent, so re-running it is safe.
--
-- TWO TABLES, AND THE SPLIT IS THE POINT.
--
--   runs         one row per TRAINING run. What you fit, with which
--                hyperparameters, and what it scored on the held-out split.
--   predictions  one row per SERVED prediction. What the deployed service was
--                asked, what it answered, how confident it was, and which
--                model version said it.
--
-- A service that logs its training but never logs what it answered has no audit
-- trail. "Our F1 was 0.88" says nothing about the specific decision a customer
-- is complaining about. You need both halves and the model_version column that
-- joins them: from a prediction row you can reach the run that produced the
-- model, and from a run you can find every decision it made while it was live.
--
-- Notice what is NOT in either table: a column for the input text. The corpus is
-- support messages and survey free text. You store the sha256 hash, which is
-- enough to recognize a repeated input and nothing more.

-- ---------------------------------------------------------------------------
-- Training runs
-- ---------------------------------------------------------------------------
create table if not exists runs (
    id                bigint generated always as identity primary key,

    -- The two models this product serves. The check constraint is the single-
    -- label binary product definition leaking, correctly, into the database:
    -- there is a classical baseline and there is a fine-tuned encoder, and a
    -- comparison between them is what the assignment asks for.
    model_kind        text        not null check (model_kind in ('baseline', 'transformer')),

    dataset_name      text,

    -- Everything needed to re-run this training. For the baseline: ngram_range,
    -- min_df, max_features, C, class_weight. For the transformer: checkpoint,
    -- learning_rate, epochs, batch_size, max_length, seed. jsonb rather than
    -- columns so adding a knob does not need a migration.
    hyperparameters   jsonb       not null default '{}'::jsonb,

    -- accuracy / precision / recall / f1 / positive_label / n_examples /
    -- support_positive, as JSON. All four numbers, always: accuracy alone on an
    -- imbalanced corpus is the metric that makes a useless model look fine.
    metrics           jsonb       not null default '{}'::jsonb,

    -- Identifies the artifact this run produced, e.g. 'distilbert-ft-v2'. This
    -- is the join key to the predictions table.
    model_version     text        not null default 'unset',

    n_train           integer,
    n_eval            integer,

    created_at        timestamptz not null default now()
);

create index if not exists idx_runs_created_at on runs (created_at desc);
create index if not exists idx_runs_model_kind on runs (model_kind);
create index if not exists idx_runs_version    on runs (model_version);

-- ---------------------------------------------------------------------------
-- Served predictions
-- ---------------------------------------------------------------------------
create table if not exists predictions (
    id                bigint generated always as identity primary key,

    -- sha256 of the input. Storing the hash instead of the text means a
    -- prediction can be identified and de-duplicated without holding anyone's
    -- message. If two rows share a hash, they had identical input.
    text_sha256       text        not null,

    predicted_label   text        not null,

    -- The CALIBRATED probability of the predicted label, so it is always at
    -- least 0.5 in a two-class problem. If you see rows below 0.5 here, your
    -- serving code is reporting the positive class's probability regardless of
    -- what it predicted — a real bug, and this constraint catches it.
    probability       double precision not null check (probability >= 0.0 and probability <= 1.0),

    model_kind        text        not null check (model_kind in ('baseline', 'transformer')),

    -- Which artifact answered. Join to runs.model_version to recover the
    -- hyperparameters and held-out metrics of the model that made this call.
    model_version     text        not null default 'unset',

    latency_ms        double precision,

    created_at        timestamptz not null default now()
);

create index if not exists idx_predictions_created_at on predictions (created_at desc);
create index if not exists idx_predictions_version    on predictions (model_version);
create index if not exists idx_predictions_sha        on predictions (text_sha256);
create index if not exists idx_predictions_label      on predictions (predicted_label);

-- ---------------------------------------------------------------------------
-- Row Level Security.
--
-- The Streamlit tier queries these tables directly with the ANON key, so
-- anonymous SELECT is allowed on both. No anonymous INSERT policy exists, which
-- means the anon key cannot write. The API uses the SERVICE-ROLE key, which
-- bypasses RLS entirely — that is why writes are server-side only.
--
-- Anonymous read is defensible here only because neither table contains input
-- text. If you add a text column "just for debugging", you have published your
-- users' support messages to anyone who opens the browser console. Don't.
--
-- Turning RLS off "to make the tab work" is the single most common way student
-- projects end up with a publicly writable database. Don't do that either.
-- ---------------------------------------------------------------------------
alter table runs        enable row level security;
alter table predictions enable row level security;

drop policy if exists "anon can read runs" on runs;
create policy "anon can read runs"
    on runs for select
    to anon
    using (true);

drop policy if exists "anon can read predictions" on predictions;
create policy "anon can read predictions"
    on predictions for select
    to anon
    using (true);
