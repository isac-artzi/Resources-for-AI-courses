-- 001_init.sql — MoodLens schema (Cloud #3).
--
-- Apply it in the Supabase dashboard: SQL Editor → New query → paste → Run.
-- It is idempotent, so re-running it is safe.
--
-- TWO TABLES, AND YOU NEED BOTH.
--
--   runs         one row per TRAINING run: what you trained, on what, with which
--                hyperparameters, and every metric it produced. This is the
--                evidence behind "the model is 89% accurate".
--
--   predictions  one row per SERVED prediction: what the deployed service told
--                a user, when, with which model version, and with the aspect
--                breakdown that came with it. This is the evidence behind "on
--                Tuesday afternoon we told this customer their review was
--                negative".
--
-- Neither answers the other's question. Held-out metrics say nothing about what
-- production actually did; a pile of predictions with no evaluation behind them
-- says nothing about whether any of them were right. An audit needs to join the
-- two: take a prediction, read its model_version, find the run that produced
-- that version, and read the slice metrics for the bucket that prediction falls
-- into. That join is only possible because model_version appears in both tables,
-- which is the only reason it is duplicated.
--
-- Notice what is NOT in either table: a column for the review text.

-- ---------------------------------------------------------------------------
-- runs — training runs
-- ---------------------------------------------------------------------------
create table if not exists runs (
    id              bigint generated always as identity primary key,

    -- The version string the API will serve under, e.g. "moodlens-v2".
    -- GET /version reports it; every prediction row carries it.
    model_version   text        not null,

    -- The pretrained checkpoint you fine-tuned FROM, e.g. "distilbert-base-uncased".
    base_model      text        not null default 'unset',

    -- Which corpus and which split. "imdb (train 20000 / val 5000 / test 25000,
    -- seed 13)" is a good value. "imdb" is not, because it does not let anyone
    -- reproduce you.
    dataset         text        not null default 'unset',

    -- Hyperparameters and everything else needed to run it again: learning rate,
    -- epochs, batch size, max sequence length, seed, truncation strategy.
    config          jsonb       not null default '{}'::jsonb,

    -- The whole evaluation payload. The Model Performance and Bias Audit tabs
    -- read this column directly with the anon key, so its shape is a contract:
    --   {
    --     "documents": [DocumentMetrics, ...],   -- transformer AND baseline
    --     "aspects":   [AspectMetrics, ...],     -- one per aspect
    --     "slices":    [SliceMetrics, ...]       -- at least two slice_names
    --   }
    -- See shared/schemas.py for the field names and db/seed.py for a worked
    -- example of the layout.
    metrics         jsonb       not null default '{}'::jsonb,

    n_train         integer,
    n_eval          integer,

    -- Room for the sentence you will otherwise forget: "aspect labels hand
    -- annotated on 300 reviews by two annotators, disagreements resolved by
    -- discussion".
    notes           text,

    created_at      timestamptz not null default now()
);

create index if not exists idx_runs_created_at on runs (created_at desc);
create index if not exists idx_runs_version    on runs (model_version);

-- ---------------------------------------------------------------------------
-- predictions — everything the service served
-- ---------------------------------------------------------------------------
create table if not exists predictions (
    id                    bigint generated always as identity primary key,

    -- sha256 of the input. Storing the hash instead of the text means a
    -- prediction can be identified, de-duplicated and complained about without
    -- the database holding anyone's writing. If two rows share a hash, they had
    -- identical input — which is also how you spot the same review being scored
    -- fifty times by a retry loop.
    text_sha256           text        not null,

    -- Length is worth keeping even though the text is not: it is the observed
    -- attribute the review_length slice in the bias audit is built on, and you
    -- cannot recompute it later from a hash.
    char_count            integer,

    label                 text        not null check (label in ('negative', 'positive')),

    -- The calibrated P(positive) that was actually served, and the confidence
    -- derived from it. calibrated=false means no calibrator was in play; store
    -- it rather than letting a future reader assume.
    probability_positive  double precision,
    confidence            double precision,
    calibrated            boolean     not null default false,

    -- The aspect breakdown exactly as served:
    --   [{"aspect": "acting", "label": "positive", "score": 0.81,
    --     "evidence": ["the lead is superb"]}, ...]
    -- Stored with the prediction, not in a side table, because an audit that
    -- can see the label but not the aspect that drove it cannot explain it.
    -- jsonb (not json) so you can query inside it later:
    --   select count(*) from predictions
    --   where aspects @> '[{"aspect":"service","label":"negative"}]';
    aspects               jsonb       not null default '[]'::jsonb,

    -- "transformer" or "tfidf-baseline". If you ever serve the baseline as a
    -- fallback when the model fails to load, this column is the only way anyone
    -- will find out afterwards.
    model_name            text        not null default 'transformer',

    -- Joins back to runs.model_version. This is the audit trail.
    model_version         text        not null default 'unset',

    created_at            timestamptz not null default now()
);

create index if not exists idx_pred_created_at on predictions (created_at desc);
create index if not exists idx_pred_version    on predictions (model_version);
create index if not exists idx_pred_label      on predictions (label);
create index if not exists idx_pred_sha        on predictions (text_sha256);

-- ---------------------------------------------------------------------------
-- Row Level Security.
--
-- The Streamlit tier queries `runs` directly with the ANON key to draw the
-- Model Performance and Bias Audit tabs, so anonymous SELECT is allowed on both
-- tables. No anonymous INSERT policy exists, which means the anon key cannot
-- write. The API uses the SERVICE-ROLE key, which bypasses RLS entirely — that
-- is why writes are server-side only.
--
-- Anonymous read on `predictions` is safe here only because of the decision two
-- screens up: there is no text column. If you ever add one — don't — this
-- policy hands every visitor a copy of your users' reviews.
--
-- Turning RLS off "to make it work" is the single most common way student
-- projects end up with a publicly writable database. Don't.
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
