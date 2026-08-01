-- 001_init.sql — TagWise schema (Cloud #3).
--
-- Apply it in the Supabase dashboard: SQL Editor → New query → paste → Run.
-- It is idempotent, so re-running it is safe.
--
-- TWO TABLES, AND THE SPLIT IS THE LESSON.
--
--   runs      — one row per tagger BUILD. Built the lookup table; fine-tuned the
--               transformer. Hyperparameters, metrics, model version.
--   taggings  — one row per SERVED REQUEST. Hashed sentence, predicted tags,
--               which model answered, when.
--
-- A build log and a request log answer different questions, and neither can
-- stand in for the other. runs tells you what a tagger scored on a held-out
-- split on the day you built it — a handful of rows for the whole project.
-- taggings tells you what the service actually did: how many sentences it saw,
-- how often the baseline hit an unknown word in real traffic, whether the new
-- model version changed anything. If you log only runs, the History tab has
-- nothing to read: a table with four rows in it, none of which are requests, is
-- not a history of anything a user did.
--
-- Notice what is NOT here: a column for the sentence.

create table if not exists runs (
    id                bigint generated always as identity primary key,

    -- "baseline" or "transformer". One table for both is what makes the
    -- comparison tab a single query.
    model             text        not null check (model in ('baseline', 'transformer')),

    -- Which label set the numbers below were computed over, e.g. "UPOS".
    -- Accuracy on 17 tags and accuracy on 45 tags are different quantities, and
    -- storing the tag set next to the number is how you stop yourself comparing
    -- them by accident six weeks later.
    tagset            text        not null default 'UPOS',

    -- Everything needed to rebuild this tagger, as JSON: treebank and revision,
    -- split sizes, base checkpoint, learning rate, epochs, seed — and for the
    -- baseline, the casing and tie-breaking rules. The lookup table has
    -- hyperparameters too; they are decisions, not numbers.
    hyperparameters   jsonb       not null default '{}'::jsonb,

    accuracy          double precision,
    macro_f1          double precision,

    -- Per-tag F1 and the confusion matrix, as
    -- {"confusion": {"labels": [...], "matrix": [[...]]}}. The comparison tab
    -- renders straight out of this column, which is why the API needs no
    -- evaluation endpoint: an evaluation is a fact about a build, computed once.
    metrics           jsonb       not null default '{}'::jsonb,

    model_version     text        not null default 'unset',
    notes             text,
    created_at        timestamptz not null default now()
);

create index if not exists idx_runs_created_at on runs (created_at desc);
create index if not exists idx_runs_model      on runs (model);

create table if not exists taggings (
    id                bigint generated always as identity primary key,

    -- sha256 of the input sentence. Storing the hash instead of the text means a
    -- request can be identified and de-duplicated without holding anyone's data.
    -- If two rows share a hash, they had identical input. A tagged sentence is
    -- still the sentence.
    sentence_sha256   text        not null,

    token_count       integer     not null,

    -- The predicted tags, in order. Postgres text[] keeps them queryable: you
    -- can ask how many taggings contained a PROPN without unpacking JSON.
    tag_sequence      text[]      not null default '{}',

    model             text        not null check (model in ('baseline', 'transformer')),
    model_version     text        not null default 'unset',

    -- How many tokens the baseline had to guess with the fallback rules. This is
    -- the column that tells you your live traffic is nothing like your test
    -- split — a fallback rate far above what you measured means the words your
    -- users type are not the words the treebank contains.
    unknown_count     integer     not null default 0,

    created_at        timestamptz not null default now()
);

create index if not exists idx_taggings_created_at on taggings (created_at desc);
create index if not exists idx_taggings_model      on taggings (model);
create index if not exists idx_taggings_sha        on taggings (sentence_sha256);

-- ---------------------------------------------------------------------------
-- Row Level Security.
--
-- The Streamlit tabs query these tables directly with the ANON key, so anonymous
-- SELECT is allowed on both. No anonymous INSERT policy exists, which means the
-- anon key cannot write. The API uses the SERVICE-ROLE key, which bypasses RLS
-- entirely — that is why writes are server-side only.
--
-- Turning RLS off "to make it work" is the single most common way student
-- projects end up with a publicly writable database. Don't.
-- ---------------------------------------------------------------------------
alter table runs     enable row level security;
alter table taggings enable row level security;

drop policy if exists "anon can read runs" on runs;
create policy "anon can read runs"
    on runs for select
    to anon
    using (true);

drop policy if exists "anon can read taggings" on taggings;
create policy "anon can read taggings"
    on taggings for select
    to anon
    using (true);
