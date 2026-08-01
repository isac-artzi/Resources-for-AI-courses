-- 001_init.sql — TokenForge schema (Cloud #3).
--
-- Apply it in the Supabase dashboard: SQL Editor → New query → paste → Run.
-- It is idempotent, so re-running it is safe.
--
-- ONE TABLE, ON PURPOSE. Topic 1 logs runs and nothing else. Later topics add
-- a predictions table alongside this one, because a service that only records
-- training but never records what it answered has no audit trail. Notice what
-- is NOT here: a column for the input text.

create table if not exists runs (
    id                  bigint generated always as identity primary key,

    -- "preprocess" or "tokenize". One table for both keeps the History tab a
    -- single query; the kind column is what lets you filter it.
    kind                text        not null check (kind in ('preprocess', 'tokenize')),

    -- sha256 of the input. Storing the hash instead of the text means a run can
    -- be identified and de-duplicated without holding anyone's data. If two runs
    -- share a hash, they had identical input.
    text_sha256         text        not null,

    -- The exact options or tokenizer list used, as JSON. This plus the hash is
    -- what makes a run reproducible.
    config              jsonb       not null default '{}'::jsonb,

    token_count_before  integer,
    token_count_after   integer,
    oov_rate            double precision,

    -- Free-text version of the NLP configuration, e.g. "tokenforge-v3".
    model_version       text        not null default 'unset',

    created_at          timestamptz not null default now()
);

create index if not exists idx_runs_created_at on runs (created_at desc);
create index if not exists idx_runs_kind       on runs (kind);
create index if not exists idx_runs_sha        on runs (text_sha256);

-- ---------------------------------------------------------------------------
-- Row Level Security.
--
-- The Streamlit History tab queries this table directly with the ANON key, so
-- anonymous SELECT is allowed. No anonymous INSERT policy exists, which means
-- the anon key cannot write. The API uses the SERVICE-ROLE key, which bypasses
-- RLS entirely — that is why writes are server-side only.
--
-- Turning RLS off "to make it work" is the single most common way student
-- projects end up with a publicly writable database. Don't.
-- ---------------------------------------------------------------------------
alter table runs enable row level security;

drop policy if exists "anon can read runs" on runs;
create policy "anon can read runs"
    on runs for select
    to anon
    using (true);
