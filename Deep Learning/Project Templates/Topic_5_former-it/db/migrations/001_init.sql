-- 001_init.sql
-- Tables for the "Former-It" product: from-scratch Transformer encoder classifier.
-- Apply this in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.

-- ---------------------------------------------------------------------------
-- datasets: one row per synthetic sequence dataset. Only the generation
-- PARAMETERS are stored -- the symbol sequences are regenerated deterministically
-- from `seed`, so no raw sequences live in the database.
-- ---------------------------------------------------------------------------
create table if not exists datasets (
    id          bigint generated always as identity primary key,
    name        text        not null,
    n_rows      integer     not null,
    seq_len     integer     not null,
    vocab_size  integer     not null,
    n_classes   integer     not null,
    seed        integer     not null,
    created_at  timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- runs: one row per training job (held-out metrics). Anon-readable.
-- ---------------------------------------------------------------------------
create table if not exists runs (
    id          bigint generated always as identity primary key,
    dataset_id  bigint      not null references datasets (id) on delete cascade,
    lr          double precision not null,
    batch_size  integer     not null,
    epochs      integer     not null,
    accuracy    double precision not null,
    macro_f1    double precision not null,
    created_at  timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- run_artifacts: the fitted Transformer, base64-encoded. Not anon-readable.
-- ---------------------------------------------------------------------------
create table if not exists run_artifacts (
    run_id      bigint      primary key references runs (id) on delete cascade,
    model_b64   text        not null,
    created_at  timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- sequence_metadata: one row per served prediction. We store the sequence's
-- SHA-256 hash + length + predicted class -- NEVER the raw symbols (privacy
-- invariant).
-- ---------------------------------------------------------------------------
create table if not exists sequence_metadata (
    id          bigint generated always as identity primary key,
    run_id      bigint      not null references runs (id) on delete cascade,
    sha256      text        not null,
    length      integer     not null,
    label       integer     not null,
    class_name  text        not null,
    confidence  double precision not null,
    created_at  timestamptz not null default now()
);

create index if not exists idx_runs_dataset_id          on runs (dataset_id);
create index if not exists idx_runs_created_at           on runs (created_at desc);
create index if not exists idx_sequence_metadata_run_id  on sequence_metadata (run_id);

-- ---------------------------------------------------------------------------
-- Row Level Security. The Streamlit "Run History" tab reads `runs` with the
-- ANON key, so we allow anonymous SELECT on runs only. The API uses the
-- SERVICE-ROLE key which bypasses RLS, so all writes stay server-side.
-- ---------------------------------------------------------------------------
alter table datasets          enable row level security;
alter table runs              enable row level security;
alter table run_artifacts     enable row level security;
alter table sequence_metadata enable row level security;

drop policy if exists "anon can read runs" on runs;
create policy "anon can read runs"
    on runs for select
    to anon
    using (true);
