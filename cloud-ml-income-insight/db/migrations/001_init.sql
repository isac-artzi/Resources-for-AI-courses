-- 001_init.sql
-- Tables for the "Income-Insight" product: tabular MLP classifier.
-- Apply this in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.

-- ---------------------------------------------------------------------------
-- datasets: one row per synthetic tabular dataset (records + labels as JSONB).
-- No real PII is stored -- every row is fabricated from a known logistic rule.
-- ---------------------------------------------------------------------------
create table if not exists datasets (
    id             bigint generated always as identity primary key,
    name           text        not null,
    n_rows         integer     not null,
    n_features     integer     not null,
    positive_rate  double precision not null,
    records        jsonb       not null default '[]',
    labels         jsonb       not null default '[]',
    created_at     timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- runs: one row per training job (held-out classification metrics only).
-- Anon-readable for the UI's "Run History" tab.
-- ---------------------------------------------------------------------------
create table if not exists runs (
    id          bigint generated always as identity primary key,
    dataset_id  bigint      not null references datasets (id) on delete cascade,
    hidden_dim  integer     not null,
    lr          double precision not null,
    batch_size  integer     not null,
    epochs      integer     not null,
    accuracy    double precision not null,
    precision   double precision not null,
    recall      double precision not null,
    f1          double precision not null,
    roc_auc     double precision not null,
    created_at  timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- run_artifacts: the fitted (preprocessor + MLP) blob, base64-encoded.
-- Kept out of `runs` and NOT anon-readable so the public key never sees it.
-- ---------------------------------------------------------------------------
create table if not exists run_artifacts (
    run_id      bigint      primary key references runs (id) on delete cascade,
    model_b64   text        not null,
    created_at  timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- predictions: one row per served /predict call (the prediction / audit log).
-- ---------------------------------------------------------------------------
create table if not exists predictions (
    id          bigint generated always as identity primary key,
    run_id      bigint      not null references runs (id) on delete cascade,
    features    jsonb       not null,
    proba       double precision not null,
    label       integer     not null,
    created_at  timestamptz not null default now()
);

create index if not exists idx_runs_dataset_id   on runs (dataset_id);
create index if not exists idx_runs_created_at    on runs (created_at desc);
create index if not exists idx_predictions_run_id on predictions (run_id);

-- ---------------------------------------------------------------------------
-- Row Level Security.
-- The Streamlit "Run History" tab reads the runs table with the ANON key, so we
-- allow anonymous SELECT on runs only. The API uses the SERVICE-ROLE key which
-- bypasses RLS entirely, so writes (and reads of artifacts/predictions/datasets)
-- stay server-side.
-- ---------------------------------------------------------------------------
alter table datasets      enable row level security;
alter table runs          enable row level security;
alter table run_artifacts enable row level security;
alter table predictions   enable row level security;

drop policy if exists "anon can read runs" on runs;
create policy "anon can read runs"
    on runs for select
    to anon
    using (true);
