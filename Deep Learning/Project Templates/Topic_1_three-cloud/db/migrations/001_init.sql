-- 001_init.sql
-- Three tables for the "Regress-It" product.
-- Apply this in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
-- Or run it from db/seed.py which executes this file before seeding.

-- ---------------------------------------------------------------------------
-- datasets: one row per synthetic dataset generated from a known linear rule.
-- ---------------------------------------------------------------------------
create table if not exists datasets (
    id          bigint generated always as identity primary key,
    name        text        not null,
    slope       double precision not null,
    intercept   double precision not null,
    noise       double precision not null,
    n_points    integer     not null,
    -- xs/ys are stored so the API can pull the exact points it trains on.
    xs          double precision[] not null default '{}',
    ys          double precision[] not null default '{}',
    created_at  timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- runs: one row per training job. weights_json holds the fitted parameters.
-- ---------------------------------------------------------------------------
create table if not exists runs (
    id            bigint generated always as identity primary key,
    dataset_id    bigint      not null references datasets (id) on delete cascade,
    lr            double precision not null,
    batch_size    integer     not null,
    epochs        integer     not null,
    mse           double precision not null,
    mae           double precision not null,
    r2            double precision not null,
    weights_json  jsonb       not null,
    created_at    timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- predictions: one row per served /predict call (the prediction log).
-- ---------------------------------------------------------------------------
create table if not exists predictions (
    id          bigint generated always as identity primary key,
    run_id      bigint      not null references runs (id) on delete cascade,
    x           double precision not null,
    yhat        double precision not null,
    created_at  timestamptz not null default now()
);

create index if not exists idx_runs_dataset_id   on runs (dataset_id);
create index if not exists idx_runs_created_at    on runs (created_at desc);
create index if not exists idx_predictions_run_id on predictions (run_id);

-- ---------------------------------------------------------------------------
-- Row Level Security.
-- The Streamlit "Run History" tab reads the runs table with the ANON key, so we
-- allow anonymous SELECT on runs only. The API uses the SERVICE-ROLE key which
-- bypasses RLS entirely, so writes stay server-side.
-- ---------------------------------------------------------------------------
alter table datasets    enable row level security;
alter table runs        enable row level security;
alter table predictions enable row level security;

drop policy if exists "anon can read runs" on runs;
create policy "anon can read runs"
    on runs for select
    to anon
    using (true);
