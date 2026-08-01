-- ===========================================================================
-- 001_init.sql — the standing schema, identical in every product this course
-- ships. Paste into the Supabase SQL editor, or run with `supabase db push`.
--
-- Design notes worth reading before you extend it:
--
--  * `episodes` is the big table. A single training run in this course writes
--    20,000 rows, and you will have dozens of runs. Index it or the run-history
--    view will time out on the free tier.
--  * `epsilon` on `episodes` is not optional bookkeeping. It is what makes the
--    exploration tax recomputable from the data months later.
--  * Every claim in your report should be expressible as a query against this
--    schema. If a number in your README cannot be, it is not yet evidence.
-- ===========================================================================

create extension if not exists "pgcrypto";

-- One row per training configuration ---------------------------------------
create table if not exists experiments (
    id              uuid primary key default gen_random_uuid(),
    algorithm       text        not null,
    env_id          text        not null,
    seed            integer     not null,
    hyperparameters jsonb       not null default '{}'::jsonb,
    git_sha         text,
    notes           text,
    created_at      timestamptz not null default now()
);

-- One row per training episode ---------------------------------------------
create table if not exists episodes (
    id            bigserial primary key,
    experiment_id uuid        not null references experiments(id) on delete cascade,
    episode_index integer     not null,
    "return"      double precision not null,
    length        integer     not null,
    epsilon       double precision,
    created_at    timestamptz not null default now(),
    unique (experiment_id, episode_index)
);
create index if not exists episodes_experiment_idx on episodes (experiment_id, episode_index);

-- One row per GREEDY evaluation sweep. Deliberately a separate table from
-- `episodes`: training return and greedy return answer different questions,
-- and storing them together is how students end up reporting the first as
-- though it were the second.
create table if not exists evaluations (
    id                  bigserial primary key,
    experiment_id       uuid        not null references experiments(id) on delete cascade,
    at_training_episode integer     not null,
    episodes            integer     not null,
    mean_return         double precision not null,
    std_return          double precision not null,
    stderr_return       double precision not null,
    created_at          timestamptz not null default now()
);
create index if not exists evaluations_experiment_idx on evaluations (experiment_id);

-- The artifact registry ----------------------------------------------------
create table if not exists policies (
    id            uuid primary key default gen_random_uuid(),
    name          text        not null,
    format        text        not null default 'npz',
    bytes         integer     not null,
    sha256        text        not null,
    kind          text        not null default 'tabular',
    obs_dim       integer,
    n_actions     integer,
    experiment_id uuid references experiments(id) on delete set null,
    created_at    timestamptz not null default now(),
    unique (sha256)
);

-- Every /act and /rollout call. State is stored as a HASH, never raw: in
-- Topic 6 the "state" is user text and raw logging would put user content in
-- your database.
create table if not exists audit_log (
    id            bigserial primary key,
    endpoint      text        not null,
    policy_sha256 text,
    state_hash    text,
    action        text,
    created_at    timestamptz not null default now()
);
create index if not exists audit_log_created_idx on audit_log (created_at desc);

-- The view GET /runs reads. Keeping the aggregation in SQL rather than in
-- Python is what makes the run-history tab cheap on the free tier.
create or replace view run_summary as
select
    e.id                                as experiment_id,
    e.algorithm,
    e.env_id,
    e.seed,
    e.hyperparameters,
    e.git_sha,
    e.created_at,
    count(ep.id)                        as episodes_logged,
    avg(ep."return") filter (
        where ep.episode_index > (select max(episode_index) - 100
                                  from episodes where experiment_id = e.id)
    )                                   as mean_return_last_100,
    (select mean_return   from evaluations v where v.experiment_id = e.id
       order by v.at_training_episode desc limit 1) as eval_mean_return,
    (select stderr_return from evaluations v where v.experiment_id = e.id
       order by v.at_training_episode desc limit 1) as eval_stderr
from experiments e
left join episodes ep on ep.experiment_id = e.id
group by e.id;

-- Row-level security: the anon key used by the Streamlit read-only views must
-- not be able to write. Enable RLS and grant select only.
alter table experiments  enable row level security;
alter table episodes     enable row level security;
alter table evaluations  enable row level security;
alter table policies     enable row level security;
alter table audit_log    enable row level security;

do $$
begin
  if not exists (select 1 from pg_policies where policyname = 'anon_read_experiments') then
    create policy anon_read_experiments on experiments for select to anon using (true);
    create policy anon_read_episodes    on episodes    for select to anon using (true);
    create policy anon_read_evaluations on evaluations for select to anon using (true);
    create policy anon_read_policies    on policies    for select to anon using (true);
  end if;
end $$;
