-- ===========================================================================
-- 002_gradient_stats.sql — the table this product is actually about.
--
-- 001_init.sql answers "did it learn": one row per episode, one return per row.
-- This migration answers "why was it noisy", which is the question in the
-- product brief. Your team lead has heard that policy gradients are unstable
-- and wants evidence rather than adjectives; the evidence is a column.
--
-- Apply it AFTER 001_init.sql — migrations are ordered by filename and the
-- foreign key below will not resolve otherwise. Never edit 001 to add these
-- columns: a migration that has already been applied to a live project is
-- history, and rewriting history is how a schema and a database stop agreeing.
-- ===========================================================================

-- One row per POLICY UPDATE, not per episode -------------------------------
-- The grain is the single most important decision in this file. A batch of ten
-- episodes produces ten rows in `episodes` and exactly ONE row here, because
-- the gradient estimate whose variance we are measuring belongs to the batch.
-- Logging it per episode would produce a number that cannot be compared across
-- arms with different batch sizes, which is exactly the comparison this
-- product's headline chart makes.
create table if not exists gradient_stats (
    id                bigserial primary key,
    experiment_id     uuid    not null references experiments(id) on delete cascade,

    update_index      integer not null,          -- 0-based, monotone within a run
    episode_index     integer not null,          -- training episodes consumed BEFORE
                                                 -- this update; this is what lets the
                                                 -- variance chart and the learning
                                                 -- curve share an x-axis

    gradient_norm     double precision not null, -- ||batch-mean gradient||, the step taken
    gradient_variance double precision not null, -- trace of the covariance of the
                                                 -- per-episode gradient estimates:
                                                 -- sum_j Var_i(g_ij). Stored rather than
                                                 -- recomputed, because a plot you cannot
                                                 -- regenerate from the database is not
                                                 -- evidence.
    policy_entropy    double precision not null, -- mean H(pi(.|s)) in nats; <= ln 2 = 0.693
                                                 -- for a two-action environment

    -- Importance sampling. Null on the on-policy arms, which is a fact worth
    -- being able to query rather than a gap to be filled with zeros: zeros
    -- would silently average into `avg(is_weight_ess)` and make the on-policy
    -- arms look like catastrophically bad off-policy ones.
    off_policy          boolean not null default false,
    is_weight_mean      double precision,
    is_weight_max       double precision,
    is_weight_p95       double precision,
    is_weight_ess       double precision,        -- (sum w)^2 / (n * sum w^2), in [0, 1]
    is_weight_histogram jsonb,                   -- {"edges": [...], "counts": [...]}

    created_at        timestamptz not null default now(),

    -- One row per (run, update). The unique constraint is what makes a re-run
    -- that crashes halfway safe to restart: the retry conflicts instead of
    -- quietly doubling the row count and halving every average you compute.
    unique (experiment_id, update_index)
);

-- The index. Every query the Streamlit app issues against this table is
-- "all updates for these runs, in order" — the Gradient Variance tab draws four
-- arms times three seeds, so twelve of these on one page load. Without the
-- index that is twelve sequential scans, and the free tier will make you feel
-- it.
create index if not exists gradient_stats_experiment_idx
    on gradient_stats (experiment_id, update_index);

-- The comparison the product exists to make, as a view. Keeping it in SQL
-- rather than in pandas is the point of having a data tier: the claim
-- "the baseline reduced gradient variance by a factor of K" is then a query
-- that a reader can run against your project, not a number in your README.
create or replace view gradient_variance_by_arm as
select
    e.algorithm,
    (e.hyperparameters ->> 'use_baseline')::boolean            as use_baseline,
    (e.hyperparameters ->> 'use_importance_sampling')::boolean as use_importance_sampling,
    g.update_index,
    count(*)                        as seeds,          -- how many runs back this point
    avg(g.gradient_variance)        as mean_gradient_variance,
    stddev_samp(g.gradient_variance) as sd_gradient_variance,
    avg(g.gradient_norm)            as mean_gradient_norm,
    avg(g.policy_entropy)           as mean_policy_entropy
from gradient_stats g
join experiments e on e.id = g.experiment_id
group by 1, 2, 3, 4;

-- Row-level security, matching 001_init.sql: the anon key the Streamlit tier
-- holds may read and may not write. A new table does NOT inherit the policies
-- of the tables around it, and forgetting this line is how a product ships with
-- one table publicly writable.
alter table gradient_stats enable row level security;

do $$
begin
  if not exists (select 1 from pg_policies where policyname = 'anon_read_gradient_stats') then
    create policy anon_read_gradient_stats on gradient_stats for select to anon using (true);
  end if;
end $$;
