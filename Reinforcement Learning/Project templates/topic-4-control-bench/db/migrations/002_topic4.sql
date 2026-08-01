-- ===========================================================================
-- 002_topic4.sql — the two tables this product is actually about.
--
-- 001_init.sql answers "did it learn": one row per episode, one return per row.
-- This migration answers the two questions in the Topic 4 brief that a return
-- column cannot:
--
--   * `policy_updates` — how big was each update, how much entropy did the
--     policy keep, and (for PPO) HOW FAR DID THE POLICY MOVE. That last column,
--     `kl_divergence`, is what makes the trust-region argument checkable in data
--     rather than only in prose: PPO clips a ratio and hopes the KL stays small,
--     and whether the hope held on your run is an empirical question.
--   * `entropy_sweep` — one row per SAC run in the temperature study, carrying
--     convergence speed, final performance, within-run spread and mean policy
--     entropy, so the α comparison is a query rather than a screenshot.
--
-- It also adds two columns to tables 001 already created. Both are additive and
-- both are nullable, which is what makes this migration safe to apply to a
-- project that already holds Topic 1–3 data.
--
-- Apply it AFTER 001_init.sql — migrations are ordered by filename and the
-- foreign keys below will not resolve otherwise. Never edit 001 to add these
-- columns: a migration that has already been applied to a live project is
-- history, and rewriting history is how a schema and a database stop agreeing.
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- Additive changes to the standing schema
-- ---------------------------------------------------------------------------

-- Cumulative environment steps at the end of each episode. Topic 4's headline
-- comparison is at MATCHED STEP BUDGETS, and episode index is not a step
-- budget: an A2C run whose episodes last 20 steps and a PPO run whose episodes
-- last 500 have consumed twenty-five times more experience at the same episode
-- number. Without this column the bake-off chart cannot be drawn honestly.
-- Nullable because Topics 1-3 wrote rows without it, and backfilling history
-- with a guess is worse than a null that says "not recorded".
alter table episodes add column if not exists env_steps integer;
create index if not exists episodes_env_steps_idx on episodes (experiment_id, env_steps);

-- The artifact registry gains the two facts the service needs to route a
-- request. `env_id` is what /rollout uses to construct the right environment;
-- `action_space` is what tells a caller whether `action` will come back as an
-- integer or as a list of floats. Both are read out of the .npz at load time —
-- these columns are the queryable copy, not the source of truth.
alter table policies add column if not exists env_id text;
alter table policies add column if not exists action_space text
    check (action_space is null or action_space in ('discrete', 'continuous'));

-- ---------------------------------------------------------------------------
-- One row per POLICY UPDATE, for all three algorithms
-- ---------------------------------------------------------------------------
-- The grain is the single most important decision in this file. A PPO iteration
-- that collects 1,024 steps and then takes ten epochs of minibatch updates
-- produces roughly two `episodes` rows and exactly ONE row here, because the KL
-- being reported is the distance between the policy that COLLECTED the batch
-- and the policy that exists after the whole iteration — which is the quantity
-- the clip is supposed to bound. A KL logged per minibatch would give forty
-- numbers per iteration, none of which answers that question.
--
-- One table for three algorithms, with nullable columns, rather than three
-- tables. The comparison the product exists to make is ACROSS algorithms, and a
-- comparison spread over three tables is three queries and a join you will get
-- wrong at least once.
create table if not exists policy_updates (
    id             bigserial primary key,
    experiment_id  uuid    not null references experiments(id) on delete cascade,

    update_index   integer not null,          -- 0-based, monotone within a run
    env_steps      integer not null,          -- environment steps consumed BEFORE this
                                              -- update; the x-axis of every matched-
                                              -- budget chart in this product
    episode_index  integer not null,          -- episodes completed before this update

    policy_loss    double precision,
    value_loss     double precision,          -- critic loss; required for SAC by the brief
    policy_entropy double precision,          -- mean H(pi(.|s)) in nats. <= ln 2 = 0.693 for
                                              -- CartPole and <= ln 3 = 1.099 for Acrobot;
                                              -- NEGATIVE for SAC's continuous actor, whose
                                              -- DIFFERENTIAL entropy has no lower bound.
                                              -- Do not put the two on one axis unlabelled.

    -- PPO only. Null on A2C and SAC, and null is the honest encoding: A2C takes
    -- one step per batch and never asks how far it moved, and SAC has no trust
    -- region at all. Zeros here would average into avg(kl_divergence) and make
    -- A2C look like the most conservative method in the study.
    kl_divergence  double precision,
    clip_fraction  double precision,          -- fraction of samples whose ratio left
                                              -- [1-eps, 1+eps]. Near 0 means the clip never
                                              -- engaged; near 1 means the batch was mostly
                                              -- outside the trust region and wasted.

    -- SAC only. Logged PER UPDATE rather than per run because under automatic
    -- tuning it moves, and its trajectory is the evidence for the claim that
    -- automatic tuning does something a fixed value cannot.
    alpha          double precision,

    created_at     timestamptz not null default now(),

    -- One row per (run, update). The unique constraint is what makes a re-run
    -- that crashed halfway safe to restart: the retry conflicts instead of
    -- quietly doubling the row count and halving every average you compute.
    unique (experiment_id, update_index)
);

-- Every query the Streamlit app issues against this table is "all updates for
-- these runs, in order" — the Bake-Off tab draws two algorithms times two
-- environments times three seeds, so twelve of these on one page load. Without
-- the index that is twelve sequential scans, and the free tier will make you
-- feel it.
create index if not exists policy_updates_experiment_idx
    on policy_updates (experiment_id, update_index);
-- A second index on env_steps, because the matched-budget chart asks
-- "everything up to step N" rather than "everything up to update N", and the
-- two orderings only coincide for one algorithm at a time.
create index if not exists policy_updates_steps_idx
    on policy_updates (experiment_id, env_steps);

-- ---------------------------------------------------------------------------
-- One row per SAC run in the temperature study
-- ---------------------------------------------------------------------------
-- A different GRAIN from policy_updates: every column here is a summary of a
-- whole run. Mixing a per-update series and a per-run summary in one table
-- forces every query to carry a filter that a reader has to notice in order to
-- trust the number.
create table if not exists entropy_sweep (
    id            bigserial primary key,
    experiment_id uuid not null references experiments(id) on delete cascade,

    mode          text not null check (mode in ('fixed', 'auto')),
    alpha_setting text not null,              -- 'alpha=0.5' | 'alpha=0.01' | 'auto'.
                                              -- Text, so that 'auto' does not have to be
                                              -- encoded as a magic number and the UI can
                                              -- group on it directly.
    alpha_value   double precision,           -- the fixed alpha, or the FINAL alpha under
                                              -- automatic tuning. Where the tuner ended up
                                              -- relative to the two hand-chosen values is
                                              -- the most interesting number in the sweep.
    seed          integer not null,
    episodes      integer not null,
    env_steps     integer not null,

    -- Convergence speed. NULL means the run never reached the bar, which is a
    -- RESULT. Do not encode it as 9999: that number would be averaged into
    -- avg(episodes_to_threshold) and would invent a slow convergence where
    -- there was none at all. The `entropy_sweep_summary` view below therefore
    -- reports the mean over the runs that reached it AND how many did.
    episodes_to_threshold integer,
    threshold             double precision not null,

    -- Final performance and WITHIN-run spread.
    mean_return_last_100  double precision not null,
    return_std_last_100   double precision not null,

    -- Exploration. Negative values are expected and correct: this is the
    -- differential entropy of a continuous density, which is not bounded below
    -- by zero the way a discrete entropy is.
    mean_policy_entropy   double precision not null,

    eval_mean_return      double precision,   -- deterministic (modal) evaluation

    created_at    timestamptz not null default now(),

    -- One row per (arm, seed). Re-running a cell conflicts rather than silently
    -- adding a second row that would double-count in every average.
    unique (experiment_id)
);
create index if not exists entropy_sweep_arm_idx on entropy_sweep (alpha_setting, seed);

-- ---------------------------------------------------------------------------
-- The views the UI and the report read
-- ---------------------------------------------------------------------------

-- The alpha comparison, as a query rather than as a screenshot. Note that
-- STABILITY is stddev_samp of the per-seed final returns — the spread ACROSS
-- SEEDS, which is a different quantity from `return_std_last_100`, the spread
-- WITHIN one run. A configuration can be steady within a seed and wildly
-- seed-dependent, and only the first of those two numbers would notice.
create or replace view entropy_sweep_summary as
select
    s.alpha_setting,
    s.mode,
    count(*)                                     as seeds,
    avg(s.mean_return_last_100)                  as mean_final_return,
    stddev_samp(s.mean_return_last_100)          as across_seed_std,
    var_samp(s.mean_return_last_100)             as across_seed_variance,
    avg(s.return_std_last_100)                   as mean_within_run_std,
    avg(s.mean_policy_entropy)                   as mean_policy_entropy,
    avg(s.alpha_value)                           as mean_final_alpha,
    count(s.episodes_to_threshold)               as reached_threshold,
    avg(s.episodes_to_threshold)                 as mean_episodes_to_threshold,
    min(s.threshold)                             as threshold,
    avg(s.eval_mean_return)                      as mean_eval_return
from entropy_sweep s
group by s.alpha_setting, s.mode;

-- PPO's trust-region evidence, per update, with the seeds it came from. This is
-- the series Topic 4 DQ 3 asks you to plot: KL divergence over training, and
-- what its behaviour implies about stability.
create or replace view kl_by_update as
select
    e.algorithm,
    e.env_id,
    u.update_index,
    u.env_steps,
    count(*)                       as seeds,
    avg(u.kl_divergence)           as mean_kl,
    stddev_samp(u.kl_divergence)   as sd_kl,
    avg(u.clip_fraction)           as mean_clip_fraction,
    avg(u.policy_entropy)          as mean_policy_entropy
from policy_updates u
join experiments e on e.id = u.experiment_id
where u.kl_divergence is not null      -- PPO rows only; see the null note above
group by 1, 2, 3, 4;

-- The bake-off, binned by environment step so that A2C and PPO share an x-axis.
-- The bin width (1,000 steps) is written into the view rather than chosen in
-- the UI so that the number in your README and the number on the chart come
-- from the same definition.
create or replace view bakeoff_curve as
select
    e.algorithm,
    e.env_id,
    (ep.env_steps / 1000) * 1000    as step_bin,
    count(distinct e.seed)          as seeds,
    avg(ep."return")                as mean_return,
    stddev_samp(ep."return")        as sd_return
from episodes ep
join experiments e on e.id = ep.experiment_id
where ep.env_steps is not null
group by 1, 2, 3;

-- ---------------------------------------------------------------------------
-- Row-level security, matching 001_init.sql: the anon key the Streamlit tier
-- holds may read and may not write. A new table does NOT inherit the policies
-- of the tables around it, and forgetting these lines is how a product ships
-- with one table publicly writable.
-- ---------------------------------------------------------------------------
alter table policy_updates enable row level security;
alter table entropy_sweep  enable row level security;

do $$
begin
  if not exists (select 1 from pg_policies where policyname = 'anon_read_policy_updates') then
    create policy anon_read_policy_updates on policy_updates for select to anon using (true);
  end if;
  if not exists (select 1 from pg_policies where policyname = 'anon_read_entropy_sweep') then
    create policy anon_read_entropy_sweep on entropy_sweep for select to anon using (true);
  end if;
end $$;
