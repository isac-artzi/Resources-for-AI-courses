-- ===========================================================================
-- 002_topic2.sql — the Policy Lab layer on top of the standing schema.
--
-- Migrations are additive and numbered, and this one is a good illustration of
-- why. Everything in 001_init.sql still means what it meant; this file adds
-- three columns, two indexes and two views. A reviewer who has read 001 only
-- has to read the diff, and a project that has already applied 001 does not
-- have to be rebuilt. Editing 001 in place instead — the tempting shortcut —
-- makes it impossible to tell which schema produced the rows already in the
-- table.
--
-- Apply AFTER 001_init.sql. Safe to run twice.
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- episodes: value iteration logs a SWEEP per row, not an episode.
-- ---------------------------------------------------------------------------
-- The planner has no episodes. It has sweeps, and the number that matters for
-- each one is the Bellman residual max_s |V_{k+1}(s) - V_k(s)|, which bounds
-- the remaining distance to V* by residual * gamma / (1 - gamma).
--
-- It gets its own column rather than borrowing `return` or `epsilon`. Reusing
-- an existing column is how a schema stops being self-describing: six weeks
-- later `return` would mean "episode return, unless algorithm = value_iteration
-- in which case it means residual", and that sentence has to live somewhere —
-- either in a column name or in tribal knowledge.
alter table episodes add column if not exists bellman_residual double precision;

-- Partial index: only planner rows have a residual, and on a table where the
-- learner writes hundreds of thousands of rows a partial index is a few
-- kilobytes rather than a few megabytes. Free-tier storage is 500 MB total.
create index if not exists episodes_residual_idx
    on episodes (experiment_id, episode_index)
    where bellman_residual is not null;

-- ---------------------------------------------------------------------------
-- evaluations: this topic evaluates an ESTIMATOR, not only a policy.
-- ---------------------------------------------------------------------------
-- The standing table answers "how much return did the greedy policy collect?"
-- This topic also has to answer "how far is the learner's value function from
-- the exact one, after n episodes?" — a different quantity with different
-- units, so it gets its own column and a discriminator saying which question
-- the row answers.
alter table evaluations add column if not exists metric text not null default 'return';
alter table evaluations add column if not exists rmse double precision;
alter table evaluations add column if not exists policy_source text;

comment on column evaluations.metric is
    'Which question this row answers: ''return'' (greedy evaluation sweep, the '
    'standing meaning) or ''value_rmse'' (distance from the exact solution).';
comment on column evaluations.policy_source is
    'value_iteration | monte_carlo — matches the policy_source accepted by POST /act.';

-- `at_training_episode` carries the EPISODE BUDGET for a value_rmse row, which
-- is exactly what that column already means. Indexing on it makes the
-- convergence view a range scan instead of a sequential scan; the syllabus
-- asks for an index here because this table gets a row per seed per budget and
-- the Streamlit Convergence tab queries it on every page load.
create index if not exists evaluations_metric_budget_idx
    on evaluations (metric, at_training_episode);

-- ---------------------------------------------------------------------------
-- The aggregate view the Convergence tab reads.
-- ---------------------------------------------------------------------------
-- Aggregating in SQL rather than pulling every row into Streamlit is what
-- keeps this page cheap on the free tier, and it also puts the definition of
-- "mean RMSE at budget n" in ONE place that both the UI and any reviewer with
-- psql can see.
--
-- The half-width uses 1.96 rather than a t quantile because Postgres has no
-- t-distribution built in. With 10 seeds that understates the interval by
-- about 15% (t_{0.975,9} = 2.262), so this view is the CHART, and the numbers
-- quoted in the README come from train/compare.py, which uses the correct
-- quantile. Where a normal-approximation interval appears in a UI it should
-- say so — this one does, in the column name.
create or replace view mc_convergence as
select
    e.at_training_episode                    as episodes,
    count(*)                                 as seeds,
    avg(e.rmse)                              as mean_rmse,
    stddev_samp(e.rmse)                      as sd_rmse,
    avg(e.rmse) - 1.96 * stddev_samp(e.rmse) / sqrt(count(*)) as normal_ci_low,
    avg(e.rmse) + 1.96 * stddev_samp(e.rmse) / sqrt(count(*)) as normal_ci_high,
    min(e.created_at)                        as first_written
from evaluations e
where e.metric = 'value_rmse'
group by e.at_training_episode
order by e.at_training_episode;

-- The convergence curve of the PLANNER, for the same chart. Two algorithms,
-- one schema — which is the entire argument of this product, expressed as a
-- pair of views over the same two tables.
create or replace view vi_convergence as
select
    ep.experiment_id,
    ep.episode_index                          as sweep,
    ep.bellman_residual,
    ep."return"                               as v_start_state
from episodes ep
join experiments x on x.id = ep.experiment_id
where ep.bellman_residual is not null
  and x.algorithm = 'value_iteration'
order by ep.experiment_id, ep.episode_index;

-- The anon key must be able to read the new views for the read-only Streamlit
-- tabs. Views inherit the row-level security of their base tables in Postgres
-- 15+ when created with security_invoker; on a free-tier project created
-- earlier they run as the owner, which is why the base tables' anon SELECT
-- policies in 001_init.sql are what actually matter here. Granting explicitly
-- so the intent survives either behaviour:
grant select on mc_convergence to anon;
grant select on vi_convergence to anon;
