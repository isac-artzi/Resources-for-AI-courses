-- ===========================================================================
-- 002_topic6.sql — the three tables this product is actually about.
--
-- 001_init.sql answers "did the agent learn": one row per episode, one return
-- per row. Alignment does not fit that grain at all. The unit of evidence here
-- is a *comparison* (which of two responses a human preferred), a *completion*
-- (what a specific model variant said to a specific prompt), and an
-- *alignment run* (what happened at one value of beta). None of those is an
-- episode, and forcing them into `episodes` would make every query in the
-- README a string-parsing exercise.
--
-- Apply AFTER 001_init.sql. Migrations are ordered by filename and the foreign
-- keys below will not resolve otherwise. Never edit 001 to add these tables: a
-- migration already applied to a live project is history, and rewriting
-- history is how a schema and a database stop agreeing.
--
-- One decision worth stating before you read the DDL: THIS SCHEMA STORES USER
-- TEXT IN EXACTLY ONE PLACE — `completions.text`, which holds text your own
-- training tier generated, and `preferences`, which holds text from a public
-- dataset. It never stores text submitted to the running service. `/score`
-- writes a SHA-256 digest to `audit_log.state_hash` and nothing else. See the
-- comment on `audit_log` in 001_init.sql; in this topic the "state" is user
-- text and that comment stops being hypothetical.
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- The preference dataset, after binarisation --------------------------------
-- ---------------------------------------------------------------------------
-- Grain: one row per COMPARISON, not per response. A comparison is the atom of
-- the Bradley-Terry likelihood — the model is identified only up to a constant
-- per prompt, so a row holding a single response with an absolute score would
-- be storing a number the data never contained.
create table if not exists preferences (
    id          bigserial primary key,

    -- `prompt_id` is a stable key you assign, not a surrogate. The completions
    -- table joins to it, and a bigserial that changes every time you re-import
    -- the dataset would silently re-point every completion at a different
    -- prompt. Use a hash of the prompt text or the upstream dataset index.
    prompt_id   text        not null,
    prompt      text        not null,
    chosen      text        not null,
    rejected    text        not null,

    -- 'train' or 'test'. Stored rather than recomputed because the held-out
    -- pairwise accuracy in your report is only meaningful if the split is the
    -- same one the model was fit against, and a split recomputed from a hash
    -- after you changed the hash is a different split.
    split       text        not null check (split in ('train', 'test')),

    -- Length in characters, denormalised on purpose. The length-bias
    -- regression is a required result of this product and it is a query
    -- against these two columns; computing char_length() over 2,000 rows of
    -- text on every load is exactly the kind of thing the free tier notices.
    chosen_len   integer,
    rejected_len integer,

    source      text,                        -- e.g. 'trl-lib/ultrafeedback_binarized'
                                             -- or 'synthetic-offline'. Never leave
                                             -- this null: a table mixing real and
                                             -- fallback data with no way to tell
                                             -- them apart is worse than no table.
    created_at  timestamptz not null default now(),

    -- One row per (prompt, split, chosen, rejected). Makes a re-import that
    -- crashes halfway safe to restart: the retry conflicts instead of quietly
    -- doubling the dataset and halving every accuracy you compute.
    unique (prompt_id, split, chosen, rejected)
);

-- The two queries this table serves: "give me the training split" (fitting)
-- and "give me the test split" (the accuracy in the report). Both are
-- covered by one index on split, and the prompt_id index is what makes the
-- join from `completions` cheap.
create index if not exists preferences_split_idx     on preferences (split);
create index if not exists preferences_prompt_id_idx on preferences (prompt_id);

-- ---------------------------------------------------------------------------
-- Offline generation: what each model variant actually said -----------------
-- ---------------------------------------------------------------------------
-- This table is the reason the deployed service can be a SCORER rather than a
-- GENERATOR. Generation happens once, in the training tier, on a machine that
-- can afford a language model; the service reads rows. See the architecture
-- note in the README — this is the design, not a workaround.
create table if not exists completions (
    id            bigserial primary key,

    prompt_id     text        not null,      -- joins to preferences.prompt_id
    prompt        text        not null,      -- denormalised so the "Base vs Aligned"
                                             -- tab is one query, not two

    -- 'base' for the reference policy, 'dpo' (or 'sft', 'grpo', ...) for an
    -- aligned variant. Kept as text rather than an enum because you will add a
    -- variant and an enum migration is a worse afternoon than a check
    -- constraint you forgot to widen.
    model_variant text        not null,

    -- NULL for the base model, which has no KL constraint because it IS the
    -- reference. Null rather than 0.0: a zero would average into
    -- `avg(beta)` and make the base model look like an infinitely
    -- unconstrained aligned one.
    beta          double precision,

    text          text        not null,
    reward_score  double precision,          -- assigned by the DEPLOYED head, via
                                             -- POST /score, so the number in your
                                             -- report is the number the service
                                             -- would return today

    -- The latent quality the synthetic generator used to label preferences, or
    -- your own hand rating on the real path. This column is what makes the
    -- reward-hacking chart possible: without a series the reward model never
    -- saw, "the proxy went up" is not evidence that anything decoupled.
    true_quality  double precision,

    tokens        integer,                   -- completion length; the x-axis of the
                                             -- length-bias plot for generated text
    run_id        uuid references experiments(id) on delete set null,
    created_at    timestamptz not null default now(),

    unique (prompt_id, model_variant, beta, text)
);

-- The "Base vs Aligned" tab asks for one prompt across every variant, and the
-- reward-hacking aggregate asks for one variant across every prompt. Two
-- indexes, because one composite index cannot serve both leading columns.
create index if not exists completions_prompt_idx  on completions (prompt_id, model_variant);
create index if not exists completions_variant_idx on completions (model_variant, beta);

-- ---------------------------------------------------------------------------
-- One row per DPO run ------------------------------------------------------
-- ---------------------------------------------------------------------------
-- Grain: one row per (experiment, beta). Not per training step — the four
-- quantities below are end-of-run summaries and logging them per step would
-- make `select * from alignment_runs` a chart instead of a table. If you want
-- the per-step curve, write it to `episodes` with the step as episode_index;
-- that is what that table is for and the run-history view already aggregates it.
create table if not exists alignment_runs (
    id            uuid primary key default gen_random_uuid(),
    experiment_id uuid references experiments(id) on delete cascade,

    beta          double precision not null,  -- the KL coefficient. THE independent
                                              -- variable of this whole product.
    final_loss    double precision,

    -- The three DPO diagnostics, named as TRL names them so your numbers are
    -- comparable with anyone else's:
    --   implicit reward = beta * log(pi_theta(y|x) / pi_ref(y|x))
    --   margin          = mean over pairs of (implicit_chosen - implicit_rejected)
    --   accuracy        = fraction of pairs with margin > 0
    -- The margin is in the reward model's units only by analogy; it is a
    -- log-ratio scaled by beta, so margins at different beta are NOT directly
    -- comparable. Say that in your report rather than plotting them together
    -- as though they were.
    implicit_reward_margin   double precision,
    implicit_reward_accuracy double precision,

    -- Mean KL(pi_theta || pi_ref) over held-out prompts, in nats, summed over
    -- the generated sequence. The quantity beta is supposed to be buying you.
    kl_from_reference        double precision,

    -- Mean score assigned by the deployed reward head to this variant's
    -- completions. Plotted against kl_from_reference, this is the
    -- reward-hacking chart.
    mean_reward_model_score  double precision,

    -- The series the reward model never saw. Optional on the real path (it is
    -- your hand ratings); required on the synthetic path, where it is exact.
    mean_true_quality        double precision,

    steps         integer,
    seed          integer,
    notes         text,
    created_at    timestamptz not null default now()
);
create index if not exists alignment_runs_beta_idx on alignment_runs (beta);

-- ---------------------------------------------------------------------------
-- The view the UI reads ----------------------------------------------------
-- ---------------------------------------------------------------------------
-- The "Reward Hacking" tab needs proxy reward and KL on the same axes against
-- beta, plus the base model as the leftmost point. Keeping the aggregation in
-- SQL rather than in pandas is the point of having a data tier: the claim
-- "the proxy kept rising after quality stopped" is then a query a reader can
-- run against your project, not a number in your README.
--
-- Note the `coalesce(beta, 'Infinity')`: the base model is the beta -> infinity
-- limit of the KL-regularised objective (infinite penalty on any deviation, so
-- pi = pi_ref). Sorting by it puts the base model at the correct end of the
-- x-axis for free, instead of at the origin where a 0.0 would have put it.
create or replace view alignment_by_beta as
select
    coalesce(c.beta, 'Infinity'::double precision) as beta,
    c.model_variant,
    count(*)                                       as completions,
    avg(c.reward_score)                            as mean_reward_score,
    stddev_samp(c.reward_score)                    as sd_reward_score,
    avg(c.true_quality)                            as mean_true_quality,
    avg(c.tokens)                                  as mean_tokens,
    max(r.kl_from_reference)                       as kl_from_reference,
    max(r.implicit_reward_accuracy)                as implicit_reward_accuracy,
    max(r.implicit_reward_margin)                  as implicit_reward_margin
from completions c
left join alignment_runs r
       on r.beta = c.beta
group by 1, 2
order by 1;

-- The length-bias evidence, as a view rather than as a notebook cell. Pearson
-- correlation between assigned reward and completion length, per variant.
-- corr() is a Postgres aggregate; it returns null for fewer than two rows,
-- which is the honest answer rather than a spurious 0.
create or replace view reward_length_correlation as
select
    model_variant,
    count(*)                        as n,
    corr(reward_score, tokens)      as reward_vs_length_r,
    avg(tokens)                     as mean_tokens
from completions
where reward_score is not null and tokens is not null
group by model_variant;

-- ---------------------------------------------------------------------------
-- Row-level security -------------------------------------------------------
-- ---------------------------------------------------------------------------
-- A new table does NOT inherit the policies of the tables around it, and
-- forgetting these lines is how a product ships with one table publicly
-- writable. The anon key the Streamlit tier holds may read and may not write.
alter table preferences    enable row level security;
alter table completions    enable row level security;
alter table alignment_runs enable row level security;

do $$
begin
  if not exists (select 1 from pg_policies where policyname = 'anon_read_preferences') then
    create policy anon_read_preferences    on preferences    for select to anon using (true);
    create policy anon_read_completions    on completions    for select to anon using (true);
    create policy anon_read_alignment_runs on alignment_runs for select to anon using (true);
  end if;
end $$;
