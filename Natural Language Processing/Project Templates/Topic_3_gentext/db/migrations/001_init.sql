-- 001_init.sql — GenText schema (Cloud #3).
--
-- Apply it in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
-- It is idempotent, so re-running it is safe.
--
-- TWO TABLES, AND THE REASON MATTERS.
--
-- training_runs answers "how were these weights made?" — base model,
-- hyperparameters, corpus, held-out perplexity. You write a row three or four
-- times over the whole project, from your offline training script.
--
-- generations answers "what did the service say, under what settings, and how
-- good was it?" You write a row every time anyone presses Generate.
--
-- Keep only training_runs and the History tab has three rows in it and no way to
-- show a single output or a single rating. Keep only generations and you can
-- see every output but cannot say which fine-tune produced it. The join key
-- between them is model_version, which is why that string appears in both
-- tables, in /version, and in the MODEL_VERSION environment variable.


-- ---------------------------------------------------------------------------
-- 1. Fine-tuning runs.
-- ---------------------------------------------------------------------------
create table if not exists training_runs (
    id                    bigint generated always as identity primary key,

    -- Hugging Face id of the decoder you started from, e.g. "gpt2".
    base_model            text        not null,

    -- The label stamped on every generation made with the resulting weights.
    -- This is the join key. Make it specific: "gentext-gpt2-reviews-v2", not "v2".
    model_version         text        not null,

    -- Learning rate, epochs, batch size, block size, method (full / lora / ...).
    -- JSON rather than columns because you will add a hyperparameter you did not
    -- plan for, and a migration in week three is time you do not have.
    hyperparameters       jsonb       not null default '{}'::jsonb,

    -- Where the corpus came from and what it hashed to AFTER filtering. Without
    -- the hash, "trained on the reviews corpus" identifies nothing six weeks on.
    corpus_source         text,
    corpus_sha256         text,
    corpus_sentence_count integer,

    -- Perplexity on the held-out split. The number that says whether the
    -- fine-tune helped. Record the base model's score too, as a separate row
    -- with method "frozen", so you have something to compare against.
    held_out_perplexity   double precision,

    notes                 text,
    created_at            timestamptz not null default now()
);

create index if not exists idx_training_runs_created_at on training_runs (created_at desc);
create index if not exists idx_training_runs_version    on training_runs (model_version);


-- ---------------------------------------------------------------------------
-- 2. Individual generations. This is what the History tab reads.
-- ---------------------------------------------------------------------------
create table if not exists generations (
    id                    bigint generated always as identity primary key,

    -- sha256 of the prompt. NOT the prompt. The product brief promises hashed
    -- prompts, and this is where that promise is either kept or quietly broken.
    -- Two generations sharing a hash had identical prompts, which is enough to
    -- group them in the Compare view without holding anyone's text.
    prompt_sha256         text        not null,

    -- The decoding strategy as an explicit value, so you can group by it. These
    -- five are different algorithms, not five settings of one algorithm.
    strategy              text        not null
        check (strategy in ('greedy', 'beam', 'temperature', 'top_k', 'top_p')),

    -- Every knob as it was at generation time: temperature, top_k, top_p,
    -- num_beams, max_new_tokens, repetition_penalty, seed. This plus the prompt
    -- hash plus model_version is what makes a generation reproducible.
    decoding_params       jsonb       not null default '{}'::jsonb,

    -- The OUTPUT is stored in full. It is the thing being evaluated; a rating
    -- attached to text nobody can read is not evidence of anything.
    generated_text        text        not null,

    prompt_token_count    integer,
    generated_token_count integer,

    -- Automatic signals. distinct_2 is the repetition detector; perplexity is
    -- the likelihood signal. Neither is a quality score on its own.
    distinct_1            double precision,
    distinct_2            double precision,
    perplexity            double precision,
    latency_ms            double precision,

    -- Joins to training_runs.model_version.
    model_version         text        not null default 'unset',

    -- The aggregate rating (rounded mean of everything in `ratings`), so the
    -- table can be sorted and filtered on it.
    human_rating          integer     check (human_rating between 1 and 5),

    -- Every INDIVIDUAL rating, each with its rater_id, its per-dimension scores
    -- and its notes:
    --   [{"rater_id": "rater-a", "rating": 4, "dimensions": {...},
    --     "notes": "...", "recorded_at": "..."}, ...]
    --
    -- The assignment asks for two independent raters and a report on where they
    -- disagreed. Averaging on the way in destroys exactly that. Keep both.
    ratings               jsonb       not null default '[]'::jsonb,

    created_at            timestamptz not null default now()
);

create index if not exists idx_generations_created_at on generations (created_at desc);
create index if not exists idx_generations_strategy   on generations (strategy);
create index if not exists idx_generations_version    on generations (model_version);
create index if not exists idx_generations_prompt     on generations (prompt_sha256);

-- Unrated outputs, oldest first — the work queue for your rating session.
-- security_invoker makes the view respect the caller's row-level security
-- instead of the view owner's. Without it a view is a hole straight through the
-- policies below, which is a mistake worth knowing about before you make it
-- somewhere that matters.
create or replace view unrated_generations
    with (security_invoker = on) as
    select id, strategy, model_version, generated_text, created_at
    from generations
    where human_rating is null
    order by created_at asc;


-- ---------------------------------------------------------------------------
-- Row Level Security.
--
-- The Streamlit History tab queries these tables directly with the ANON key, so
-- anonymous SELECT is allowed. No anonymous INSERT or UPDATE policy exists,
-- which means the anon key cannot write — including cannot write a rating. The
-- rating form in the UI therefore posts to the API, which holds the SERVICE-ROLE
-- key and bypasses RLS. That round trip is the point, not an inconvenience: a
-- browser that can write ratings directly is a browser anyone can use to write
-- five stars four hundred times.
--
-- Turning RLS off "to make it work" is the single most common way student
-- projects end up with a publicly writable database. Don't.
-- ---------------------------------------------------------------------------
alter table generations   enable row level security;
alter table training_runs enable row level security;

drop policy if exists "anon can read generations" on generations;
create policy "anon can read generations"
    on generations for select
    to anon
    using (true);

drop policy if exists "anon can read training runs" on training_runs;
create policy "anon can read training runs"
    on training_runs for select
    to anon
    using (true);
