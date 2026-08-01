-- 001_init.sql — EntityFinder schema (Cloud #3).
--
-- Apply it in the Supabase dashboard: SQL Editor → New query → paste → Run.
-- It is idempotent, so re-running it is safe.
--
-- FOUR TABLES, AND THE SHAPE IS THE LESSON:
--
--   runs         one row per TRAINING run (CRF and transformer both land here)
--   extractions  one row per SERVED request
--   entities     one row per PREDICTED entity, child of an extraction
--   reviews      one row per HUMAN DECISION, child of an entity
--
-- Read that as a chain. A run produces a model; a model produces extractions;
-- an extraction produces entities; a low-confidence entity produces a review.
-- Every link is append-only, so at any point you can ask "what did version 3
-- say about this span, and what did the human say back?" — which is the only
-- way to know whether the model is getting better or just getting different.
--
-- Notice what is NOT here: a column for the document text.

-- ---------------------------------------------------------------------------
-- runs — training, not serving
-- ---------------------------------------------------------------------------
create table if not exists runs (
    id              bigint generated always as identity primary key,

    -- 'crf' or 'transformer'. One table for both is what makes the comparison
    -- tab a single query instead of two spreadsheets you reconcile by hand.
    model_type      text        not null check (model_type in ('crf', 'transformer')),

    dataset         text        not null default 'conll2003',

    -- Hyperparameters, and for the CRF the feature list. This is the difference
    -- between "we got 0.83" and "we got 0.83 and here is how to get it again".
    config          jsonb       not null default '{}'::jsonb,

    -- ENTITY-level scores. Not token-level. If you put token-level accuracy in
    -- these columns, every chart in the UI becomes a lie that flatters you.
    precision       double precision,
    recall          double precision,
    f1              double precision,

    -- Full EntityScores payload, including the per-type breakdown. The
    -- aggregate hides which type you are actually bad at; keep both.
    metrics         jsonb,

    model_version   text        not null default 'unset',
    notes           text,
    created_at      timestamptz not null default now()
);

create index if not exists idx_runs_created_at on runs (created_at desc);
create index if not exists idx_runs_model_type on runs (model_type);

-- ---------------------------------------------------------------------------
-- extractions — one row per served request
-- ---------------------------------------------------------------------------
create table if not exists extractions (
    id              bigint generated always as identity primary key,

    -- sha256 of the document. Storing the hash instead of the text means an
    -- extraction can be identified and de-duplicated without holding a customer
    -- document in a course project's database. If two rows share a hash, they
    -- had identical input — which is how you show a version changed the output
    -- and the input did not.
    text_sha256     text        not null,

    model           text        not null check (model in ('crf', 'transformer')),
    model_version   text        not null default 'unset',
    entity_count    integer     not null default 0,
    latency_ms      integer,
    created_at      timestamptz not null default now()
);

create index if not exists idx_extractions_created_at on extractions (created_at desc);
create index if not exists idx_extractions_sha        on extractions (text_sha256);

-- ---------------------------------------------------------------------------
-- entities — one row per predicted entity. APPEND ONLY.
-- ---------------------------------------------------------------------------
create table if not exists entities (
    id              bigint generated always as identity primary key,
    extraction_id   bigint      not null references extractions (id) on delete cascade,

    -- The surface string. Stored, unlike the document, because it IS the
    -- prediction and a reviewer cannot review a span with no text in it.
    text            text        not null,

    -- Character offsets into the original document. The document is gone, so
    -- these are only meaningful together with the surface string and the hash —
    -- which is exactly why the offsets have to be right when they are written.
    start_char      integer     not null check (start_char >= 0),
    end_char        integer     not null check (end_char > start_char),

    entity_type     text        not null,

    -- The model's score for this span, in [0, 1]. The review queue is a query
    -- over this column, so a placeholder value here empties the queue.
    confidence      double precision not null check (confidence >= 0 and confidence <= 1),

    -- OPTIONAL surrounding snippet, for reviewers. Default null, on purpose:
    -- turning it on puts raw document text back in the database, which is the
    -- thing hashing was meant to avoid. If your reviewers genuinely cannot
    -- decide without context, enable it, keep the window small, and write the
    -- tradeoff down in the MODEL_CARD instead of letting it happen quietly.
    context         text,

    created_at      timestamptz not null default now()
);

create index if not exists idx_entities_extraction on entities (extraction_id);
create index if not exists idx_entities_confidence on entities (confidence);
create index if not exists idx_entities_type       on entities (entity_type);

-- ---------------------------------------------------------------------------
-- reviews — one row per human decision. ALSO APPEND ONLY.
--
-- Every review carries a SNAPSHOT of the prediction it is ruling on:
-- original_type, original_start_char, original_end_char, original_confidence.
-- The duplication is deliberate. It is what lets you answer "what did the model
-- say before the human touched it?" with a single select, and it survives even
-- if a later migration changes how entities are stored.
--
-- There is no UPDATE anywhere in this design. If a reviewer changes their mind,
-- they add another review; the newest one wins and the earlier one is still
-- there. Correcting the entities row in place would destroy both the audit
-- trail (was this the model or a human?) and the training signal (the pair of
-- what-it-said and what-was-true is precisely what you would fine-tune on).
-- ---------------------------------------------------------------------------
create table if not exists reviews (
    id                    bigint generated always as identity primary key,
    entity_id             bigint      not null references entities (id) on delete cascade,

    reviewer_id           text        not null,

    -- accept: prediction stands. correct: real entity, wrong type or boundary.
    -- reject: no entity here at all. Keeping 'reject' separate from 'correct'
    -- is what distinguishes "you mislabelled something real" from "you invented
    -- an entity", and those two failures have different fixes.
    decision              text        not null check (decision in ('accept', 'correct', 'reject')),

    original_type         text        not null,
    original_start_char   integer     not null,
    original_end_char     integer     not null,
    original_confidence   double precision not null,

    corrected_type        text,
    corrected_start_char  integer,
    corrected_end_char    integer,

    note                  text,
    created_at            timestamptz not null default now()
);

create index if not exists idx_reviews_entity     on reviews (entity_id);
create index if not exists idx_reviews_created_at on reviews (created_at desc);
create index if not exists idx_reviews_decision   on reviews (decision);

-- ---------------------------------------------------------------------------
-- Row Level Security.
--
-- The Streamlit comparison tab reads `runs` directly with the ANON key, so
-- anonymous SELECT is allowed there and nowhere else. extractions, entities and
-- reviews carry per-document detail, so the browser does not get to read them
-- and certainly does not get to write them: the review queue goes through the
-- API, which holds the SERVICE-ROLE key and bypasses RLS server-side.
--
-- That is why POST /review exists at all. A UI that wrote to Postgres directly
-- would need an anon INSERT policy, and an anon INSERT policy means anyone with
-- your public key can fabricate reviewer decisions in your audit trail.
--
-- Turning RLS off "to make it work" is the single most common way student
-- projects end up with a publicly writable database. Don't.
-- ---------------------------------------------------------------------------
alter table runs        enable row level security;
alter table extractions enable row level security;
alter table entities    enable row level security;
alter table reviews     enable row level security;

drop policy if exists "anon can read runs" on runs;
create policy "anon can read runs"
    on runs for select
    to anon
    using (true);
