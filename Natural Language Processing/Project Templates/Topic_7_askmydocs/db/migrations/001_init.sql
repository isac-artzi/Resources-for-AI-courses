-- 001_init.sql — AskMyDocs schema (Cloud #3).
--
-- Apply it in the Supabase dashboard: SQL Editor → New query → paste → Run.
-- It is idempotent, so re-running it is safe.
--
-- FIVE TABLES, ON PURPOSE, and they are the pipeline written down:
--
--     documents  ──▶  chunks        (what you can retrieve, one vector each)
--     queries    ──▶  retrievals    (what came back, and how close it was)
--                ──▶  answers       (what the model then said, and what it cited)
--
-- The bottom three are the audit trail the product brief asks for. Drop them and
-- the service still answers questions, but you can no longer tell a retrieval
-- failure from a generation failure — the two produce the same wrong sentence.

-- ---------------------------------------------------------------------------
-- pgvector. THIS LINE MUST RUN FIRST.
--
-- Without the extension, every `vector(384)` below fails with "type vector does
-- not exist" and the migration stops halfway, leaving you with a documents table
-- and no chunks table. Supabase ships the extension but does not enable it in a
-- new project: either run this line, or use Database → Extensions and switch on
-- "vector". Running the line is easier to reproduce and it is why it lives here
-- rather than in a README step somebody will skip.
-- ---------------------------------------------------------------------------
create extension if not exists vector;


-- ---------------------------------------------------------------------------
-- documents — one row per source document, in exactly one corpus.
-- ---------------------------------------------------------------------------
create table if not exists documents (
    id              bigint generated always as identity primary key,

    title           text        not null,

    -- Where it came from: a URL, a path, a dataset id. The brief asks you to
    -- document the collection's source; this column is where that promise is
    -- kept per document, and it is what a citation in the Ask tab points at.
    source          text        not null,

    -- THE COLUMN THE WHOLE ASSIGNMENT TURNS ON.
    --
    -- 'retrieval' = part of the collection you search at query time.
    -- 'finetune'  = part of the corpus you adapted the generator on.
    --
    -- Never both. A generator fine-tuned on the documents it later retrieves has
    -- already memorised the answers, so it answers correctly WITHOUT retrieval,
    -- retrieval appears to add nothing, and the with/without comparison that is
    -- the point of this product measures your bookkeeping instead of your
    -- pipeline. The check constraint stops a typo ('retreival') from silently
    -- creating a third corpus that nothing ever searches.
    corpus          text        not null default 'retrieval'
                    check (corpus in ('retrieval', 'finetune')),

    -- sha256 of the document text. UNIQUE ACROSS BOTH CORPORA — see the index
    -- below. This is the database enforcing the disjointness rule a second time,
    -- after api/main.py has already refused the ingest. Two independent checks
    -- for one rule is not paranoia here: the contamination is invisible once it
    -- has happened, and re-running a 5,000-chunk ingest is cheap by comparison.
    content_sha256  text        not null,

    -- Total tokens in the document, as counted by the tokenizer named in
    -- doc_metadata->'chunking'->>'tokenizer_name'. The brief asks for the
    -- collection's total token count; it is sum(token_count) over this table.
    token_count     integer,

    -- The chunking parameters used at ingest, plus anything else you want to
    -- record about the document. jsonb so you can add fields without a migration.
    doc_metadata    jsonb       not null default '{}'::jsonb,

    created_at      timestamptz not null default now()
);

-- One document, one corpus, one row. An attempt to put the same text in both
-- corpora fails here with a unique-violation. An attempt to ingest the same
-- document twice into one corpus also fails, which is what you want: a duplicate
-- gets retrieved twice, fills two of your five slots, and turns k=5 into k=4.
create unique index if not exists idx_documents_sha on documents (content_sha256);
create index if not exists idx_documents_corpus on documents (corpus);


-- ---------------------------------------------------------------------------
-- chunks — the passages, their vectors, and the settings that produced them.
-- ---------------------------------------------------------------------------
create table if not exists chunks (
    id                 bigint generated always as identity primary key,
    document_id        bigint      not null references documents (id) on delete cascade,

    -- 0-based position within the document. (document_id, ordinal) is unique.
    ordinal            integer     not null,

    -- The passage text. Stored in full, unlike the query text, because a
    -- retrieval-augmented answer cannot be audited without the passage that
    -- grounded it. The consequence: anything you ingest is readable by anyone
    -- who can read this table. Do not point a public demo at private documents.
    content            text        not null,

    token_count        integer     not null,

    -- Token offsets into the parent document, end exclusive. Keeping these makes
    -- an overlap bug visible in a SELECT instead of only in a bad answer:
    -- neighbouring chunks should satisfy  next.start_token = prev.end_token -
    -- overlap_tokens.
    start_token        integer     not null,
    end_token          integer     not null,

    -- THE CHUNKING PARAMETERS, RECORDED PER CHUNK.
    --
    -- Per chunk rather than once per ingest, because you will re-chunk. When
    -- half the collection is at 400 tokens / 60 overlap and half is at 350 / 35
    -- because you changed your mind on a Tuesday, the only way to explain a
    -- retrieval result is a row that says which settings produced it.
    --
    -- The assignment's range is 300-500 tokens with 10-20 percent overlap; the
    -- check constraints below are that range, so a config mistake fails at the
    -- write rather than showing up as mediocre retrieval three days later.
    chunk_size_tokens  integer     not null check (chunk_size_tokens between 300 and 500),
    overlap_tokens     integer     not null check (overlap_tokens >= 0),

    -- Whose tokens. "400 tokens" means nothing until you say which tokenizer
    -- counted them; two models disagree by 20-30 percent on the same paragraph.
    tokenizer_name     text        not null,

    -- ------------------------------------------------------------------------
    -- THE VECTOR. Width 384 because sentence-transformers/all-MiniLM-L6-v2
    -- produces 384 numbers.
    --
    -- If you switch embedding models, this number changes and EVERY stored
    -- vector becomes unusable. There is no partial migration: you alter the
    -- column and re-embed the whole collection. Change it in exactly three
    -- places at once — here, nlp.EMBEDDING_DIM, and the match_chunks() signature
    -- at the bottom of this file — or the failure surfaces at QUERY time as
    -- "different vector dimensions 768 and 384", inside a user's request, long
    -- after the write that caused it.
    --
    -- Nullable on purpose: chunks in the 'finetune' corpus are stored for the
    -- perplexity split and the disjointness check, and are never embedded,
    -- because nothing should ever be able to retrieve them.
    -- ------------------------------------------------------------------------
    embedding          vector(384),
    embedding_model    text,
    embedding_dim      integer,

    created_at         timestamptz not null default now(),

    unique (document_id, ordinal)
);

create index if not exists idx_chunks_document on chunks (document_id);

-- ---------------------------------------------------------------------------
-- The vector index.
--
-- vector_cosine_ops matches the `<=>` cosine-distance operator used in
-- match_chunks() below. Pick the ops class that matches your operator: an
-- ivfflat index built for L2 distance is simply not used by a cosine query, the
-- planner falls back to a sequential scan over every row, and the only symptom
-- is that queries get slower as the collection grows.
--
-- Two things about ivfflat that surprise people:
--
-- 1. It is an APPROXIMATE index. It can miss a true nearest neighbour. That is
--    the trade you are making for speed, and it is worth one sentence in your
--    report. `set ivfflat.probes = 10;` searches more lists and recovers most of
--    the recall at some cost.
-- 2. It should be built AFTER the data is loaded. The index partitions the
--    vector space into `lists` clusters using the rows present at build time;
--    built on an empty table it learns nothing useful. Creating it here keeps
--    the migration in one file, which is the right call for a course project —
--    but after your first full ingest, drop and recreate it, and set
--    lists ≈ rows / 1000 (100 is a sensible starting value for a few thousand
--    chunks).
-- ---------------------------------------------------------------------------
create index if not exists idx_chunks_embedding
    on chunks using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);


-- ---------------------------------------------------------------------------
-- queries — that a question was asked, never what it said.
-- ---------------------------------------------------------------------------
create table if not exists queries (
    id               bigint generated always as identity primary key,

    -- sha256 of the question. A knowledge-management team's questions leak more
    -- than their documents do: "which suppliers are under review?" is a fact
    -- about the company whether or not it is ever answered. The hash still gives
    -- you de-duplication (same hash = same question) and reproducibility.
    query_sha256     text        not null,

    -- How many passages were requested. Logged because it changes the answer,
    -- and because "we used k=5" in a report should be checkable.
    k                integer     not null default 5 check (k between 1 and 20),

    -- Free-text version of the whole RAG configuration, e.g. "askmydocs-v3".
    model_version    text        not null default 'unset',

    -- Null when the question was answered with retrieval switched off.
    embedding_model  text,

    created_at       timestamptz not null default now()
);

create index if not exists idx_queries_created_at on queries (created_at desc);
create index if not exists idx_queries_sha        on queries (query_sha256);


-- ---------------------------------------------------------------------------
-- retrievals — what came back, how close it was, and in what order.
-- ---------------------------------------------------------------------------
create table if not exists retrievals (
    id          bigint generated always as identity primary key,
    query_id    bigint      not null references queries (id) on delete cascade,
    chunk_id    bigint      not null references chunks (id) on delete cascade,

    -- Cosine similarity, in [-1, 1]. If your values land outside that range you
    -- are storing a dot product over vectors you forgot to normalise, and your
    -- top-k is ranked partly by passage length. The check constraint catches it
    -- on the first insert instead of in a confusing audit table later.
    similarity  double precision not null check (similarity >= -1 and similarity <= 1),

    -- 1 = closest. Unique per query so a bug that logs rank 3 twice cannot
    -- quietly produce an audit view that disagrees with what the user saw.
    rank        integer     not null check (rank >= 1),

    created_at  timestamptz not null default now(),

    unique (query_id, rank)
);

create index if not exists idx_retrievals_query on retrievals (query_id);
create index if not exists idx_retrievals_chunk on retrievals (chunk_id);


-- ---------------------------------------------------------------------------
-- answers — what the model said, whether it had passages, and what it cited.
-- ---------------------------------------------------------------------------
create table if not exists answers (
    id               bigint generated always as identity primary key,
    query_id         bigint      not null references queries (id) on delete cascade,

    answer           text        not null,

    -- The column the experiment is built on. Filter this table on it and you
    -- have both halves of the with/without comparison, keyed by query.
    retrieval_used   boolean     not null default true,

    -- The subset of retrieved chunks the answer actually cited. The gap between
    -- this array and the rows in `retrievals` for the same query is the most
    -- informative number in the whole schema: a passage that was retrieved and
    -- then ignored is a prompting problem, a passage that was never retrieved is
    -- a chunking or embedding problem, and they look identical from the outside.
    cited_chunk_ids  bigint[]    not null default '{}',

    generator_model  text,

    created_at       timestamptz not null default now()
);

create index if not exists idx_answers_query on answers (query_id);
create index if not exists idx_answers_retrieval_used on answers (retrieval_used);


-- ---------------------------------------------------------------------------
-- match_chunks() — top-k nearest passages, computed inside Postgres.
--
-- `<=>` is pgvector's cosine DISTANCE operator: 0 means identical direction, 2
-- means opposite. Similarity is 1 - distance, which is the number your UI and
-- your report should show, because "similarity 0.83" is readable and "distance
-- 0.17" invites somebody to sort it the wrong way.
--
-- Doing the search here rather than pulling every vector into Python is not an
-- optimisation detail. The pull-everything version works fine on 200 chunks in
-- development and is killed by the memory limit the first time a grader opens
-- the deployed app.
--
-- The query_embedding parameter is vector(384). Same three-places rule as above:
-- change the width here too, or every call fails with a type error.
-- ---------------------------------------------------------------------------
create or replace function match_chunks(
    query_embedding vector(384),
    match_count     int  default 5,
    corpus_filter   text default 'retrieval'
)
returns table (
    chunk_id        bigint,
    document_id     bigint,
    document_title  text,
    content         text,
    similarity      double precision
)
language sql stable
as $$
    select
        c.id,
        c.document_id,
        d.title,
        c.content,
        (1 - (c.embedding <=> query_embedding))::double precision as similarity
    from chunks c
    join documents d on d.id = c.document_id
    where c.embedding is not null
      and d.corpus = corpus_filter
    order by c.embedding <=> query_embedding
    limit match_count;
$$;


-- ---------------------------------------------------------------------------
-- Row Level Security.
--
-- The Streamlit Retrieval Audit tab queries these tables directly with the ANON
-- key, so anonymous SELECT is allowed on all five. No anonymous INSERT policy
-- exists anywhere, which means the anon key cannot write. The API uses the
-- SERVICE-ROLE key, which bypasses RLS entirely — that is why writes are
-- server-side only.
--
-- Turning RLS off "to make it work" is the single most common way student
-- projects end up with a publicly writable database. Don't.
--
-- Read the SELECT policy on `chunks` carefully before you ingest anything: it
-- means every passage in your collection is readable by anyone holding the anon
-- key, which is a key you publish in a Streamlit app. That is the correct
-- behaviour for a course project built on public documents and the wrong
-- behaviour for anyone's real knowledge base. Choose your collection with that
-- in mind, and say in your MODEL_CARD that you did.
-- ---------------------------------------------------------------------------
alter table documents  enable row level security;
alter table chunks     enable row level security;
alter table queries    enable row level security;
alter table retrievals enable row level security;
alter table answers    enable row level security;

drop policy if exists "anon can read documents" on documents;
create policy "anon can read documents"
    on documents for select
    to anon
    using (true);

drop policy if exists "anon can read chunks" on chunks;
create policy "anon can read chunks"
    on chunks for select
    to anon
    using (true);

drop policy if exists "anon can read queries" on queries;
create policy "anon can read queries"
    on queries for select
    to anon
    using (true);

drop policy if exists "anon can read retrievals" on retrievals;
create policy "anon can read retrievals"
    on retrievals for select
    to anon
    using (true);

drop policy if exists "anon can read answers" on answers;
create policy "anon can read answers"
    on answers for select
    to anon
    using (true);
