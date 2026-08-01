-- ===========================================================================
-- 002_topic5.sql — the Search Arena layer on top of the standing schema.
--
-- Migrations are additive and numbered, and this one shows why. Everything in
-- 001_init.sql still means what it meant; this file adds three tables, five
-- indexes and two views. A reviewer who has read 001 only has to read the
-- diff, and a project that has already applied 001 — the one you used for your
-- earlier products — does not have to be rebuilt. Editing 001 in place instead
-- makes it impossible to say which schema produced the rows already in the
-- table.
--
-- Apply AFTER 001_init.sql. Safe to run twice.
--
-- The standard this schema is built to: EVERY NUMBER IN THE WRITE-UP MUST BE A
-- QUERY AGAINST `games`. Not a printout, not a number copied out of a terminal
-- into a markdown table. If a claim in your report cannot be expressed as SQL
-- over these tables, it is not yet evidence, and the usual fix is that
-- something the harness measured was never written down.
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- games — one row per game, the atom of every claim in this product.
-- ---------------------------------------------------------------------------
-- One row per game, NOT two. It is tempting to log each game twice, once from
-- each side, so that "select where agent = X" needs no thought. Resist it: the
-- row count then double-counts every game, and every aggregate is silently
-- wrong by a factor of two unless the query remembers to halve it. The
-- `win_rate_matrix` view below reads the single row from both directions, which
-- puts that subtlety in exactly one reviewable place.
--
-- `result` is ALWAYS from `agent`'s point of view. Saying so in a comment is
-- not enough — the check constraint is what stops a future insert from putting
-- 'agent_a' in this column because that is what the API response called it.
create table if not exists games (
    game_id       uuid primary key default gen_random_uuid(),
    experiment_id uuid references experiments(id) on delete cascade,
    agent         text        not null,
    opponent      text        not null,
    result        text        not null check (result in ('win', 'loss', 'draw')),

    -- Connect Four is a first-player win under perfect play, and even between
    -- ordinary agents moving first is worth several points of win rate. A
    -- tournament that does not record who moved first cannot distinguish a
    -- stronger agent from a luckier draw, so this column is NOT NULL.
    agent_played_first boolean not null,

    moves         integer     not null check (moves between 0 and 42),

    -- The cost columns, IN PAIRS. `nodes_expanded` is `agent`'s decisions;
    -- `opponent_nodes_expanded` is the other side's. Summing them would make
    -- every row a statement about the pairing rather than about either agent,
    -- and the pairing is what `matches` is for.
    --
    -- Why both halves rather than just `agent`'s: the win-rate matrix reads
    -- each game row from BOTH directions (see the view below), so without the
    -- opponent's costs every cell in one triangle of the cost table is empty.
    -- The first version of this schema stored only `agent`'s half and the
    -- matrix looked fine while the cost table was half zeros.
    --
    -- `peak_kib` is peak Python allocation during a SINGLE DECISION, maximised
    -- over that agent's decisions in this game — not the whole game, which
    -- would attribute an expensive opponent's tree to a cheap agent.
    nodes_expanded          integer          not null,
    wall_clock_ms           double precision not null,
    search_depth            integer          not null,
    peak_kib                double precision,
    opponent_nodes_expanded integer,
    opponent_wall_clock_ms  double precision,
    opponent_search_depth   integer,
    opponent_peak_kib       double precision,

    -- Budget provenance. A tournament in which decisions were truncated is a
    -- tournament of budget-limited agents — a different experiment from the one
    -- you meant to run. Recorded per game so that "were any of these numbers
    -- produced under truncation?" is a query rather than a memory.
    node_budget            integer,
    budget_exhausted_moves integer not null default 0,

    seed          integer,
    created_at    timestamptz not null default now()
);

-- The three access patterns, and nothing speculative. Every index costs write
-- throughput and free-tier storage; add one when a query is slow, not when a
-- column looks important.
create index if not exists games_pairing_idx    on games (agent, opponent);
create index if not exists games_experiment_idx on games (experiment_id, created_at desc);
-- Partial: only the truncated games matter for this question, and there should
-- be very few of them. A partial index over a rare predicate is kilobytes.
create index if not exists games_truncated_idx  on games (experiment_id)
    where budget_exhausted_moves > 0;

-- ---------------------------------------------------------------------------
-- matches — the head-to-head aggregate.
-- ---------------------------------------------------------------------------
-- A materialised aggregate rather than "just query games every time", for one
-- reason: the Streamlit Tournament tab renders on every page load, on a free
-- tier, against a table that has thousands of rows and will have tens of
-- thousands by the end of the course. Aggregating 4,500 rows per page view is
-- how a demo becomes a demo that times out in front of the person you built it
-- for.
--
-- The cost of that decision, stated plainly: `matches` can go stale. The unique
-- constraint plus the upsert in shared/store.py is what keeps a re-run
-- CORRECTING the row rather than adding a second one, and the
-- `win_rate_matrix` view below recomputes the same numbers from `games` so the
-- two can be compared whenever you doubt them.
create table if not exists matches (
    id            bigserial primary key,
    experiment_id uuid references experiments(id) on delete cascade,
    agent         text    not null,
    opponent      text    not null,
    games         integer not null check (games >= 0),
    wins          integer not null,
    draws         integer not null,
    losses        integer not null,
    win_rate      double precision not null check (win_rate between 0 and 1),
    mean_nodes    double precision not null,
    mean_ms       double precision not null,
    mean_peak_kib double precision,
    created_at    timestamptz not null default now(),

    -- The upsert target. Without this, `train/benchmark.py` run twice leaves two
    -- rows per pairing and every downstream AVG() mixes an old experiment into
    -- a new one.
    unique (experiment_id, agent, opponent),

    -- Cheap arithmetic invariant. It has caught a real bug in this harness
    -- (draws counted into `wins` when both sides filled the board), and a
    -- constraint that fires at insert time is worth more than a chart nobody
    -- looked at closely.
    check (wins + draws + losses = games)
);
create index if not exists matches_agent_idx on matches (agent, opponent);

-- ---------------------------------------------------------------------------
-- search_probes — the scalability sweep.
-- ---------------------------------------------------------------------------
-- Its own table rather than `games` rows with NULLs. A probe is a single search
-- from a single position: it has no opponent, no result and no move count, so
-- storing it in `games` would leave half the columns NULL for half the rows and
-- force every query on that table to know which kind of row it was looking at.
-- Two small honest tables beat one table with a discriminator, when the two
-- things genuinely are different things.
create table if not exists search_probes (
    id             bigserial primary key,
    experiment_id  uuid references experiments(id) on delete cascade,
    variant        text    not null,      -- 'exhaustive' | 'natural+ab' | 'heuristic+ab' | 'beam3'
    depth          integer not null check (depth between 1 and 20),
    nodes          integer not null,
    leaves         integer not null default 0,
    cutoffs        integer not null default 0,
    wall_clock_ms  double precision not null,
    peak_kib       double precision,
    completed      boolean not null default true,

    -- Node counts are meaningless without the position they were measured from:
    -- the empty board and a crowded midgame differ by an order of magnitude at
    -- the same depth. NOT NULL with a default, so a probe cannot be logged
    -- anonymously.
    position_label text    not null default 'empty',
    created_at     timestamptz not null default now()
);
create index if not exists probes_variant_depth_idx on search_probes (variant, depth);

-- ---------------------------------------------------------------------------
-- The win-rate matrix. This is the view the "Tournament" tab renders.
-- ---------------------------------------------------------------------------
-- Recomputed from `games` rather than read from `matches`, deliberately. Two
-- routes to the same number is not duplication here, it is the check: if the
-- table and the view disagree, the harness wrote something the SQL does not
-- agree with, and you want to find that out from a diff rather than from a
-- viva.
--
-- The union is where the "one row per game" decision is paid for. Each game
-- contributes once as (agent -> opponent) and once, inverted, as
-- (opponent -> agent). Read the second branch carefully: 'win' becomes 'loss',
-- AND the cost columns switch to their `opponent_` counterparts. Both halves
-- of the swap have to happen together — inverting the result while keeping
-- `agent`'s node count would file the loser's cost under the winner's name.
create or replace view win_rate_matrix as
with directed as (
    select agent, opponent, result,
           nodes_expanded, wall_clock_ms, peak_kib, search_depth,
           experiment_id
    from games
    union all
    select opponent as agent,
           agent    as opponent,
           case result when 'win' then 'loss' when 'loss' then 'win' else 'draw' end,
           opponent_nodes_expanded, opponent_wall_clock_ms,
           opponent_peak_kib, opponent_search_depth,
           experiment_id
    from games
)
select
    agent,
    opponent,
    count(*)                                                  as games,
    count(*) filter (where result = 'win')                    as wins,
    count(*) filter (where result = 'draw')                   as draws,
    count(*) filter (where result = 'loss')                   as losses,
    -- A draw counts as half a win. State the scoring rule in the view rather
    -- than in a caption: "win rate" is ambiguous in a game with draws, and two
    -- charts using two definitions is a reporting error nobody notices.
    (count(*) filter (where result = 'win')
     + 0.5 * count(*) filter (where result = 'draw'))::double precision
        / nullif(count(*), 0)                                 as win_rate,
    avg(nodes_expanded)                                       as mean_nodes,
    avg(wall_clock_ms)                                        as mean_ms,
    avg(peak_kib)                                             as mean_peak_kib,
    max(search_depth)                                         as max_search_depth
from directed
group by agent, opponent;

-- Decision quality against the FIXED reference opponent, which is the number
-- the product brief actually asks for ("which agent is stronger"). Split out as
-- its own view because it is the one comparison that stays valid across weeks:
-- 'random' never improves, so a win rate against it in week 5 is comparable
-- with one in week 12. A win rate against "my previous agent" is a moving
-- target and cannot be plotted against anything.
create or replace view decision_quality as
select
    agent,
    games,
    win_rate,
    mean_nodes,
    mean_ms
from win_rate_matrix
where opponent = 'random'
order by win_rate desc, mean_nodes;

-- Row-level security, exactly as 001_init.sql does it: the anon key the
-- Streamlit read-only tabs use must be able to read and must not be able to
-- write.
alter table games         enable row level security;
alter table matches       enable row level security;
alter table search_probes enable row level security;

do $$
begin
  if not exists (select 1 from pg_policies where policyname = 'anon_read_games') then
    create policy anon_read_games   on games         for select to anon using (true);
    create policy anon_read_matches on matches       for select to anon using (true);
    create policy anon_read_probes  on search_probes for select to anon using (true);
  end if;
end $$;

grant select on win_rate_matrix  to anon;
grant select on decision_quality to anon;
