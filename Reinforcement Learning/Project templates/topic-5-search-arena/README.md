<!-- =========================================================================
     TOPIC 5 TEMPLATE README — "Search Arena"

     The first fifteen lines are graded. Replace the bracketed placeholders
     before your first commit, not the night before submission. Everything
     that is NOT bracketed is real: every number below was produced by a
     command in this repository and is reproducible with it. The commands are
     named next to the numbers.
     ========================================================================= -->

# Search Arena — a Connect Four agent you can play in the browser, and an honest account of what it cost

An education client wants a browser demo that makes search algorithms legible
to someone who has never written one. **Search Arena is a Connect Four service
holding six agents** — a full-width exhaustive search, the same search with
heuristic move ordering and alpha–beta pruning, a forward-pruned beam, Monte
Carlo tree search with UCT, a revised MCTS, and a PUCT agent guided by a network
trained by self-play — **behind one API, over one schema, under an enforced
per-request node budget.** A visitor plays any of them, watches the number of
positions each one considered before moving and how long it took, and reads a
straight answer to the question the demo exists to raise: *which of these is
actually stronger, and what did the strength cost?* The answer is a query
against a table of every game ever played, not a claim in a caption.

| | |
|---|---|
| **Live app** | https://[your-app].streamlit.app |
| **Supabase project** | `[your-project-ref]` — **the same project as your earlier products**; schema in [`db/migrations/`](db/migrations/) |
| **Service tier (local)** | `uvicorn api.main:app --port 8000` → http://127.0.0.1:8000/docs |
| **Service capture** | [link to your screen recording of `POST /act` with an agent, `POST /game`, the node-budget case, and `GET /docs`] |
| **Author** | [name] |

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.11–3.13. Not 3.14.
pip install -r requirements-train.txt                 # laptop / Colab
cp .env.example .env                                  # then fill in your keys

# 1. schema — apply BOTH migrations, in order, to the project you already have
#    (paste db/migrations/001_init.sql then 002_topic5.sql into the SQL editor)
python -m db.seed

# 2. train the learned agent by self-play, and export it. ~90 s, no GPU.
python -m train.train

# 3. the evidence: round-robin tournament + scalability sweep
python -m train.benchmark --games 30          # sandbox default, ~4 min
python -m train.benchmark --agents all --games 200   # the SUBMISSION budget, ~40 min

# 4. the four correctness checks, at a budget worth quoting (~4 min)
python -m train.verify

# 5. serve, and demonstrate it serving
uvicorn api.main:app --reload --port 8000
curl -s localhost:8000/agents | python -m json.tool | head -20
curl -s localhost:8000/act -H 'content-type: application/json' \
  -d "{\"state\": $(python -c "
from envs.connect_four import Position, encode_state
p = Position()
for c in [2,0,3,6]: p.push(c)
print(encode_state(p))"), \"agent\": \"heuristic\"}"
curl -s localhost:8000/game -H 'content-type: application/json' \
  -d '{"agent_a":"heuristic","agent_b":"mcts","seed":0}'

# 6. the user interface
streamlit run ui/app.py

# 7. the gate
pytest -q && ruff check .
```

**Reuse one Supabase project across every topic.** The free tier allows two
active projects per person and you will want the headroom. Migrations are
additive and numbered precisely so that this works: `001_init.sql` is untouched
and `002_topic5.sql` adds three tables, five indexes and two views on top of it.

## What is where

```
api/        the service tier. Owns every agent. NumPy only — see the note below.
  main.py     the standing endpoints, plus /agents, /game, /tournament, /scalability
  policy.py   the serving-side forward pass, and the value-net archive loader
search/     every decision procedure. ON THE SERVING PATH — NumPy, nothing more.
  minimax.py  ONE recursion. Exhaustive and heuristic search are the same
              function with a different ordering callback. Read this first.
  ordering.py the ordering callbacks — where the two baselines actually differ
  mcts.py     UCT, and the PUCT variant the learned agent uses
  net.py      NumPy forward pass for the exported policy-value network
  agents.py   the registry: one name, one configured agent, one table
  budget.py   the node budget, which is a hard bound and not a suggestion
envs/       the game.
  connect_four.py  the RULES. Pure Python + NumPy, imports no framework.
  gym_env.py       the Gymnasium wrapper, imported lazily by make_env()
train/      the training tier. Runs on your laptop. Never deployed.
  train.py      the one command that produces the deployable learned agent
  selfplay.py   self-play + the AlphaZero loss. The ONLY module importing torch.
  benchmark.py  the round robin, the scalability sweep, the node-count table
  verify.py     the four correctness checks, at a quotable budget
  export.py     trained network -> .npz. The seam between the tiers.
ui/         the presentation tier. No search code, no training code, no writes.
  app.py      Concepts · Play · Tournament · Scalability · Revision · Run History · Model Card
  service.py  the one switch between in-process and HTTP service calls
shared/     the contracts. Pydantic models, settings, the data-tier interface,
            and preprocess.py — every transformation, in one importable place.
db/         migrations (001 standing, 002 this topic) and a seed script.
tests/      the standing four, plus the environment, the forced-win position,
            alpha-beta equivalence, MCTS strength, the node budget, the import
            graph, and the torch/NumPy equivalence of the exported network.
policies/   alphazero_c4.npz (105 KiB). Committed — it is what you deployed.
reports/    benchmark.json, verification.json, training.json. Committed, because
            GET /tournament and GET /scalability fall back to them on a clone
            with no database credentials.
```

## The game, and why this one

**Connect Four, 6×7.** The syllabus permits Tic-Tac-Toe and recommends Connect
Four, and the recommendation is worth taking. Tic-Tac-Toe's whole game tree is
smaller than one depth-6 Connect Four search: an exhaustive solver finishes it
in well under a second, so the scalability study — which is a third of the
assessed work — would have nothing to measure. Connect Four's tree is about
4.5 × 10¹² positions. Exhaustive search runs out of laptop somewhere around
depth 8, move ordering visibly changes the node count, and MCTS has a reason to
exist. **Pick the game that makes your measurement interesting, not the one
that makes your code short.**

Three properties of the implementation are load-bearing rather than decorative:

**The position type is built for search, not for convenience.** `Position` has
`push(col)` / `pop()` and mutates in place. A search that copies the board at
each node spends most of its time in the allocator: at depth 6 a full-width
search visits ~137,000 nodes, and copying a 42-element list at every one of them
is 137,000 allocations to do arithmetic that a make/unmake pair does for free.
The Gymnasium wrapper copies where Gymnasium's contract requires it — which is
correct there and would be ruinous inside the recursion.

**Terminal detection is incremental.** A win can only be created by the piece
just played, so `push` scans only the lines through that cell (at most 13,
typically 7) rather than all 69. The slow, obviously-correct full scan is kept
for positions arriving from outside, and
`tests/test_env_connect_four.py` asserts the two agree — which is the standard
shape of this kind of optimisation: keep both, and test that they match.

**The board survives a JSON round trip, and the round trip is tested.** `/act`
takes `state` as 43 floats: 42 cells then the side to move. `decode_state`
rejects fractional values, floating pieces, impossible piece counts, and a
`player` that contradicts the board — because a search service that answers
questions about unreachable positions is a service whose answers cannot be
checked against a real game.

## The design requirement: ONE scaffold, two strategies

The syllabus makes this a requirement rather than a suggestion, and it is the
single most important thing in the repository:

> *Implement a single search scaffold that supports both exhaustive evaluation
> of all legal continuations and a heuristic-ordered variant... The two share
> one implementation and differ in the node-ordering callback.*

`search/minimax.py` contains one recursion — depth-limited negamax. What varies
is an injected callback (`search/ordering.py`), a boolean for whether cutoffs
are taken, and a leaf evaluator. Nothing else. The `exhaustive` and `heuristic`
agents in `search/agents.py` are two instances of one class.

Three reasons, and they generalise well past this assignment:

1. **Two implementations produce two sets of bugs.** The most common failure in
   game-search coursework is a sign error, and a sign error in one of two copies
   produces a heuristic agent that is worse than exhaustive for a reason that
   looks like *"heuristics are worse"* rather than like a defect. With one
   recursion a sign error breaks both agents identically and is found in an
   afternoon.
2. **Otherwise the comparison stops being a comparison.** If the two differ in
   the recursion as well as the ordering, a node-count difference cannot be
   attributed to the ordering. The experiment has a confound in its source code.
3. **It names the actual idea.** Exhaustive search and heuristic search are not
   two algorithms. They are one algorithm under two policies for which child to
   look at next.

### The result that surprises people

**Reordering with pruning switched off changes nothing at all.** A full-width
search visits every node whatever order it visits them in. Ordering only pays
when there is a cutoff to trigger — which is why the node-count table below has
four rows and not two, and why `tests/test_alpha_beta.py` contains a test
asserting that `exhaustive` and `heuristic-without-pruning` expand *exactly* the
same number of nodes. Good move ordering is not a search strategy; it is a
multiplier on alpha–beta.

The ordering callback is also allowed to **drop** moves, which is what
`make_beam_ordering` does. That is forward pruning, and it is a categorically
different thing: reordering is value-preserving and provably returns the
exhaustive answer, while a beam is *unsound* — it can discard the winning move
and no extra depth recovers it, because the branch is gone.
`tests/test_forced_win.py` demonstrates that failing on purpose.

## Build-step checklist

- [x] Fork the template. **Reuse the existing Supabase project** — do not
      provision a second one. `002_topic5.sql` is additive on top of `001_init.sql`.
- [x] `envs/connect_four.py`: Connect Four as a Gymnasium-compatible
      environment, with legal-move masking (`action_mask()`), incremental
      terminal detection, and a board encoding that survives a JSON round trip
      through the API. The rules module imports no framework; the Gymnasium
      wrapper is `envs/gym_env.py`, imported lazily.
- [x] **One shared search scaffold** (`search/minimax.py`) supporting both
      exhaustive evaluation and a heuristic-ordered variant, differing only in
      the injected node-ordering callback, with α–β pruning as a flag on the
      same recursion. Node counts for each reported below.
- [x] MCTS with UCT (`search/mcts.py`): selection, expansion, simulation,
      backpropagation, exploration constant `C` configurable per agent.
- [x] AlphaZero-inspired agent: the random rollout replaced by a learned
      policy-value network trained by self-play in `train/selfplay.py`, exported
      to a NumPy archive and served like every other policy in this course.
- [x] `train/benchmark.py`: round-robin tournament logging every game to
      `games`, aggregated into `matches`, plus a scalability sweep into
      `search_probes`. Computation time, peak memory, decision quality against a
      fixed reference, and scalability against depth.
- [x] `db/migrations/002_topic5.sql`: `games`, `matches`, `search_probes`, five
      indexes, and the `win_rate_matrix` and `decision_quality` views.
- [x] FastAPI service: `POST /act` (agent name plus board state in, move out),
      `POST /rollout`, `POST /game`, `GET /runs`, `GET /healthz`, `GET /version`,
      plus `/agents`, `/tournament` and `/scalability` for the UI. **A
      per-request node budget is enforced and the consumption is returned.**
- [x] Streamlit frontend: Concepts, Play, Tournament, Scalability, Revision,
      Run History, Model Card.
- [x] Pytest: the forced win on a constructed position, node-budget
      enforcement, α–β identical to plain minimax, MCTS beating random by a wide
      margin, and the standing four untouched.
- [ ] Critical analysis report (≈600 words) — skeleton below; **finish it**.
- [ ] Revised agent — `mcts_v2` is implemented and benchmarked; **write up the
      trade in your own words and re-run the harness on your own numbers**.
- [ ] Deploy the Streamlit app; verify the Supabase project is active; record
      the service tier running locally under uvicorn.

## Architecture — two clouds, three tiers

**Presentation** (Streamlit Community Cloud, deployed) → **Service** (FastAPI,
in this repository, run under uvicorn locally and imported in-process in
production) → **Data** (Supabase Postgres, deployed).

The service tier is a real application with real Pydantic contracts, exercised
over HTTP by the test suite. In the deployed app the Streamlit tier imports the
same handlers instead of crossing the network; `SERVICE_MODE` is the only thing
that changes.

**The tier boundary that matters most in this topic** is the one between the UI
and the service. The "Play" tab holds the board in `st.session_state` and
applies the *human's* move locally — but every agent move is a `POST /act`, and
whether the game has ended is the service's 422, not a rule reimplemented in
Streamlit. Game state is presentation state; **decisions are not**. If the UI
constructed its own `MCTS(iterations=400)`, the agent a visitor plays would be
the browser's copy, and the win rates in the "Tournament" tab would describe
different software from the game on the screen. `search/agents.py` is the one
place an agent is constructed, and the API, the UI and the benchmark harness all
resolve through it.

**Why a database tier at all.** Every claim this product makes has the form
*agent A beats agent B at cost C*, and with 30 games per pairing the standard
error on a win rate is about 9 percentage points — wider than most of the
differences you will be tempted to explain. With every game as a row you cannot
accidentally report the pairing that went your way, and the comparison tables
fall out of a `GROUP BY`.

## The node budget — the one thing this service has that the others do not

Every other product in this course serves a **fixed-cost** forward pass: an
observation arrives, a few matrix multiplies happen, an action leaves, and the
work is the same whatever the observation was. A tree search is not like that.
The same 43-number board with `depth: 9` instead of `depth: 4` is roughly two
thousand times the work. **An unbounded search endpoint is a denial-of-service
vector against your own free-tier instance, reachable by anyone who can type a
number into a JSON body — including your own Streamlit app with a slider on it.**

Four properties, each because the obvious alternative is worse:

| property | why the alternative fails |
|---|---|
| The budget counts **nodes**, not depth | α–β at a fixed depth varies ~30× in node count depending on move ordering. A bound that varies 30× is not a bound. |
| Exhaustion **degrades**, it does not raise | A 500 in the middle of a human's game is worse than a slightly weaker move. The search stops descending and scores statically. |
| The consumption is **returned** | `nodes_expanded` comes back in the response and is the same counter that lands in `games`. A budget you cannot observe is a budget you cannot tune. |
| A request **above the ceiling** is a 422 | Rejected by the Pydantic validator before a single node is expanded, not after five minutes of work. |

```
$ curl -s localhost:8000/act -d '{"state": [...], "agent": "exhaustive",
                                  "depth": 8, "node_budget": 500}'
{"action": 3, "nodes_expanded": 500, "node_budget": 500,
 "budget_exhausted": true, "search_depth": 8, "legal_moves": [0,1,2,3,4,5,6], ...}
```

`tests/test_node_budget.py` asserts all four, and it is a required test.

## The no-PyTorch-in-serving rule — and what this topic adds to it

`import torch` alone occupies roughly **490 MB** of resident memory against
Streamlit Community Cloud's **690 MB** guarantee. The entire deployed stack here
measures **82 MB**. So: train in PyTorch outside the deployed app, export the
weights to a NumPy `.npz`, and evaluate the forward pass in NumPy. For the
policy-value network in this topic that is four matrix multiplies, a ReLU, a
softmax and a `tanh` — read `search/net.py`; it is 60 lines, and it is the
entire "AI" of the strongest agent at inference time.

**This topic adds a second, subtler guard.** `gymnasium` *is* a serving
requirement (`POST /rollout` runs episodes server-side), so the no-torch test
cannot notice a search agent that quietly grew a gymnasium dependency. The claim
in `requirements-serve.txt` — *the search agents need nothing beyond NumPy* — is
checked two ways: `tests/test_import_graph.py` imports `api.main` in a clean
subprocess and asserts gymnasium is absent from `sys.modules`, and CI
uninstalls gymnasium entirely and plays a move.

That guard found a real defect while this template was being written:
`envs/connect_four.py` had `try: import gymnasium` at module scope inside an
`except ImportError` that made it look optional. **A defensive import still
imports.** The fix was the split into `connect_four.py` (rules, pure) and
`gym_env.py` (wrapper, lazy).

## Free-Tier Notes

| Limit | Value | How this product handles it |
|---|---|---|
| Streamlit Cloud memory | 690 MB guaranteed | ~82 MB measured; no framework and no gymnasium on the search path (`tests/test_import_graph.py`), and the one policy artifact is 105 KiB |
| Streamlit Cloud CPU | shared, ~1 vCPU | **this is the binding constraint for this topic, not memory.** The node budget is what keeps a search inside a request timeout; the UI's default is 10,000 nodes (~0.3 s), not the 200,000 ceiling |
| Streamlit Cloud sleep | after 12 h idle | wakes on first request; note the cold start in your demo |
| Supabase storage | 500 MB | one row per game. A 200-games-per-pairing round robin between 6 agents is 3,000 rows ≈ 0.5 MB. Per-MOVE logging would be 40× that and is deliberately not done — the move records live in the `/game` response, not in a table |
| Supabase projects | **2 active per person** | one project reused across every topic; `002_topic5.sql` is additive |
| Supabase pause | after 1 week idle | the UI degrades visibly (health banner), and `/tournament` and `/scalability` fall back to the committed `reports/benchmark.json` while flagging `degraded: true` |
| Python version | 3.11–3.13 | pinned in CI; 3.14 has no Box2D wheels for later topics |

## Theoretical Brief

*350–600 words, mirrored in the Streamlit "Concepts" tab, which carries the
equations. Replace this with your own words.* The argument the rest of this
repository makes:

**Negamax, not minimax.** Connect Four is zero-sum, so `min(a,b) = −max(−a,−b)`
and the two-function minimax/maximin pair collapses into one function that
always maximises and negates on the way back. That halves the code and removes
the classic bug — a `min` node that got a `max` body during a copy-paste. The
price is that every value is *from the point of view of the side to move*, and
you must hold that in your head at every return.

**What α–β buys.** Pruning cannot change the value; it changes how much of the
tree has to be examined to be sure of it. With perfect move ordering the node
count falls from `b^d` to about `b^(d/2)`, which is the same as **doubling the
reachable depth for the same budget**. The measured growth factors below show
this happening.

**A subtlety worth knowing before you show anyone your per-move numbers.** With
α–β on, a root child that "fails low" returns an *upper bound*, not a value — it
stopped as soon as it knew the move could not win. Only the best move's value is
exact. `search_root(..., exact_root_values=True)` pays 2–4× the nodes to make
all seven exact; `stats.root_values_are_bounds` records which you got, and the
"Play" tab labels them accordingly rather than showing a human a number the
search never computed.

**UCT.** MCTS needs no evaluation function: it estimates a position's value by
playing it out, and concentrates effort with `Q_i + C·√(ln N / N_i)`. The
exploration term decays as a child is visited and grows as the parent is, so a
rarely-tried move keeps getting a chance until the evidence against it is
strong. `C = √2` is the value UCB1's regret bound is derived for when rewards
lie in `[0,1]`; ours lie in `[−1,1]`, so `C` is a hyperparameter to sweep rather
than a constant to trust.

**And what AlphaZero changes.** Two things: replace the random playout with a
**learned value**, and replace UCT's uniform exploration term with a **learned
prior** (PUCT: `Q_i + C·P_i·√N/(1+N_i)`). The network is then trained on the
search's own visit distribution — the search plays better than the network
alone, so its output is a training target for free. On a board with `b = 7` the
prior matters little; on Go's `b ≈ 250` it is the difference between possible
and impossible.

## Quantitative Analysis

*Every number in this section came from a command in this repository, and the
command is named. Re-run them; the seeds are fixed.*

### Node counts: exhaustive vs heuristic-ordered vs α–β

`python -m train.benchmark` · from the empty board · `search_probes` table

| variant | ordering | α–β | d=2 | d=3 | d=4 | d=5 | d=6 | growth / ply |
|---|---|---|---|---|---|---|---|---|
| `exhaustive` | left-to-right | off | 57 | 400 | 2,801 | 19,608 | 137,257 | **7.03** |
| `natural+ab` | left-to-right | **on** | 48 | 212 | 991 | 2,921 | 16,924 | 4.77 |
| `heuristic+ab` | domain evaluation | **on** | 21 | 82 | 221 | 864 | **1,784** | **3.04** |
| `beam3` | top 3 by evaluation | on | 9 | 22 | 55 | 86 | 320 | 2.50 |

**Read the growth column, not the totals.** A full-width search should grow by
`b = 7` per ply and measures **7.03**. α–β with good move ordering should
approach `√b ≈ 2.65` and measures **3.04**. That is the classic theoretical
result, measured rather than quoted, and it is why `heuristic+ab` reaches depth
6 for 1,784 nodes — less than the 2,801 that `exhaustive` spends to reach depth
**4**. Two extra plies, for less than the original price.

The row that is deliberately **missing** is "heuristic ordering, α–β off". It
would be identical to `exhaustive` — 57 / 400 / 2,801 / 19,608 / 137,257 —
because ordering with nothing to cut off is a no-op.
`tests/test_alpha_beta.py::test_ordering_alone_changes_nothing_without_pruning`
asserts that equality so it cannot silently stop being true.

Averaged over 25 random midgame positions at depth 6 rather than from the empty
board alone (`python -m train.verify`, `reports/verification.json`):

| | mean nodes at depth 6 | vs the row above |
|---|---|---|
| exhaustive (full width, left-to-right) | 125,563 | — |
| heuristic ordering, **no** pruning | 125,563 | **1.00×** — identical, by construction |
| natural order + α–β | 16,157 | 7.8× |
| **heuristic order + α–β** | **1,459** | **11.1×** |
| beam width 3 + α–β (unsound) | 184 | 7.9× |

α–β alone buys **7.8×**. Ordering on top of α–β buys a further **11.1×**.
The ordering is worth more than the pruning — and neither is worth anything
without the other.

### The four correctness checks

`python -m train.verify` · written to `reports/verification.json`

| check | budget | result |
|---|---|---|
| α–β returns the **identical** value to plain full-width minimax | 1,000 random reachable positions, depth 4 | **0 mismatches** |
| the strong search (`heuristic+ab`, depth 6) never **loses** to random | 300 games, alternating colours | **300 W / 0 D / 0 L** |
| MCTS beats random by a wide margin at a high budget | 100 games, 800 simulations | **100 W / 0 D / 0 L — 100%** |
| heuristic ordering **reduces** the node count under α–β | 25 positions, depth 6 | **11.1× fewer nodes** |

The first is `==` and not `approx`: α–β and full-width minimax do the same
arithmetic in a different order, and values propagate by negation and maximum
only — never by addition — so the results are bit-identical. A tolerance would
hide exactly the one-in-a-hundred wrong value that a bad cutoff produces. Node
counts over those same 1,000 positions at depth 4: **2,579 full width → 779 with
α–β → 188 with ordering**.

Note the second claim's wording: *never loses*, not *always wins*. A draw is
possible in principle and demanding 100% wins would be demanding something the
game does not guarantee. **This is not a perfect-play agent** — perfect play in
Connect Four is a 42-ply solve, and nothing here claims that. It is the
strongest agent in this arena, and the check is that it is not broken.

### The round-robin tournament

`python -m train.benchmark --agents random,exhaustive,heuristic,mcts,mcts_v2,alphazero --games 30`
· 15 pairings · **450 games**, each one a row in `games` · 586 s

Win rate, row agent against column opponent. **Draws count as half a win** —
that definition lives in `db/migrations/002_topic5.sql`, in
`train/benchmark.py` and in the UI, and the three must not drift.

|            | random | exhaustive | heuristic | mcts | mcts_v2 | alphazero |
|---|---|---|---|---|---|---|
| **random**     |   —    | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **exhaustive** | 1.000  |   —   | 0.500 | 0.667 | 0.633 | 1.000 |
| **heuristic**  | 1.000  | 0.500 |   —   | 0.550 | 0.600 | 1.000 |
| **mcts**       | 1.000  | 0.333 | 0.450 |   —   | 0.333 | 0.867 |
| **mcts_v2**    | 1.000  | 0.367 | 0.400 | 0.667 |   —   | 0.983 |
| **alphazero**  | 1.000  | 0.000 | 0.000 | 0.133 | 0.017 |   —   |

Cost per **decision**, from the same 450 games:

| agent | nodes / decision | ms / decision | peak KiB / decision |
|---|---|---|---|
| `random` | 1.0 | 0.03 | **0.4** |
| `exhaustive` (depth 4, full width) | 2,097.8 | 53.8 | **1.7** |
| `heuristic` (depth 4, ordered + α–β) | **180.1** | **19.2** | **2.6** |
| `mcts` (300 simulations) | 922.0 | 53.1 | **97.4** |
| `mcts_v2` (300 tactical simulations) | 985.8 | 100.6 | **97.2** |
| `alphazero` (200 PUCT simulations) | 649.4 | 33.0 | **100.2** |

*Milliseconds were measured on a shared 2-vCPU container under concurrent load
and are roughly 2.5× what the same code does on an idle machine; they also
include the `tracemalloc` overhead (`--no-memory` removes it). **Node counts are
the portable number** — they are identical on any machine, which is exactly why
the budget counts nodes. Quote both, and say which machine.*

**Four things to read out of these two tables.**

1. **`heuristic` and `exhaustive` are dead level at 0.500 — while `heuristic`
   expands one twelfth of the nodes.** That is the cleanest possible statement
   of what move ordering plus pruning buys, because the two agents *are the same
   recursion at the same depth with the same evaluator*: they return the same
   move (`tests/test_alpha_beta.py` proves it), so they must draw the series,
   and the entire difference is 2,098 nodes against 180. Any node-count
   difference here is attributable to the ordering callback and to nothing else,
   which is the whole reason for the shared scaffold.
2. **MCTS is the weakest of the searchers here, and that is the honest result.**
   On a 7-wide board with a decent hand-written evaluator, α–β is simply better
   value: 180 nodes and 19 ms buys more than 922 nodes and 53 ms. MCTS's
   advantages — no evaluation function required, anytime behaviour, cost set by
   a simulation count rather than by `b^d` — do not pay at `b = 7`. They pay at
   `b = 250`, which is why AlphaGo is a Go program and not a Connect Four
   program. **Do not report this as MCTS being disappointing; report it as a
   measurement of where the crossover is.**
3. **`alphazero` beats random 100% and loses to both α–β agents 0.000.** 200
   network evaluations per move, from a network that saw 3,822 positions in 90
   seconds of self-play. That is what this budget buys, and claiming more would
   be claiming something it did not. At `--iterations 20 --games 200
   --simulations 200` the picture changes; run it and report what you find.
   Reporting a weak learned agent honestly is worth more marks than reporting a
   strong one you cannot reproduce.
4. **Memory separates the two families by 40×** — see below.

### Peak memory: the tree and the stack

| family | peak KiB per decision |
|---|---|
| α–β (`exhaustive`, `heuristic`) | 1.7 – 2.6 |
| MCTS / PUCT (`mcts`, `mcts_v2`, `alphazero`) | 97 – 100 |

α–β holds a **stack** — one `Position` mutated in place, a recursion 6 deep.
MCTS holds a **tree** — several hundred `Node` objects that live until the
search ends. A 40× difference is worth a measured number rather than an
assertion, and it is the reason `Node` uses `__slots__`.

**Getting this number right took three attempts, and both failures are
instructive:**

* `resource.getrusage(RUSAGE_SELF).ru_maxrss` is a process-wide high-water mark
  that never goes down. After the first MCTS game, every later game reports the
  same figure — including the α–β games, which then appear to use exactly as
  much memory as MCTS.
* `tracemalloc` bracketing the whole **game** measures both agents together, so
  a cheap agent inherits its opponent's footprint. The random agent came out at
  258 KiB a move, which is a statement about who it played.
* `tracemalloc.reset_peak()` does **not** reset to zero — it resets the peak to
  the current traced total. Reading the raw peak still reports every live
  allocation the game has accumulated. The working measurement is
  `peak_after − current_before`, per decision, maximised over the game.

### Scalability

`python -m train.benchmark` · `search_probes` · rendered in the "Scalability" tab

| depth | exhaustive nodes | exhaustive ms | heuristic+ab nodes | heuristic+ab ms |
|---|---|---|---|---|
| 1 | 8 | 0.2 | 8 | 0.4 |
| 2 | 57 | 0.9 | 21 | 1.7 |
| 3 | 400 | 6.4 | 82 | 6.1 |
| 4 | 2,801 | 52.9 | 221 | 22.0 |
| 5 | 19,608 | 364.2 | 864 | 65.2 |
| 6 | 137,257 | **3,145.3** | 1,784 | **226.0** |

At depth 6 the exhaustive search takes **3.1 seconds per move** on this
container; depth 8 would be roughly 150 seconds, past every timeout that
matters and past the 2,000,000-node request ceiling. The ordered α–β search at
depth 6 takes 226 ms and 1,784 nodes. **The entire argument for move ordering is
in that one row**, and it is also the reason the node budget is a budget on
nodes: at depth 6 the two variants differ by 77× in cost while differing not at
all in the move they return.

The sweep is run from two positions (`empty` and a crowded `midgame`) because a
node count is meaningless without the position it came from. The midgame
position prunes slightly better — 1,398 nodes for `heuristic+ab` at depth 6
against 1,784 — because more pieces means more terminal detections and more
cutoffs.

**On board size rather than depth.** The syllabus asks for scalability as depth
*or* board size grows. Depth is swept here because it is the axis this codebase
can sweep honestly: `ROWS`, `COLS` and `CONNECT` are module constants in
`envs/connect_four.py` and the line tables are derived from them, so an 8×8
board is a three-line change — **but** the committed evaluation weights were
chosen for 6×7, and a scalability curve that also changes the evaluator is
measuring two things at once. If you extend it, say which of the two you
changed.

## The revised agent

**What changed.** `mcts_v2` differs from `mcts` in exactly two places, both in
`search/agents.py` and both one argument:

1. **`tactical_playout` instead of `random_playout`.** The simulation takes an
   immediate win and blocks an immediate loss; otherwise it moves at random.
2. **`C = 0.9` instead of `√2 ≈ 1.414`.**

**Why, from the benchmark rather than from taste.** The original `mcts` loses
to `heuristic` 0.450 and to `exhaustive` 0.333 while expanding **five times**
the nodes of `heuristic` (922 against 180 per decision). Two hypotheses fit
that: the search is too shallow, or the estimates it is averaging are too noisy.
The second is testable and turned out to be the case — a uniform random playout
from a *won* Connect Four position walks past the win roughly half the time, so
a won position is routinely backed up as a loss. Averaging more samples of a
biased estimator buys very little; reducing the bias buys a lot. The lower `C`
follows from the same argument: once the value estimate is less noisy, less
exploration is needed before it can be trusted, and the original `C` spends
simulations re-checking children the better estimate had already settled.

Note what was *not* changed and why: the simulation count stayed at 300. Raising
it would have improved the agent too, and would have made the comparison
uninterpretable — a revision that changes both the algorithm and the budget
cannot tell you which one helped.

**Which metrics evaluate it, and why those are the right ones.**

| metric | why it is here |
|---|---|
| Win rate head-to-head against `mcts` | the direct claim. Same game set, same seeds, same harness. |
| Win rate against the fixed reference (`random`) | comparable across weeks, because `random` never improves. Both agents saturate it at 1.000, which is itself informative: this metric is a floor test, not a discriminator |
| Win rate against every other agent in the arena | one opponent is one opponent. A revision that beats only its predecessor may have learned to exploit it |
| **Nodes** per decision | distinguishes *thinking better* from *thinking more*. Without it, any revision that quietly raised the compute budget looks like a better algorithm. |
| **Milliseconds** per decision | what the trade actually cost the user |

The nodes/milliseconds pair is what makes this an argument rather than an
anecdote. A revision that improved the win rate by tripling the node count would
not be an improvement to the *algorithm*; it would be a bigger budget wearing an
algorithm's name. Reporting cost next to quality is the only thing that
distinguishes the two, and it is the reason `games` has cost columns at all.

**Results, through the identical harness on the identical game set** (450
games, 30 per pairing, seeds derived from the pairing index so both agents saw
the same sequences):

| | `mcts` | `mcts_v2` | change |
|---|---|---|---|
| **Head to head** (revised vs original) | 0.333 | **0.667** — 20 W / 0 D / 10 L | **+0.334** |
| vs `exhaustive` | 0.333 | **0.367** | +0.034 |
| vs `alphazero` | 0.867 | **0.983** | +0.116 |
| vs `heuristic` | 0.450 | **0.400** | **−0.050** |
| vs `random` (fixed reference) | 1.000 | 1.000 | 0.000 |
| Nodes / decision | 922.0 | 985.8 | **+6.9%** |
| ms / decision | 53.1 | 100.6 | **+89%** |
| Peak KiB / decision | 97.4 | 97.2 | −0.2% |

**The trade, named and defended.** The revised agent wins the head-to-head
series 20–10 **at essentially the same node count** (+6.9%) and the same memory.
That is what makes it an algorithmic improvement rather than a bigger budget:
the extra strength is in the quality of each simulation, not in the number of
them. It costs **1.9× the wall clock**, because `winning_moves` runs at every
step of every playout.

That cost binds in exactly one place, and it is worth stating precisely: **under
a wall-clock budget rather than a node budget, `mcts_v2` would run about half as
many simulations, and this table would not hold.** This service enforces a node
budget, so the trade is favourable *here*. A deployment that charged for CPU
seconds would have to re-run the comparison with time on the x-axis before
accepting it. *Name the budget your comparison is under; a result that is true
under one and false under another is not a result until you say which.*

**The one metric that went the wrong way, reported rather than dropped.**
`mcts_v2` scores 0.400 against `heuristic` where `mcts` scored 0.450 — 12 wins
against 13 wins and a draw, over 30 games. The standard error on a win rate at
n = 30 is about 9 percentage points, so a 5-point difference is comfortably
inside the noise and this is **not** evidence that the revision hurt against
that opponent. It is also not evidence that it helped. The honest statement is:
the revision is decisively better head-to-head and against two of the four other
opponents, indistinguishable against the third, and unmeasurable against the
fourth because both agents beat it every time. Re-running at `--games 200` is
what would settle the `heuristic` column, and it is the first thing to do if
that cell matters to your argument.

`tests/test_mcts.py::test_the_revised_agent_beats_the_original_head_to_head`
asserts the *direction* of the head-to-head effect at a small budget, so a
change that silently reverses it fails the build. The magnitude belongs to the
benchmark, not to a test — 20 games has a standard error of about 11 points.

## Critical analysis report (≈600 words)

*Required, in this README. The skeleton below is the argument; write it in your
own words and cite your own numbers.*

### Code logic and key functions

*Cover: `search_root` / `negamax` and the injected ordering callback;
`heuristic_ordering`'s tier-then-value sort; `NodeBudget.spend` and why the
charge happens before descending; `MCTS.search`'s four phases and the negation
in `_backpropagate`; `PolicyValueNet.evaluate`'s masking; and
`canonical_planes`' side-to-move encoding.*

### MCTS against heuristic search: computation cost, decision quality, applicability

*The measured answer on this board is above, and it is that α–β wins on all
three at `b = 7`. The interesting content is WHY, and where the crossover is:
MCTS's cost is set by a simulation count rather than by `b^d`, it needs no
evaluation function, and it is anytime. Each of those is worth nothing at `b = 7`
with a decent evaluator and worth everything at `b = 250` with none. State the
conditions under which your conclusion reverses — a comparison that does not is
a comparison you cannot transfer.*

### Where these techniques are used in practice

*Required coverage: **AlphaGo** (MCTS + supervised policy net + value net + RL);
**AlphaZero** (the supervised bootstrap dropped, one network, self-play only,
generalising to chess and shogi); **MuZero** (the model itself learned, so the
search runs in a latent space and the rules are never given); **planning** in
operations research and logistics; **robotics** (sampling-based motion planning,
model-predictive control — the same "search over futures at decision time"
shape); and **search at inference time in reasoning models**, which is the
direct descendant: sample multiple continuations, score them, and spend more
compute per query on the ones that look promising. The connection to make
explicit is that MCTS is the ancestor of test-time compute scaling, and the same
trade — more compute per decision against better decisions — is being made
again.*

### The most challenging part, and how you would optimise it

*Candidates, honestly: the sign conventions (three places, one convention:
`negamax`, `_backpropagate`, `PolicyValueNet.value_of`); the root-value bounds
subtlety; and making the node budget a hard bound rather than an approximate
one — the first two implementations here overshot, once at the root loop and
once in MCTS's selection descent, both by a small enough margin to be easy to
miss.*

*For optimisation: a transposition table (Connect Four transposes heavily —
column orders that reach the same position are common), iterative deepening with
the previous iteration's best move first, bitboards for the position type, and a
tuned evaluator. Estimate the gain before you implement any of them; the node
counts above tell you which is worth the most.*

## AI-Assistance Disclosure

*Required. What did you generate, with which tool, and how did you verify it?
Generated code must be read, understood and tested by you; blind paste-through
is not acceptable. Note that this topic is unusually easy to verify — the
engine's correctness is checkable, which is what `train/verify.py` is for.*

## Limitations & Responsible Use

*At least four concrete limitations, each with how you would test whether it
binds. The Model Card tab carries the full version; keep the two in sync.*

1. **None of these agents plays Connect Four well in absolute terms.** The game
   is solved — a first-player win with correct play from the centre column — and
   the deepest search here is 6 plies out of 42. Test: play the strong agent
   against a published solver's opening book and count how many plies it takes
   to leave the winning line.
2. **The static evaluation function was written by hand and never tuned.** The
   weights `(0, 1, 12, 60)` and the centre bonus of 3 were chosen for
   plausibility. Every α–β result is conditional on them. Test: perturb each
   weight by ±50%, re-run the tournament, and see whether the ordering of the
   agents changes; if it does, the ranking is a ranking of evaluators.
3. **The node budget makes every agent weaker under load in a way the
   tournament did not measure.** The tournament ran at 200,000 nodes; the
   deployed UI defaults to 10,000. Test: re-run the round robin at the UI's
   default budget and compare — `games.budget_exhausted_moves` already records
   the truncations, so it is a `GROUP BY`.
4. **30 games per pairing has a standard error of about 9 percentage points.**
   Several differences in the matrix above are inside that. Test: re-run at
   `--games 200` and check which orderings survive.
5. **Every agent in the tournament shares one RNG stream by construction.** The
   seeds are derived from the pairing index, so `mcts` and `mcts_v2` see
   correlated game sequences. That is deliberate — it is a paired comparison and
   reduces variance — but it means the games are not independent samples and a
   naive binomial interval is optimistic. Test: re-run with independent seeds
   per agent and compare the interval widths.

Then: foreseeable misuse, reward-specification risk, and the worldview
reflection your topic calls for. For a game agent the honest answer to "what
goes wrong if someone optimises this harder" is usually *not much* — which makes
it worth thinking instead about the **decision-support** version of the same
system, where a bounded search that is confidently wrong past its horizon looks
exactly like one that is right, and the node budget that protects your instance
is also the thing quietly limiting how far ahead anyone can see.
