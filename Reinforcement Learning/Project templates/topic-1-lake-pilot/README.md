<!-- =========================================================================
     TOPIC 1 TEMPLATE README — "Lake Pilot"

     The first fifteen lines are graded. Replace the bracketed placeholders
     before your first commit — not the night before submission.

     The product pitch below is written for you as a worked example of the
     register the quality bar asks for. Rewrite it in your own words once the
     numbers are yours; a pitch you did not write reads like one.
     ========================================================================= -->

# Lake Pilot — watch an agent learn to cross ice it cannot walk straight on

Lake Pilot is a live demonstration of what "an agent learns from experience"
actually means. A client picks an agent, presses a button, and watches it try
to cross a frozen lake where every step slides: the intended move happens only
a third of the time. An untrained agent wanders and drowns. A trained agent
crosses roughly half the time — and does it by hugging walls and refusing
shortcuts, which is not what a person would have taught it. Both agents run
through the same typed HTTP endpoint, every training episode of every run is a
row you can query, and the comparison on the screen is the same comparison in
the database. It is for a non-technical stakeholder who needs to see the
difference between guessing and learning before they will fund either.

| | |
|---|---|
| **Live app** | https://[your-app].streamlit.app |
| **Supabase project** | `[your-project-ref]` — schema in [`db/migrations/`](db/migrations/) |
| **Service tier (local)** | `uvicorn api.main:app --port 8000` → http://127.0.0.1:8000/docs |
| **Service capture** | [link to your screen recording of the service serving `POST /act` and `GET /docs`] |
| **Author / section** | [name] |

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.11–3.13. Not 3.14.
pip install -r requirements-train.txt                 # laptop / Colab
cp .env.example .env                                  # then fill in your keys

# 1. apply the schema — paste db/migrations/001_init.sql into the Supabase SQL editor
python -m db.seed

# 2. record the spaces (build step 3 asks for these in this README)
python -m envs

# 3. train: baseline, then Q-learning, then export and register the artifact
python -m train.train                                 # ~20,000 episodes x 3 seeds
python -m train.train --episodes 4000 --seeds 2       # a 10-second smoke test

# 4. serve, and demonstrate it serving
uvicorn api.main:app --reload --port 8000

# 5. the user interface
streamlit run ui/app.py

# 6. the gate
pytest -q && ruff check .
```

`policies/` ships with test fixtures only. Step 3 is what puts
`q_table.npz` and `untrained_policy.npz` there; until you run it, the "Watch"
tab has nothing trained to show.

## The environment

Paste your own `python -m envs` output here. It should look like this:

```
env_id             : FrozenLake-v1
map_name           : 8x8
is_slippery        : True
observation_space  : Discrete(64)
action_space       : Discrete(4)   {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'}
max_episode_steps  : 200   (a truncation, not a termination)
reward             : 1.0 on reaching G, 0.0 on every other transition
```

**Why the stochastic transition function makes this harder than the
deterministic variant, in one sentence** (build step 3 asks for exactly one):
because the same action from the same square lands you somewhere different on
different visits, a solution can no longer be a memorised route and must
instead be a decision for every square, learned from many noisy samples rather
than from one successful trajectory. The longer argument — including why the
optimal policy deliberately steers into walls — is at the top of
[`envs/__init__.py`](envs/__init__.py).

## Build-step checklist

Tick these off as you go. They are the graded build steps, in order.

- [ ] Fork the template, provision a free-tier Supabase project, and fill in
      the first fifteen lines of this README.
- [ ] Apply `db/migrations/001_init.sql` and seed a configuration with
      `python -m db.seed`.
- [ ] Instantiate `FrozenLake-v1` 8×8 with `is_slippery=True`; record the
      observation and action spaces above, and the one-sentence answer on
      stochasticity.
- [ ] Run the random baseline — `python -m train.random_agent` — for 1,000
      episodes across at least 3 seeds, one row per episode into `episodes`.
      Record the mean return and its standard error below.
- [ ] Train tabular Q-learning — `python -m train.qlearning` — for at least
      20,000 episodes across the same seeds, logging **every** episode with the
      ε in force at that episode.
- [ ] Export the Q-table to `policies/*.npz` and register it in `policies`
      with its byte size and SHA-256 (`python -m train.train` does both).
- [ ] Confirm the service tier answers on all seven endpoints: `POST /act`,
      `POST /rollout`, `GET /runs`, `GET /episodes`, `GET /policies`,
      `GET /healthz`, `GET /version` — plus the built-in `GET /docs`.
- [ ] Fill in the five Streamlit tabs: Concepts, Watch, Compare, Run History,
      Model Card.
- [ ] Write the engineering report (≈500 words) — the questions are below.
- [ ] Write the worldview reflection (≈200 words) — the prompt is below.
- [ ] Green tests: `pytest -q` and `ruff check .`, including the no-torch guard.
- [ ] Deploy: Streamlit Cloud URL reachable, Supabase project **active, not
      paused**, and a screen capture of the service tier under uvicorn linked
      above.
- [ ] At least eight meaningful commits. Not one dump.

## What is where

```
api/        the service tier. Owns the policy. Imports NumPy, never a framework.
  main.py     the standing endpoints + /episodes + /docs
  policy.py   the entire serving-side forward pass — an array lookup, here
ui/         the presentation tier. No policy code, no training code, no writes.
  app.py      the five tabs
  service.py  the one switch between in-process and HTTP service calls
train/      the training tier. Runs on your laptop or in Colab. Never deployed.
  train.py        the entry point: both agents, every seed, then export
  random_agent.py the baseline — the denominator for every claim you make
  qlearning.py    the learner, the ε schedules, and greedy evaluation
  telemetry.py    experiment rows, buffered episode writes, evaluation rows
  export.py       trained table -> .npz. The seam between the tiers.
shared/     the contracts. Pydantic models, settings, the data-tier interface.
envs/       FrozenLake-v1, 8x8, slippery, behind make_env(). `python -m envs`.
db/         migrations and a seed script. The schema is checked in and tested.
tests/      the standing four, plus tests/test_topic1.py.
policies/   exported artifacts. Committed, because they are what you deployed.
```

## Architecture — two clouds, three tiers

**Presentation** (Streamlit Community Cloud, deployed) → **Service** (FastAPI,
in this repository, run under uvicorn locally and imported in-process in
production) → **Data** (Supabase Postgres, deployed).

The service tier is a real application with real Pydantic contracts, exercised
over HTTP by the test suite. In the deployed app the Streamlit tier imports the
same handlers instead of crossing the network; `SERVICE_MODE` is the only thing
that changes. Separation of concerns is a property of the code, not of the
hosting bill — a repository with clean tiers can move one onto its own host in
an afternoon.

**Why a database tier at all.** A supervised product can be judged from one
number on a held-out set. A reinforcement learning product cannot: the
deliverable is a comparison of learning curves, and every honest claim has the
form *configuration A reached return R in N episodes and configuration B did
not*. That is a query, not a plot. With every episode of every run as a row,
you cannot accidentally report the one seed that worked, and the comparison
tables the rubric asks for fall out of a `GROUP BY`.

## The no-PyTorch-in-serving rule

`import torch` alone occupies roughly **490 MB** of resident memory against
Streamlit Community Cloud's **690 MB** guarantee — most of the budget spent
before a single weight is loaded, on bundled CUDA libraries a CPU application
never calls. The entire deployed stack here measures **82 MB**.

So: train outside the deployed app, export the policy to a NumPy `.npz`, and
evaluate the forward pass in NumPy. For this topic's tabular policy the
"forward pass" is `int(np.argmax(Q[state]))` — one line, no framework.
`tests/test_no_torch.py` asserts `"torch" not in sys.modules` after importing
the app, and `requirements-serve.txt` does not list it. If the guard fails, the
build fails. See [`docs/no-torch.md`](docs/no-torch.md).

**One exception, made deliberately:** `gymnasium` *is* in
`requirements-serve.txt`, because `POST /rollout` runs evaluation episodes
server-side and therefore needs the environment. FrozenLake is toy-text and
pure Python and no renderer is ever instantiated, so the cost is a few hundred
kilobytes. Gymnasium is an environment library, not a training framework — the
rule is about the 490 MB, not about the letters "RL".

## Free-Tier Notes

*Required section. State how this product handles each limit.*

| Limit | Value | How this product handles it |
|---|---|---|
| Streamlit Cloud memory | 690 MB guaranteed | 82 MB measured; no framework in the serving path, and the policy artifact is ~1 KB |
| Streamlit Cloud sleep | after 12 h idle | wakes on first request; note the cold start in your demo video |
| Supabase storage | 500 MB | one row per **episode**, never per step — a 20,000-episode run is ~20k rows ≈ 2 MB; a per-step schema would have been 1M rows |
| Supabase projects | 2 active per person | one project reused across every topic |
| Supabase pause | after 1 week idle | the store returns `degraded=True` instead of raising; the health banner and every tab say so out loud |
| Supabase insert size | large inserts rejected | `train/telemetry.py` buffers and writes in chunks of 500 |
| Python version | 3.11–3.13 | pinned in CI; 3.14 has no Box2D wheels |

## Theoretical Brief

*350–600 words, mirrored in the Streamlit "Concepts" tab — where the LaTeX
already renders. Write it once and keep the two in step.*

Your brief must cover, at minimum: the agent–environment loop; the Bellman
equation and the Q-learning update derived from it; the exploration–exploitation
trade-off and what ε-greedy costs you; and the taxonomy — model-free versus
model-based, value-based versus policy-based — with a real application named
for each quadrant, and a sentence on why *this* problem lands where it does.

## Quantitative Analysis

*Every number reproducible from `experiments` and `episodes`. Name the seeds.
A learning curve with no seed and no row count is not evidence.*

| Configuration | Seeds | Episodes | Mean greedy return (± SE) | Episodes to 0.3 | Query |
|---|---|---|---|---|---|
| Random baseline | | | | n/a | |
| Q-learning, α=?, γ=?, ε=? | | | | | |
| A configuration that failed | | | | | |

Reference query for the greedy column:

```sql
select e.algorithm, e.seed, e.hyperparameters->>'eps_schedule' as eps,
       v.mean_return, v.stderr_return, v.at_training_episode
from experiments e
join evaluations v on v.experiment_id = e.id
where v.at_training_episode = (select max(at_training_episode)
                               from evaluations where experiment_id = e.id)
order by v.mean_return desc;
```

## Engineering Report (≈500 words)

Defend three decisions with evidence from your own tables. These are the
questions the report has to answer — not a list of topics to mention.

**1. The ε schedule.**
- Which schedule string did you run, and what does ε equal at episode 1, at the
  midpoint, and at the last episode? (`python -m train.qlearning` prints this
  before it starts; paste it.)
- How many episodes did the agent spend at your floor value, and what fraction
  of the run is that? Is the floor low enough that the tail of the *training*
  curve is comparable to the *greedy* score, or does a visible gap remain?
- Run one configuration with `--eps-schedule const:0.1` and one with a decay
  that hits its floor in the first 5% of the run. What happened to each, and
  which failure did each one show — never exploring, or never exploiting?

**2. The learning rate.**
- What α did you use, and why is a value appropriate here smaller than one you
  would pick on the *deterministic* lake? (The target `r + γ·max Q(s′,a′)` is a
  noisy sample here and an exact quantity there.)
- Show a run where α was too high. Distinguish *noisy* — jitter around a rising
  trend — from *unstable* — a greedy score that moves up and down between
  consecutive evaluations. Only the second is an α problem.

**3. The stopping criterion.**
- Look at the `evaluations` rows for one seed. At which evaluation did the
  greedy score stop improving by more than its own standard error?
- What would you have needed to observe to justify training longer? State the
  criterion as a rule someone else could apply, not as "it looked flat".

**Two metrics, and what each hides.** Compare at least two — for example mean
greedy return and episodes-to-threshold. Mean return hides *when* the agent got
good, so a fast learner and a slow one that ended in the same place are
indistinguishable. Episodes-to-threshold hides everything after the threshold,
including a policy that peaked and then decayed. Say which of your conclusions
would change if you had only quoted one.

**The run that failed.** Include the learning curve of a hyperparameter setting
that did not converge, and explain how the run-history table made that
comparison cheap — specifically, what you did *not* have to re-run in order to
make the comparison.

## Worldview Reflection (≈200 words)

Reinforcement learning defines "good" as whatever maximises a scalar reward.
This agent has no concept of a hole, of danger, or of care; it has a number,
and the only reason it avoids drowning is that drowning ends the episode before
the number can be collected.

Reflect on how that contrasts with a Christian understanding of reward, which
weighs intention and faith and not outcome alone. Consider how the Fruit of the
Spirit (Galatians 5:22–23) describes outcomes that *flow from intrinsic
character* rather than from optimisation against a target, and how eternal life
is framed as the reward of faithfulness (Matthew 5:12) rather than as a score.

Then be concrete: **what does this mean for how you describe your agent's
"goal" to the non-technical client in the product brief?** A client who hears
"the agent wants to reach the goal" has been told something false in a way that
will cost them later. Write the sentence you would actually say instead.

## AI-Assistance Disclosure

*Required. What did you generate, with which tool, and how did you verify it?
Generated code must be read, understood and tested by you; blind paste-through
is not acceptable.*

## Limitations & Responsible Use

*At least four concrete limitations, each with how you would test whether it
binds in a deployment scenario. Then foreseeable misuse, reward-specification
risk, and the worldview reflection above.*

Three that are true of this product and that you should not have to discover
the hard way:

1. **The policy is a lookup table indexed by square number.** Change the map
   and every entry becomes meaningless — not degraded, meaningless. There is no
   generalisation of any kind. Test: export a table trained on the 8×8 lake,
   serve it, and watch `/act` return confident actions for a 4×4 map's indices.
2. **The value estimate `/act` returns is biased upwards.** Q-learning
   bootstraps from a maximum over four noisy estimates, and the maximum of
   noisy estimates is optimistic. Do not present `value_estimate` to a
   stakeholder as a probability of success.
3. **The agent cannot signal that it is out of its depth.** Every state gets an
   action, including states it visited twice in twenty thousand episodes. Test:
   count visits per state during training and check how many states the
   deployed policy has an opinion about but almost no evidence for.
