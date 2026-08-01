<!-- =========================================================================
     TOPIC 2 TEMPLATE README — "Policy Lab"

     The first fifteen lines are graded. Replace the bracketed placeholders
     before your first commit, not the night before submission. Everything
     that is NOT bracketed is real: the numbers below were produced by
     `python -m train.train` in this repository and are reproducible with it.
     ========================================================================= -->

# Policy Lab — the exact answer and the learned one, side by side

A logistics client has a small routing problem and two consultants giving them
opposite advice. One says the problem should be solved exactly, because the
environment is fully known. The other says the model will never be accurate
enough and the system should learn from what actually happens. **Policy Lab
ships both.** It is a service that holds two agents solving the same 5×5
routing grid — one handed the transition model and computing the optimal policy
exactly, one that never sees the model and estimates it from sampled returns —
behind a single API, over a single schema, with the gap between them measured
as a function of experience and reported with a confidence interval. A
dispatcher opens one link, watches both agents route the same job, and reads a
straight answer to "how much experience would we need before the learned
version is as good as the exact one?"

| | |
|---|---|
| **Live app** | https://[your-app].streamlit.app |
| **Supabase project** | `[your-project-ref]` — **the same project as your first product**; schema in [`db/migrations/`](db/migrations/) |
| **Service tier (local)** | `uvicorn api.main:app --port 8000` → http://127.0.0.1:8000/docs |
| **Service capture** | [link to your screen recording of `POST /act` with both policy sources, and `GET /docs`] |
| **Author** | [name] |

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.11–3.13. Not 3.14.
pip install -r requirements-train.txt                 # laptop / Colab
cp .env.example .env                                  # then fill in your keys

# 1. schema — apply BOTH migrations, in order, to the project you already have
#    (paste db/migrations/001_init.sql then 002_topic2.sql into the SQL editor)
python -m db.seed

# 2. train: plan, learn, and measure the gap. ~17 s, no GPU.
python -m train.train

# 3. serve, and demonstrate it serving
uvicorn api.main:app --reload --port 8000
curl -s localhost:8000/act -H 'content-type: application/json' \
     -d '{"state":[0],"policy_source":"value_iteration"}'
curl -s localhost:8000/act -H 'content-type: application/json' \
     -d '{"state":[0],"policy_source":"monte_carlo"}'

# 4. the user interface
streamlit run ui/app.py

# 5. the gate
pytest -q && ruff check .
```

**Reuse one Supabase project across every topic.** The free tier allows two
active projects per person, and you will want the headroom later in the course.
Migrations are additive and numbered precisely so that this works: `001_init.sql`
is untouched and `002_topic2.sql` adds three columns, two indexes and two views
on top of it.

## What is where

```
api/        the service tier. Owns both policies. NumPy + Gymnasium, no framework.
  main.py     the standing endpoints, plus /value_map and /convergence
  policy.py   the entire serving-side forward pass — an argmax over a table row
ui/         the presentation tier. No policy code, no training code, no writes.
  app.py      Concepts · Value Map · Convergence · Run History · Model Card
  service.py  the one switch between in-process and HTTP service calls
train/      the training tier. Runs on your laptop. Never deployed.
  train.py           the one command that reproduces every number below
  value_iteration.py the PLANNER — reads env.unwrapped.P, never calls step()
  monte_carlo.py     the LEARNER — calls step(), never reads P
  compare.py         the convergence study, the intervals, the equivalence test
  export.py          trained policy -> .npz. The seam between the tiers.
shared/     the contracts. Pydantic models, settings, the data-tier interface,
            and preprocess.py — every transformation, in one importable place.
envs/       gridworld.py: a 5×5 Gymnasium environment with an explicit P.
            Importable, so Topics 3 and 5 reuse it rather than re-implement it.
db/         migrations (001 standing, 002 this topic) and a seed script.
tests/      the standing four, plus the environment, the two algorithms, the
            statistics, and the two-source routing. All service tests go
            through an HTTP test client.
policies/   value_iteration.npz and monte_carlo.npz. Committed — they are what
            you deployed.
reports/    convergence.json. Committed, because GET /convergence falls back to
            it on a clone with no database credentials.
```

## The environment

`GridWorld5x5-v1` — a 5×5 grid, four actions (up, right, down, left), start at
(0, 0), goal at (4, 4) worth +1, pits at (1, 3) and (3, 1) worth −1, a step
cost of −0.02, γ = 0.95, and a **20% slip probability**: the intended direction
happens with probability 0.8 and each perpendicular direction with probability
0.1. Walking into a wall leaves the agent where it was. Episodes are truncated
at 100 steps by `gymnasium.wrappers.TimeLimit`.

Two properties of the environment are load-bearing rather than decorative:

**The transition model is tabulated explicitly** in Gymnasium's toy-text
convention, `P[s][a] -> [(probability, next_state, reward, terminated), ...]`,
and reachable as `env.unwrapped.P`. This is what makes the planner possible at
all, and it is the same layout `FrozenLakeEnv` exposes, so `value_iteration()`
in this repository runs unmodified against FrozenLake. **Read it through
`.unwrapped`** — the `TimeLimit` wrapper forwards `step` and `reset` but not
arbitrary attributes, so `env.P` raises `AttributeError`.

**The slip probability is not zero.** With deterministic transitions every
sampled return equals the true value exactly and a Monte Carlo estimator
converges after one visit per state — the entire convergence study would
measure nothing. Stochasticity is what makes "how much experience?" a question
with an answer.

## Preprocessing: which transformation each algorithm needs, and why

Every transformation between raw environment output and learner input lives in
[`shared/preprocess.py`](shared/preprocess.py) and nowhere else. No `train/`
module reshapes, clips, encodes or normalises inline. The reason is blunt: a
preprocessing step applied at training time and forgotten at serving time is
the single most common cause of a policy that scores well offline and behaves
randomly in the deployed app, and keeping the functions in one importable place
makes that failure a diff rather than a mystery.

**The two algorithms in this repository do not need the same preprocessing, and
that is a design fact worth stating rather than a nuisance to hide.**

| Function | Value iteration | Monte Carlo | Why |
|---|---|---|---|
| `dense_model(P, nS, nA)` | **required** | never | The planner's only input is the model. It turns the nested dict into `T[s,a,s']`, `R[s,a]` and a bootstrap mask `B`, so a whole sweep is one `einsum` instead of 100 Python loops. The learner is not allowed to call it — that is what "model-free" means here. |
| `discounted_returns(rewards, γ)` | never | **required** | The learner's target for step *t* is a function of the whole remainder of the episode, so it cannot be computed until the episode ends. The planner has no episodes. |
| `first_visit_indices(states)` | never | **required** | Decides which of the repeated visits inside one episode contributes a sample. Flipping this one call is the entire difference between first-visit and every-visit MC. |
| `one_hot` / `from_one_hot` | never | not yet | Both learners here are **tabular**: they index a table with an integer, so one-hot encoding would be encoding something nothing consumes. The moment the table becomes a network — Topic 3 — this is the first thing you reach for. |
| `discretise` / `bin_centre` | n/a | n/a | Neither is needed: the observation is already a discrete index. Shipped and tested because the discretisation scheme is the transformation a *continuous* observation would need, and the half-bin round-trip error is the resolution you would be giving up. |
| `clip_reward` | **must not** | **must not** | Rewards here span −1 to +1 already, so clipping is a no-op — and applying it to the planner would silently change the MDP being solved rather than stabilise anything. |
| `normalise_returns` | never | never | A policy-gradient transformation. It makes the update scale-free; neither a table lookup nor an average of returns has an update to rescale. Included for Topic 3. |
| `normalise_observation` | never | never | Requires statistics computed at training time and exported with the weights. Nothing to standardise for a discrete index. |

**The short version.** A value-iteration planner consumes the *model* and needs
no observation processing at all, because it never sees an observation. A Monte
Carlo learner consumes *trajectories* and needs the two transformations that
only make sense on a completed episode — folding a reward sequence into return
targets, and deciding which visits count. Neither needs an encoder, because
both are tabular. The one thing they do share, `clip_reward`, is the one thing
neither should use. A single "preprocessing pipeline" applied to both would
have to be a no-op to be correct.

Every function is unit-tested in [`tests/test_preprocess.py`](tests/test_preprocess.py),
and the pairs that are supposed to invert each other are tested as round trips:
a transform whose inverse you have never run is a transform whose convention
you do not actually know.

## Build-step checklist

- [x] Fork the template. **Reuse the existing Supabase project** — do not
      provision a second one.
- [x] `envs/gridworld.py`: a 5×5 gridworld as a proper Gymnasium environment,
      four actions, an explicit transition-probability matrix at
      `env.unwrapped.P`, and a configurable reward function (`RewardSpec`, or
      pass your own `reward_fn`). Shipped as an importable module so Topics 3
      and 5 can `from envs import make_env`.
- [x] `shared/preprocess.py`: one-hot encoding, a discretisation scheme,
      reward clipping and return normalisation, plus the two transformations
      this topic actually needs. Every function unit-tested; every learner
      imports from here rather than transforming inline. Rationale table above.
- [x] `train/value_iteration.py`: Bellman optimality backups; persists both the
      optimal value function and the optimal policy; logs one row per sweep to
      `episodes` with the Bellman residual.
- [x] `train/monte_carlo.py`: first-visit MC policy evaluation and MC control
      with exploring starts. Every episode logged.
- [x] `train/compare.py`: 10 seeds × 6 budgets, RMSE against the exact
      solution, written to `evaluations`.
- [x] Statistical analysis: mean RMSE with a 95% confidence interval at each
      budget, and the episode count at which the estimate becomes
      statistically indistinguishable from exact at the 5% level, with the test
      named and its assumptions stated.
- [x] FastAPI service: `POST /act` (with a `policy_source` of
      `"value_iteration"` or `"monte_carlo"`), `POST /rollout`, `GET /runs`,
      `GET /policies`, `GET /healthz`, `GET /version` — plus `/value_map` and
      `/convergence` for the two topic-specific tabs.
- [x] Streamlit frontend: Concepts, Value Map, Convergence, Run History, Model
      Card.
- [ ] Engineering report (≈500 words) — skeleton below; **finish it**.
- [x] Pytest, including value iteration recovering a known-optimal policy on a
      hand-solvable 2×2 case, and the standing four from the quality bar.
- [ ] Deploy the Streamlit app; verify the Supabase project is active; record
      the service tier running locally under uvicorn.

## Architecture — two clouds, three tiers

**Presentation** (Streamlit Community Cloud, deployed) → **Service** (FastAPI,
in this repository, run under uvicorn locally and imported in-process in
production) → **Data** (Supabase Postgres, deployed).

The service tier is a real application with real Pydantic contracts, exercised
over HTTP by the test suite. In the deployed app the Streamlit tier imports the
same handlers instead of crossing the network; `SERVICE_MODE` is the only thing
that changes. Separation of concerns is a property of the code, not of the
hosting bill.

**How two agents fit one contract.** `POST /act` takes an optional
`policy_source` with a closed set of values, and the service maps it onto the
`policy_name` the base contract already understands through one dictionary in
`shared/schemas.py`. What that deliberately is *not*: a second endpoint, a
second request model, or a field whose meaning changes depending on another
field. Forking `/act` into `/act_vi` and `/act_mc` would have been quicker to
write and would have doubled the surface every later change has to be applied
to twice.

**Why a database tier at all.** The deliverable in reinforcement learning is a
comparison, and every honest claim takes the form *configuration A reached
return R in N episodes and configuration B did not*. That is a query, not a
plot. With every sweep and every episode as a row, you cannot accidentally
report the one seed that worked, and the tables below fall out of a `GROUP BY`.

## The no-PyTorch-in-serving rule

`import torch` alone occupies roughly **490 MB** of resident memory against
Streamlit Community Cloud's **690 MB** guarantee — most of the budget spent
before a single weight is loaded, on bundled CUDA libraries a CPU application
never calls. The entire deployed stack here measures about **87 MB**: the base stack's 82
MB plus a measured **4.9 MB** for Gymnasium.

Nothing in this topic needs a framework anyway: both policies are 25×4 tables
and the forward pass is `int(np.argmax(Q[state]))`. `tests/test_no_torch.py`
asserts `"torch" not in sys.modules` after importing the app, CI runs the same
check as its own step, and `requirements-serve.txt` does not list it. See
[`docs/no-torch.md`](docs/no-torch.md).

**Two deliberate dependency decisions specific to this topic.** Gymnasium *is*
in `requirements-serve.txt`, because `POST /rollout` runs evaluation episodes
inside the deployed process and therefore has to build the environment;
Gymnasium is 2.1 MB of pure Python over NumPy with no compiled extension —
measured at **+4.9 MB resident** and +66 ms of import time on top of
`numpy` and `fastapi` — and
the rule this course enforces is "the artifact you deploy is not the training
graph", not "few dependencies". SciPy is *not*: it is ~90 MB, it is used only
by `train/compare.py` for the t distribution, and that module falls back to a
self-contained implementation when it is absent — which is the path CI takes,
since CI installs the serving requirements only. The two implementations are
checked against each other in `tests/test_topic2.py`, because a hand-rolled
statistical function nobody validated is worse than no statistics at all.

## Free-Tier Notes

| Limit | Value | How this product handles it |
|---|---|---|
| Streamlit Cloud memory | 690 MB guaranteed | ~87 MB measured (82 MB base stack + 4.9 MB Gymnasium); no framework in the serving path, and the two policy artifacts are 933 and 1,048 bytes |
| Streamlit Cloud sleep | after 12 h idle | wakes on first request; note the cold start in your demo |
| Supabase storage | 500 MB | one row per sweep (45) and one per episode; a full `train.train` run writes ~110k rows ≈ 13 MB. The convergence study adds 60 rows, not 600k, because it snapshots at six budgets rather than logging every evaluation episode |
| Supabase projects | **2 active per person** | **one project reused across every topic**; `002_topic2.sql` is additive on top of `001_init.sql` |
| Supabase pause | after 1 week idle | the UI degrades visibly (health banner), and `GET /convergence` falls back to the committed `reports/convergence.json` while flagging `degraded: true` |
| Python version | 3.11–3.13 | pinned in CI; 3.14 has no Box2D wheels for later topics |

## Theoretical Brief

*Mirrored in the Streamlit "Concepts" tab, which carries the full derivations.
Summary here.*

The **Bellman expectation equation** falls out of one substitution. The value of
a state under a policy is the expected discounted return from it,
*v*<sub>π</sub>(*s*) = E<sub>π</sub>[*G*<sub>t</sub> | *S*<sub>t</sub> = *s*].
Split the first reward off the tail, *G*<sub>t</sub> = *R*<sub>t+1</sub> + γ*G*<sub>t+1</sub>,
substitute, and average over the two things that can happen on the first step —
which action the policy chooses and where the environment puts you:

*v*<sub>π</sub>(*s*) = Σ<sub>a</sub> π(*a*|*s*) Σ<sub>s′</sub> *p*(*s′*|*s*,*a*) [ *r*(*s*,*a*,*s′*) + γ *v*<sub>π</sub>(*s′*) ]

Replace the average over actions with a maximum and it becomes the **Bellman
optimality** equation, which `train/value_iteration.py` iterates to a fixed
point. That it converges at all is a contraction argument: the backup shrinks
the max-norm distance to the true values by a factor of at least γ each sweep,
which is also where the stopping rule comes from — a residual below θ
guarantees the answer is within θγ/(1−γ) of exact. This run stopped at a
residual of 7.19 × 10⁻¹¹, an error bound of 1.37 × 10⁻⁹.

The **policy improvement theorem** is what licenses the other agent. If
*q*<sub>π</sub>(*s*, *a*) > *v*<sub>π</sub>(*s*) for some action *a*, the policy
that takes *a* in *s* and behaves as before elsewhere is at least as good in
*every* state — so "act greedily with respect to your current value estimate"
is a step that cannot make things worse, however rough that estimate is. The
ordering is only *partial*, though: two policies can each be better than the
other from different states, which is why every value below is quoted at a
named state rather than as one score.

The learner has to solve a problem the planner does not: it can only estimate
the value of a state it has visited. Under a deterministic policy from a fixed
depot most of this grid is never seen, so `train/monte_carlo.py` uses
**exploring starts** — a uniformly chosen initial state, and for control a
uniformly chosen first action. That is a real assumption, not a free lunch: it
needs an environment you can reset into any state, which a simulator gives you
and a forklift does not. The deployable alternative is a **soft policy** such
as ε-greedy, which buys coverage without special resets at the cost of
evaluating a policy that differs from the one you intend to ship by O(ε).

**First-visit** and **every-visit** Monte Carlo both converge to *v*<sub>π</sub>;
they differ in which visits count when one episode passes through a state
twice. First-visit samples are independent across episodes, so the usual
standard-error formula is honest. Every-visit samples are correlated within an
episode — the second visit's return is a suffix of the first's — so the same
formula understates the uncertainty. **Importance sampling** extends all of
this off-policy, re-weighting each return by ∏ π(*A*<sub>k</sub>|*S*<sub>k</sub>) / *b*(*A*<sub>k</sub>|*S*<sub>k</sub>);
the ordinary estimator is unbiased with unbounded variance, the weighted one
biased and consistent with far less. For someone else to recompute that from
your data alone, the *behaviour* probabilities must have been recorded at the
time — which makes it a schema requirement, not an algorithmic one.

## Quantitative Analysis

All numbers below are reproducible with `python -m train.train` on this
repository at `GridWorld5x5-v1`, γ = 0.95.

### The two agents

| | Planner (value iteration) | Learner (Monte Carlo, exploring starts) |
|---|---|---|
| Input | `env.unwrapped.P` — the model | `reset()` and `step()` only |
| Budget | 45 sweeps | 100,000 control episodes (seed 0) |
| Wall clock | 0.85 ms | 6.1 s |
| Seeds | none — a planner has no random stream | 1 for the artifact; 10 for the study below |
| *V*(depot) | **0.317579** (exact) | **0.309324** — 2.60% short |
| Largest per-state value loss | 0 by definition | 0.03045 |
| Arrow agreement with the plan | — | 17 of 22 non-terminal cells |
| Artifact | 933 B | 1,048 B |

The five disagreeing cells are worth reading rather than summarising. **Three
of them are exact ties**: on the diagonal cells (1,1), (2,2) and (3,3), "right"
and "down" have *identical* action values in *Q*\*, so the learner is not wrong
there, it broke a tie the other way. The remaining two, (0,1) and (0,3), differ
by 0.0059 and 0.0066 in *Q*\* — under 1% of the value range.

That gap does not close with a bigger budget: the greedy policy is **identical
at 20k, 50k, 100k and 200k episodes**, and only at 400k does the loss drop
(to 0.007). The reason is computable. Exploring starts hands each of the 88
state-action pairs about 1/100 of the episodes as a first visit, so at 100k
episodes those two cells have ~1,100 samples each of a return whose spread is
an order of magnitude larger than the 0.006 difference being resolved. Monte
Carlo control plateaus here for a statistical reason, not because the
implementation is broken — and "my agent stopped improving" is a claim you
should always be able to decompose like this.

### Convergence: RMSE against the exact solution

First-visit Monte Carlo evaluation of the optimal policy, scored against *V*\*
over the 23 non-terminal states, **10 independent seeds (0–9)**, one run per
seed snapshotted at each budget.

| Episodes | Mean RMSE | SD across seeds | 95% CI (t, 9 df) | Upper 95% bound | *p* vs δ | Indistinguishable? |
|---|---|---|---|---|---|---|
| 100 | 0.16375 | 0.04126 | [0.13423, 0.19327] | 0.18767 | 1.00 | no |
| 300 | 0.10173 | 0.04375 | [0.07043, 0.13303] | 0.12709 | 1.00 | no |
| 1,000 | 0.05458 | 0.01717 | [0.04230, 0.06687] | 0.06454 | 1.00 | no |
| 3,000 | 0.03193 | 0.00589 | [0.02772, 0.03615] | 0.03535 | 1.00 | no |
| 10,000 | 0.01632 | 0.00386 | [0.01356, 0.01909] | 0.01856 | 0.98 | no |
| 30,000 | **0.00916** | 0.00229 | [0.00753, 0.01080] | 0.01049 | **1.4 × 10⁻⁴** | **yes** |

The error falls as roughly 1/√n, as theory says it must: a hundredfold increase
in episodes (100 → 10,000) buys a tenfold reduction in RMSE (0.164 → 0.016).

**Reproduce any row.** `python -m train.compare`, or from the data tier:

```sql
select at_training_episode as episodes,
       count(*) as seeds,
       avg(rmse) as mean_rmse,
       stddev_samp(rmse) as sd_rmse
from evaluations
where metric = 'value_rmse'
group by at_training_episode
order by at_training_episode;      -- or: select * from mc_convergence;
```

### The statistical claim, and what it rests on

**Result. From 30,000 episodes onward the Monte Carlo estimate is statistically
indistinguishable from the exact solution at the 5% level** (mean RMSE 0.00916,
one-sided upper 95% bound 0.01049, δ = 0.01332, *p* = 1.4 × 10⁻⁴, 10 seeds).
At 10,000 episodes it is not (upper bound 0.01856, *p* = 0.98).

**Test.** A one-sided one-sample Student *t*-test on the ten per-seed RMSE
values against a pre-declared equivalence margin δ — the upper arm of a two
one-sided tests (TOST) equivalence procedure. The lower arm is vacuous, since
RMSE is non-negative by construction.

- H₀: mean RMSE ≥ δ (still materially different from exact)
- H₁: mean RMSE < δ (indistinguishable, at tolerance δ)
- reject H₀ when *t* = (x̄ − δ)/(*s*/√10) < −*t*<sub>0.95, 9</sub> = −1.833

**Why an equivalence test rather than a *t*-test against zero.** Failing to
reject equality is not evidence of equality: a test against 0 would "fail to
find a difference" simply because the study was small, and with enough seeds it
would find one at any budget, because the true RMSE is never exactly zero. The
claim has to be *"closer than δ"*, and δ has to come from the problem.

**δ = 2% of the span of *V*\* over non-terminal states = 0.01332**, fixed before
the study ran. The client's routing decisions are made by comparing state
values to one another, and a difference of 2% of the total spread cannot flip a
routing decision anywhere on this grid. Quoting δ in raw reward units instead
would make it meaningless the moment the reward specification changed.

**Assumptions, stated because they are how the result could be wrong.**

1. The ten per-seed RMSE values are **independent** — guaranteed by
   construction (each seed drives its own generator and its own environment
   stream), not inferred from the data.
2. They are **approximately normal across seeds**. This is the weakest
   assumption: RMSE is non-negative and right-skewed. With 10 seeds the *t*-test
   tolerates mild skew; if you doubt it, the drop-in replacement is a bootstrap
   percentile interval over the same ten values.
3. The **variance is estimated from the same ten values**, which is why this is
   a *t*-test and not a *z*-test. At 9 degrees of freedom the critical value is
   1.833 rather than 1.645 — an 11% wider interval that a normal approximation
   would quietly omit. (The Streamlit chart draws ±1.96 SE because the service
   tier has no *t* distribution; it says so on the tab.)
4. **δ was fixed in advance.** An equivalence margin chosen after seeing the
   curve is not a hypothesis test, it is a description.
5. Every seed ran the **same budget grid** and no seed was dropped. Snapshots
   within one seed come from a single run, so points on one curve are *not*
   independent of each other; only the across-seed comparison at a fixed budget
   is.
6. The test is applied at each budget and the smallest passing budget reported.
   That is one claim about where a monotone curve crosses a line, not six
   independent discoveries — no multiplicity correction is applied, and none
   would be meaningful, but a reader is entitled to know which it is.

## Engineering report (≈500 words)

*Skeleton with the technical content filled in. Finish the client-facing
argument in your own words, using numbers from your own `evaluations` table.*

### Linear programming as a third route

The optimal value function is not only the fixed point of the Bellman
optimality operator — it is also the solution of a linear program. *V*\* is the
**smallest** value function that is feasible for every Bellman inequality, so:

> minimise Σ<sub>s</sub> μ(*s*) *V*(*s*)
> subject to *V*(*s*) ≥ Σ<sub>s′</sub> *p*(*s′*|*s*,*a*) [ *r*(*s*,*a*,*s′*) + γ*V*(*s′*) ] for every (*s*, *a*)

with any strictly positive state weighting μ. Here that is 25 free variables and
100 constraints, solvable by `scipy.optimize.linprog` in a few milliseconds; the
optimal policy is read off the constraints that are *tight* at the solution, or
equivalently from the dual, whose variables are the discounted state-visitation
frequencies. The dual is the more interesting object: it says the LP is choosing
an occupancy measure directly rather than a policy, which is the same idea that
reappears in Topic 3 as the policy-gradient objective.

**Computational profile, against the two dynamic-programming methods.**

| | per iteration | iterations | scaling | in practice |
|---|---|---|---|---|
| Value iteration | O(\|S\|²\|A\|) | O( log(1/ε) / log(1/γ) ) | trivially parallel; the cheapest per step | 45 sweeps, 0.85 ms here |
| Policy iteration | O(\|S\|³) for the exact evaluation solve, plus O(\|S\|²\|A\|) to improve | very few — typically 3–10, and **finite**: it terminates exactly | the solve dominates and does not parallelise well | 8 iterations here |
| Linear programming | — | — | polynomial in \|S\|\|A\| via interior point; simplex is exponential in the worst case | competitive at 25 states, uncompetitive at 10⁵ |

The honest summary is that LP is the theoretically satisfying route and rarely
the practical one. Its advantages are real but specific: it terminates exactly
rather than asymptotically, it accepts *additional linear constraints* (a
budget, a safety limit, a service-level agreement) that neither DP method can
express at all, and it hands you the dual for free. Its disadvantage is that
the constraint matrix is |S||A| × |S| and dense in the successor states, so it
stops fitting in memory long before value iteration stops being fast. Value
iteration wins on scale and on parallelism; policy iteration wins on iteration
count when |S| is small enough that the exact solve is cheap; LP wins when the
problem is genuinely a *constrained* one.

### When to plan and when to learn

*Your argument, with numbers from your own evaluations table. The shape of it:
the planner is exact and free (3 ms) whenever a trustworthy model exists — but
every number it produces is conditional on that model. The learner needed
30,000 episodes to reach a value estimate indistinguishable from exact at the
2%-of-range tolerance, and its control policy plateaus 2.6% below optimal. So
the question is not which method is better; it is how much you trust the model,
and the crossover is a number you can compute: perturb `slip` by ±0.05, re-plan,
evaluate the old plan under the new dynamics, and compare that loss with the
0.00916 the learner reaches after 30,000 episodes.*

## AI-Assistance Disclosure

*Required. What did you generate, with which tool, and how did you verify it?
Generated code must be read, understood and tested by you; blind paste-through
is not acceptable.*

## Limitations & Responsible Use

*At least four concrete limitations, each with how you would test whether it
binds. The Model Card tab in the Streamlit app carries the full version; keep
the two in sync.*

1. **Exploring starts is an assumption a deployed system cannot make.** It
   needs an environment you can reset into any state. Test: re-run with an
   ε-soft policy from the fixed depot and compare both the RMSE curve and the
   visit count of the least-visited cell.
2. **The 100-step truncation biases returns downward** for any policy that
   fails to terminate. Measured truncation rate during control: 0.063%. Test:
   re-run at a larger cap and check whether the value function moves.
3. **Every planner number is conditional on the model being correct** — which
   is exactly the second consultant's objection, and the reason this product
   exists. Test: perturb `slip` by ±0.05, re-plan, and evaluate the *old* plan
   under the *new* dynamics.
4. **The reward specification encodes an unexamined trade-off.** A step cost of
   −0.02 against a pit penalty of −1 says a fifty-step detour is worth exactly
   one accident. Nobody said that out loud; it fell out of two numbers chosen
   for convenience. Test: sweep `pit_penalty` and record where the optimal
   policy changes.

Then: foreseeable misuse, reward-specification risk, and the worldview
reflection your topic calls for.
