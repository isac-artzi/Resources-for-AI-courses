<!-- =========================================================================
     PRODUCT 4 README — "Control Bench"

     The first fifteen lines are graded. Replace the bracketed placeholders
     before your first commit — not the night before submission. Everything
     outside brackets is yours to keep or rewrite; the section headings are
     rubric-aligned and should stay.
     ========================================================================= -->

# Control Bench — [one line a stakeholder would recognise, e.g. "three control agents, one contract, and the evidence to choose between them"]

[**One paragraph, before any code or commands.** Your organisation is
standardising on a single control-agent service and needs a bake-off. This
service hosts three trained agents behind one API contract — an advantage
actor-critic agent on a discrete task, a proximal-policy-optimisation agent on a
harder discrete task, and a soft actor-critic agent on a continuous task — and
lets a stakeholder run any of them, compare their sample efficiency on equal
terms, and see what happens to the continuous agent when its exploration
temperature is changed. Write your own version of this paragraph for the client,
not for the grader. If it only makes sense to someone who has read the
assignment, rewrite it.]

| | |
|---|---|
| **Live app** | https://[your-app].streamlit.app |
| **Supabase project** | `[your-project-ref]` — schema in [`db/migrations/`](db/migrations/) |
| **Service tier (local)** | `uvicorn api.main:app --port 8000` → http://127.0.0.1:8000/docs |
| **Service capture** | [link to your screen recording of the service serving `POST /act` and `GET /docs`] |
| **Environments** | `CartPole-v1`, `Acrobot-v1`, `Pendulum-v1` (`gymnasium[classic-control]`) |
| **Deployed policies** | `a2c_cartpole` (4→64→64→2, softmax) · `ppo_acrobot` (6→64→64→3, softmax) · `sac_pendulum` (3→256→256→2, tanh-squashed Gaussian, ±2 torque) |
| **Author** | [name] |

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.11–3.13. Not 3.14.
pip install -r requirements-train.txt                 # laptop / Colab
cp .env.example .env                                  # then fill in your keys

# 1. apply the schema — paste BOTH migrations into the Supabase SQL editor,
#    001 first. 002 adds entropy_sweep and policy_updates (which carries the KL).
python -m db.seed

# 2. the three deployable agents (writes telemetry as it goes, then exports)
python -m train.train                  # add --quick for a two-minute smoke test

# 3. the evidence. See "Budgets" below before you run these.
python -m train.compare --steps 60000 --seeds 3      # A2C vs PPO, matched budget
python -m train.entropy_sweep --seeds 3               # SAC at three temperatures

# 4. serve, and demonstrate it serving
uvicorn api.main:app --reload --port 8000

# 5. the user interface
streamlit run ui/app.py

# 6. the gate
pytest -q && ruff check .
```

### Budgets

| | command | wall clock (CPU) | what it is for |
|---|---|---|---|
| sandbox | `python -m train.train --quick` | ~1.5 min | finding bugs cheaply; **not** a result |
| deployable | `python -m train.train` | ~4 min | the three artifacts you serve |
| bake-off | `python -m train.compare --steps 60000 --seeds 3` | ~10 min | 12 runs: 2 algorithms × 2 envs × 3 seeds |
| entropy sweep | `python -m train.entropy_sweep --seeds 3` | ~35 min | 9 runs: 3 temperatures × 3 seeds, 20,000 steps each |

`--quick` exists so that a fresh fork finishes in a lunch break and you discover
your bugs before you spend half an hour on them. Every number it produces is a
smoke test. Say in your Quantitative Analysis which command produced the numbers
you are quoting.

**SAC is the expensive one and it is expensive for a reason worth knowing.** It
takes one gradient update per environment step, where PPO takes about forty per
thousand. It is therefore far slower per environment step and far cheaper per
unit of progress — Pendulum is usable after 10,000 steps, which is fifty
episodes. Do not compare SAC's wall clock against PPO's and call it sample
efficiency; they are different axes, and every comparison in this product is on
environment steps.

## What is where

```
api/        the service tier. Owns the policies. Imports NumPy, never a framework.
  main.py     the standing endpoints, plus /episodes, /policy_updates, /entropy_sweep
  forward.py  THE NUMPY FORWARD PASS — softmax head and squashed-Gaussian head
  policy.py   artifact discovery, checksums, env_id routing, the act() contract
ui/         the presentation tier. No policy code, no training code, no writes.
  app.py      Concepts / Bake-Off / Entropy Sweep / Play / Run History / Model Card
  service.py  the one switch between in-process and HTTP service calls
train/      the training tier. Imports torch. Never deployed, never imported by api/.
  nets.py         the shared networks: categorical actor, critic, squashed Gaussian, twin Q
  onpolicy.py     the step collector A2C and PPO share — the matched-budget machinery
  a2c.py          advantage actor-critic on CartPole-v1
  ppo.py          PPO with a clipped surrogate on Acrobot-v1, logging per-update KL
  sac.py          soft actor-critic on Pendulum-v1, with automatic entropy tuning
  compare.py      A2C vs PPO at matched environment-step budgets, >= 3 seeds
  entropy_sweep.py  SAC under alpha = 0.5, alpha = 0.01 and auto, >= 3 seeds
  train.py        one command: train all three, evaluate, export, register
  export.py       PyTorch -> .npz. The seam between the tiers.
shared/     the contracts. Pydantic models, settings, the data-tier interface,
            and preprocess.py — every transformation, in one importable place.
envs/       the three classic-control environments, exposed as make_env(env_id).
db/         migrations 001 (standing schema) and 002 (entropy_sweep, policy_updates).
tests/      the standing four, plus THREE equivalence tests and the 422 matrix.
policies/   exported artifacts. Committed, because they are what you deployed.
```

## Build-step checklist

Tick these off in order. Each one is a rubric line.

- [ ] **Fork the template, reuse your Supabase project.** Apply `001_init.sql`,
      then `002_topic4.sql` — which adds `entropy_sweep` (experiment_id, alpha
      setting, mode, mean return, return standard deviation, mean policy
      entropy, plus convergence speed and stability), `policy_updates` with the
      per-update `kl_divergence` column, the indexes, and the three views the UI
      reads.
- [ ] **Instantiate all three environments** from `gymnasium[classic-control]`
      in `envs/make_env(env_id)`. Never `gymnasium[all]`, and never LunarLander.
- [ ] **A2C** in `train/a2c.py`, trained on CartPole-v1. One `episodes` row per
      episode; save the learning curve.
- [ ] **PPO with a clipped surrogate** in `train/ppo.py`, trained on Acrobot-v1.
      One `episodes` row per episode AND the mean KL divergence per update, so
      the trust-region behaviour is visible in data and not only in prose.
- [ ] **Compare A2C and PPO** on sample efficiency at matched environment-step
      budgets, ≥ 3 seeds each (`train/compare.py`). Write the 150–250 word
      summary of what the comparison shows **and what it does not**.
- [ ] **SAC** in `train/sac.py`, trained on Pendulum-v1. Log episode returns,
      critic loss and mean policy entropy.
- [ ] **The entropy sweep**: retrain SAC under α = 0.5, α = 0.01 and automatic
      tuning, ≥ 3 seeds each — at ≥ 20,000 steps, so that a run has at least
      100 episodes and `mean_return_last_100` means what its name says. Report
      convergence speed (episodes to a stated
      threshold), final performance (mean return over the last 100 episodes) and
      training stability (variance across seeds) in the `entropy_sweep` table,
      rendered in Streamlit. Write the 100–250 word interpretation.
- [ ] **Export all three policies** to `.npz` and register them in `policies`.
      The SAC actor outputs a mean and a log standard deviation followed by a
      tanh squash; reproducing that in NumPy is three additional lines and is
      part of the exercise.
- [ ] **`POST /act` rejects a dimensionality mismatch with a clear 422**, not a
      stack trace. CartPole is 4-dimensional, Acrobot 6 and Pendulum 3, and all
      three answer at the same URL — this is a real hazard here, not a drill.
- [ ] **The three equivalence tests pass** — `tests/test_equivalence.py`, not a
      manual check. Quote all three measured differences below.
- [ ] **Comparative analysis** (≈500 words) in this README, every empirical
      claim citing a row in your own tables.
- [ ] **Limitations analysis** (≈300 words), a *separate* section from the one
      above.
- [ ] **FastAPI service**: `POST /act`, `POST /rollout`, `GET /runs`,
      `GET /policies`, `GET /healthz`, `GET /version`. Run it under uvicorn and
      record it.
- [ ] **Streamlit frontend** with the six tabs.
- [ ] **Deploy** the Streamlit app, confirm the Supabase project is active, and
      link the service tier running locally.

## Architecture — two clouds, three tiers

**Presentation** (Streamlit Community Cloud, deployed) → **Service** (FastAPI, in
this repository, run under uvicorn locally and imported in-process in
production) → **Data** (Supabase Postgres, deployed).

The service tier is a real application with real Pydantic contracts, exercised
over HTTP by the test suite. In the deployed app the Streamlit tier imports the
same handlers instead of crossing the network; `SERVICE_MODE` is the only thing
that changes.

**Why a database tier at all.** A supervised product can be judged from one
number on a held-out set. A reinforcement learning product cannot: the
deliverable is a comparison of learning curves, and every honest claim has the
form *configuration A reached return R in N steps and configuration B did not*.
That is a query, not a plot. This topic sharpens the point twice over — the
bake-off claim is about twelve runs at a matched step budget, and the entropy
claim is about the *variance across seeds* of nine more. Neither is a number you
can eyeball off a chart.

**One contract, three agents.** The single design decision worth defending in
your write-up is that the environment is a property of the ARTIFACT, not of the
request. `train/export.py` writes `env_id` into the `.npz`; `api/policy.py` reads
it back; `POST /rollout` constructs that environment and no other. A caller picks
a policy; the policy picks its world. The alternative — letting the caller name
an environment — makes it possible to evaluate the Pendulum actor on CartPole,
which fails deep inside Gymnasium with a shape error rather than at the boundary
with a message.

## The no-PyTorch-in-serving rule

`import torch` alone occupies roughly **490 MB** of resident memory against
Streamlit Community Cloud's **690 MB** guarantee. The entire deployed stack here
measures **82 MB**.

This topic adds a second serving head — the tanh-squashed Gaussian — and with it
three new ways for the NumPy forward pass to be silently wrong: the mean/log-σ
split, the log-σ clamp, and the action scale. All three are covered by
`tests/test_equivalence.py`, which compares the pre-squash mean, the clamped
log σ and the final action *separately*, because which one disagrees tells you
which mistake you made. See [`docs/no-torch.md`](docs/no-torch.md).

`gymnasium[classic-control]` is in `requirements-serve.txt` — a few megabytes,
and genuinely a serving dependency because `POST /rollout` runs episodes
server-side. `torch` is not, and never will be.

## Free-Tier Notes

*Required section. State how this product handles each limit.*

| Limit | Value | How this product handles it |
|---|---|---|
| Streamlit Cloud memory | 690 MB guaranteed | [82 MB measured; NumPy forward pass, no framework in the serving path] |
| Streamlit Cloud sleep | after 12 h idle | [wakes on first request; note the cold start in your demo] |
| Supabase storage | 500 MB | [~12k episode rows + ~3k policy_updates rows for the full bake-off and sweep ≈ [M] MB. SAC's per-update rows are SUMMARISED every 50 updates rather than stored one per update — see `update_log_every` in train/sac.py — because one-per-step would be 100k rows per sweep] |
| Supabase projects | 2 active per person | [one project reused across every topic] |
| Supabase pause | after 1 week idle | [UI degrades visibly; see the health banner] |
| Python version | 3.11–3.13 | [pinned in CI; 3.14 has no Box2D wheels — which is also why LunarLander is absent from `envs/`] |
| Artifact size | reviewable in a PR | [~19 KB each for A2C and PPO, ~250 KB for SAC (256-wide hidden layers). The SAC **critics are not exported** — they are training-time objects and would triple the bundle] |

## Theoretical Brief

*350–600 words, mirrored in the Streamlit "Concepts" tab, which carries the full
derivations. Summarise here; do not duplicate at length.*

An **actor-critic** method keeps both objects a pure method throws away. Pure
value-based learning (Topics 1–2) learns `Q` and takes an argmax, which needs a
discrete action space and offers no natural stochasticity. Pure policy-gradient
learning (Topic 3) learns `π` directly and pays in variance, because the weight
on each `log π(a|s)` is a Monte Carlo return: a sum of hundreds of noisy rewards.
Actor-critic replaces that sum with a learned `V_φ(s)` and weights by the
**advantage** `A(s,a) = Q(s,a) − V(s)`. Subtracting any function of the state
leaves the gradient unbiased — `E_a[∇log π(a|s)·b(s)] = b(s)·∇Σ_a π(a|s) = 0` —
and changes only its variance. Intuitively: reinforce actions that beat their own
state's average, rather than every action taken somewhere good.

The **performance difference lemma**,
`J(π′) − J(π) = (1−γ)⁻¹ E_{s∼d_{π′}} E_{a∼π′}[A_π(s,a)]`, explains why that is
not the end of the story. The advantage is the old policy's; the state
distribution is the new policy's. Every method in this topic approximates
`d_{π′} ≈ d_π`, and that approximation is good only while the two policies are
close — which is *why* a trust region exists at all.

**PPO** enforces the region by making the objective flat: it clips the
likelihood ratio into `[1−ε, 1+ε]` and takes the pessimistic branch, so an update
that would push a good action's probability past the ceiling earns nothing and
receives zero gradient. **TRPO** enforces it as a genuine constraint,
`E[KL(π_old ‖ π_θ)] ≤ δ`, bought with a conjugate-gradient solve and a line
search. The practical difference is that PPO's clip does *not* bound the KL — it
only makes large moves unrewarding — so whether the KL stayed small is an
empirical question. `policy_updates.kl_divergence` is where this product answers
it.

**SAC** changes the objective rather than the loss:
`J(π) = E[Σ γ^t (r_t + α·H(π(·|s_t)))]`. Entropy is now part of the return, so
the Bellman backup becomes soft — `V(s) = E_{a∼π}[Q(s,a) − α log π(a|s)]`, whose
optimum is `π*(a|s) ∝ exp(Q(s,a)/α)` and whose value is a log-sum-exp rather than
a max. As α → 0 that recovers greedy Q-learning; as α → ∞ it recovers a uniform
policy. The optimal policy of *this* objective is stochastic by construction.

The two regularisers are therefore **not the same thing**. The entropy term
measures randomness against the uniform distribution and lives inside the reward,
so it changes what the optimum *is*. The trust region measures movement against
the previous policy and lives outside the objective, so it changes only the path
taken to an unchanged optimum. Both are KL penalties against a reference
distribution — uniform in one case, `π_old` in the other — which is precisely the
identity Topic 6 inverts.

[Write your own 350–600 words. A brief that is a paraphrase of this one is not a
brief.]

## Quantitative Analysis

*Every number must be reproducible from `experiments`, `episodes`,
`policy_updates` and `entropy_sweep`, and the seeds must be named. A learning
curve with no seed and no row count is not evidence.*

### The three deployed agents

Quote **your** numbers. Reference values from this template, one seed each at
the deployable budget (`python -m train.train`, seed 0):

| Agent | Env | Steps | Episodes | Train mean, last 100 | Random baseline | Deterministic eval (± SE) | Artifact |
|---|---|---|---|---|---|---|---|
| A2C | CartPole-v1 | 60,000 | 496 | 210.6 | ≈ 22 | 500.0 ± 0.0 | 18.8 KB |
| PPO | Acrobot-v1 | 60,000 | 483 | −87.1 | ≈ −500 | −86.6 ± 4.4 | 19.5 KB |
| SAC | Pendulum-v1 | 15,000 | 75 | −454.2 (−132.9 over the last 10) | ≈ −1200 | −130.4 ± 40.6 | 253 KB |

Two readings that the table alone will not give a stakeholder, and which you
should write out:

* **On Acrobot, −500 is a floor rather than a mean.** The reward is −1 per step
  until the goal height is reached; a random policy essentially never reaches it,
  so nearly every random episode truncates at exactly −500. An agent at −450 has
  genuinely learned something even though it has moved only 10% up the axis.
* **SAC's 100-episode mean is dragged down by design.** Its first 1,000 steps are
  uniformly random actions (`start_steps`), so on a 75-episode run five of those
  episodes are not the agent at all. Quote the window with the number, always.

### A2C vs PPO at a matched environment-step budget

```sql
-- the comparison, as a query rather than as a screenshot
select e.algorithm,
       e.env_id,
       count(distinct e.seed)                        as seeds,
       avg(ep."return") filter (where ep.env_steps > 50000) as mean_return_late,
       max(ep.env_steps)                             as budget
from episodes ep
join experiments e on e.id = ep.experiment_id
where e.algorithm in ('a2c', 'ppo')
group by 1, 2
order by 2, 1;
```

Reference values from this template (`python -m train.compare --steps 60000
--seeds 3`, seeds 0–2, threshold 195 on CartPole and −100 on Acrobot):

| Env | Algorithm | Seeds | Env steps | Gradient steps | Mean return, last 100 (± SE across seeds) | Deterministic eval | Steps to threshold | Mean KL |
|---|---|---|---|---|---|---|---|---|
| CartPole-v1 | A2C | 3 | 60,000 | 468 | 202.6 ± 13.9 | 454.6 | 15,360 (3/3) | — |
| CartPole-v1 | PPO | 3 | 60,000 | 2,320 | **446.2 ± 0.6** | 500.0 | **10,581 (3/3)** | 0.0067 |
| Acrobot-v1 | A2C | 3 | 60,000 | 468 | −493.2 ± 3.2 | −378.5 | never (0/3) | — |
| Acrobot-v1 | PPO | 3 | 60,000 | 2,320 | **−224.3 ± 137.7** | −200.0 | 33,792 (2/3) | 0.0051 |

**Write 150–250 words on what this shows and what it does not.** What it does
show, in these numbers: PPO reached the CartPole bar in about two-thirds of the
environment steps A2C needed and finished more than twice as high, and on Acrobot
A2C essentially did not learn at all inside the budget (−493 against a −500
floor) while PPO cleared the goal on two seeds of three. What it does **not**
show is at least as important:

* **PPO's Acrobot standard error is 137.7 on a mean of −224.** One of the three
  seeds never took off. "PPO beats A2C on Acrobot" is supportable; "PPO reaches
  −224 on Acrobot" is not — that number has a seed inside it.
* **Two classic-control tasks at one budget is not "in general".**
* **Nobody tuned A2C.** Both arms use this repository's defaults, which is a
  deliberate control *and* a limitation. The A2C learning rate here was itself
  chosen by a small sweep (see the comment in `train/a2c.A2CConfig`), and PPO's
  was not tuned at all, so the comparison is not symmetric in effort either.
* **The gradient-step column is the other half of the trade.** PPO took five
  times as many optimiser steps on the same data. If your constraint is compute
  rather than interaction with the world, that column is the one that bills you.

### The trust region, measured

```sql
select e.env_id,
       count(*)                       as updates,
       avg(u.kl_divergence)           as mean_kl,
       max(u.kl_divergence)           as max_kl,
       avg(u.clip_fraction)           as mean_clip_fraction
from policy_updates u
join experiments e on e.id = u.experiment_id
where u.kl_divergence is not null      -- PPO rows only; A2C and SAC are NULL, not 0
group by 1;
```

Compare your median per-update KL against the **δ ≈ 0.01** a TRPO implementation
would enforce, and say whether the clip alone was enough on your runs. A clip
fraction near zero means the clip never engaged and you were effectively running
an ordinary importance-weighted surrogate; near one means most of the batch fell
outside the region and was wasted.

This template measured a mean per-update KL of **0.0067 on CartPole** and
**0.0051 on Acrobot** across three seeds each — comfortably under δ = 0.01,
which is the interesting result: the clip alone kept the policy inside a region
TRPO would have had to solve a constrained optimisation to guarantee.

**Now look at the tail, because the mean is the misleading statistic here.** On
the same three Acrobot seeds the 95th percentile of the per-update KL was 0.0097,
0.0109 and 0.0107 — right at δ — and the *maximum* was 0.0120, 0.0122 and
**0.0393**. One update on seed 0 moved the policy roughly four times further than
a TRPO trust region would have permitted, and nothing in PPO stopped it, because
nothing in PPO is looking. That single number is the whole difference between a
heuristic and a guarantee, and it is why this column is logged rather than
assumed. Quote your own mean, p95 **and** maximum.

### The entropy sweep

Reference values from this template (`python -m train.entropy_sweep --seeds 3`,
20,000 steps per run, seeds 0–2, threshold −300, random baseline ≈ −1200):

| Arm | Mode | Seeds | Mean final return | Across-seed SD | Across-seed variance | Mean policy entropy | Final α | Episodes to −300 | Deterministic eval |
|---|---|---|---|---|---|---|---|---|---|
| α = 0.5 | fixed | 3 | −375.0 | 12.3 | 151.5 | **−0.168** | 0.5 | 27.0 (3/3) | −130.1 |
| α = 0.01 | fixed | 3 | −393.4 | 20.0 | 401.9 | **−3.254** | 0.01 | 30.7 (3/3) | −131.4 |
| auto | auto | 3 | −378.5 | 7.2 | **51.8** | −0.907 | **0.050** | 29.7 (3/3) | −130.9 |

**Write 100–250 words interpreting it.** The prediction the theory makes is
specific and therefore falsifiable: a higher α should buy a **higher mean policy
entropy** — that is the mechanism — and the effect on the RETURN depends on
whether the extra exploration paid for itself at your budget. Check the mechanism
column before you interpret the outcome column: if α did not move the entropy, it
did not reach the objective and the returns tell you nothing about α.

On these numbers the mechanism is unambiguous and the outcome is not:

* **The entropy ordering is exactly as predicted**, and the gap is enormous:
  −0.17 nats at α = 0.5 against −3.25 at α = 0.01, with automatic tuning between
  them at −0.91. Higher temperature bought a visibly less deterministic policy.
* **The automatic tuner settled at α ≈ 0.050**, between the two hand-chosen
  values and an order of magnitude below its own starting point of 0.2. It found
  a temperature neither fixed arm was.
* **The returns did not separate.** −375, −393 and −378 with across-seed standard
  deviations of 12, 20 and 7: the spread between arms is of the same order as the
  spread within them. The honest sentence is *"this study did not separate the
  arms on final return"*, not *"α = 0.5 was slightly better"*. Pendulum is easy
  enough that all three solved it — every arm's deterministic evaluation lands at
  −130.
* **The one return-side result that does look real is stability.** Automatic
  tuning had roughly an eighth of the across-seed variance of α = 0.01 (51.8
  against 401.9). That is the shape of the argument for tuning α: not that it
  finds a better policy, but that it is less sensitive to the seed than a fixed
  temperature chosen badly.

Three seeds is enough to notice a large effect and not enough to measure a small
one. If you want to claim the variance result rather than merely observe it, run
more seeds and say how many.

### NumPy / PyTorch equivalence

Required, measured, and stated as numbers rather than as a claim that they
passed. `make equivalence` prints all of them.

| Artifact | Quantity compared | Tolerance | This template measured |
|---|---|---|---|
| `a2c_cartpole` | action probabilities | 1e-5 | **8.9e-08** |
| `ppo_acrobot` | action probabilities | 1e-5 | **7.3e-08** |
| `sac_pendulum` | pre-squash mean | 1e-5 | **2.4e-07** |
| `sac_pendulum` | clamped log σ | 1e-5 | **2.8e-07** |
| `sac_pendulum` | deterministic action | 2e-5 | **4.6e-07** |

The residual is float32 weights evaluated in float64; anything at 1e-3 or above
is a bug — almost always a transposed weight matrix, a missing bias, a
one-sided log-σ clamp or a dropped action scale. The diagnostic recipe is at the
bottom of `api/forward.py`, and the SAC test reports the three quantities
separately precisely so that which one fails tells you which mistake you made.

### One deliberate deviation from common practice

`train/sac.py` stores a `policy_updates` row every **50** updates, carrying the
MEAN of the intervening updates rather than a sampled one. SAC takes one update
per environment step, so storing every update would write ~14,000 rows per run
and ~126,000 across the sweep — a meaningful fraction of a 500 MB free tier for a
series no chart on the page can resolve. It is a summary, not a subsample, and it
is a decision you should restate in your own write-up rather than inherit
silently.

## Comparative analysis

*≈500 words, in this README. Required. Every empirical claim must cite a row in
your own tables — a paragraph that cites the literature and not your own numbers
has not done the assignment. Address all six of the following.*

1. **How actor-critic methods differ from pure policy gradient and pure
   value-based methods.** *Which object does each family learn, and what does it
   give up? Cite the variance argument, and point at the A2C-vs-Topic-3 gap if
   you have run one.*
2. **Off-policy actor-critic, and its advantages over on-policy methods.** *SAC
   reuses every transition in its replay buffer many times; PPO reuses a batch
   ten times and discards it; A2C once. Cite your gradient-step column and your
   step budgets. What does the replay buffer cost — what has to be true about the
   data for a stale transition to still be informative, and what do the twin
   critics and the target network exist to protect?*
3. **Why the advantage function improves learning efficiency.** *State the
   unbiasedness argument, then say what actually changes: which term of the
   variance the baseline removes, and why `A = Q − V` asks a better question than
   `Q` alone.*
4. **The key difference between how PPO and TRPO enforce stability.** *A flat
   objective versus a hard constraint. Then use your own `kl_divergence` column:
   did the clip alone keep the KL near δ ≈ 0.01, or did it not? What did the clip
   fraction do over training, and what does its trajectory imply?*
5. **The performance difference lemma and its implications.** *Write out the
   identity, read the subscripts aloud, and explain why the mismatch between
   `d_{π′}` and `d_π` is the thing a trust region exists to control.*
6. **How the entropy term relates to the trust region as a second, distinct form
   of regularisation.** *Reference point, location in the objective, effect on
   the optimum, failure mode in each direction. Then the unifying view: both are
   KL penalties against a reference distribution, and `KL(π ‖ uniform) = −H(π) +
   log|A|` is the identity that connects them. Cite your entropy sweep for the
   first and your KL column for the second.*

[Your ≈500 words here.]

## Limitations analysis

*≈300 words, in this README, and **a separate section from the one above**. Name
at least four concrete limitations of what YOU built, and for each state how you
would test whether it binds in a deployment scenario. "It might not generalise"
is not a limitation; "the observation clipping I chose discards states beyond
±3σ, and I have not measured how often those occur" is.*

Four that are genuinely true of this artifact and worth taking seriously. Use
them as a starting point, not as your answer — the rubric wants limitations of
*your* runs:

1. **SAC is sensitive to the reward scale, and α is priced in reward units.**
   Pendulum's per-step reward reaches −16; α = 0.5 means something quite
   different on a task whose rewards live in [0, 1]. *Test: rescale Pendulum's
   reward by 10× with α held fixed and re-run one seed per arm. If the ordering
   of the arms changes, α is not a transferable hyperparameter and the automatic
   tuner is not a convenience but a requirement.*
2. **Three seeds cannot separate small effects.** The across-seed standard
   deviation in the entropy sweep is the number that decides whether any of your
   arm comparisons is a finding. *Test: bootstrap the three per-seed means and
   report how often the ordering flips. If it flips more than one time in twenty,
   you have not separated the arms.*
3. **Advantages are standardised within each batch** (`normalise_advantages`),
   which makes one learning rate work across CartPole's +1-per-step rewards and
   Acrobot's −1-per-step ones — and rescales the gradient, so a claim about
   gradient magnitude does not survive it. *Test: re-run one seed with it off and
   compare both the curve and the update-to-update variance of `policy_loss`.*
4. **Classic-control results do not transfer.** All three environments are
   low-dimensional, fully observed, noiseless and deterministic given the seed. A
   deployment has none of those properties. *Test: add Gaussian sensor noise to
   the observation at evaluation time — `POST /act` accepts any vector of the
   right width — and plot return against noise scale. The scale at which the
   return falls below the random baseline is the operating margin you actually
   have.*
5. **The deployed artifact is one seed, chosen for the demo.** Its evaluation
   return is not an estimate of what a fresh seed would give you. *Test: train
   three new seeds with the same command and compare their mean against the
   deployed one's.*

[Your ≈300 words. Then: foreseeable misuse, reward-specification risk, and the
worldview reflection this topic calls for.]

## AI-Assistance Disclosure

*Required. What did you generate, with which tool, and how did you verify it?
Generated code must be read, understood and tested by you; blind paste-through is
not acceptable. For this product in particular: if a tool wrote your squashed
Gaussian, say so — and say that the three-way equivalence test is how you know
it is right.*
