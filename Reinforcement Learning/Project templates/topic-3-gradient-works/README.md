<!-- =========================================================================
     PRODUCT 3 README — "Gradient Works"

     The first fifteen lines are graded. Replace the bracketed placeholders
     before your first commit — not the night before submission. Everything
     outside brackets is yours to keep or rewrite; the section headings are
     rubric-aligned and should stay.
     ========================================================================= -->

# Gradient Works — [one line a stakeholder would recognise, e.g. "a policy gradient service that shows you its own noise"]

[**One paragraph, before any code or commands.** Your team lead has heard that
policy gradient methods are unstable and wants evidence, not adjectives, before
approving them for a control project. This service trains the same policy four
ways — with and without a value baseline, crossed with and without
importance-sampled off-policy updates — and reports not only what each one
scored but **the variance of the gradient estimate that produced it**. The
client's question is not "does it work"; it is "why is it noisy, and what fixes
it". Write your own version of this paragraph for the client, not for the
grader. If it only makes sense to someone who has read the assignment, rewrite
it.]

| | |
|---|---|
| **Live app** | https://[your-app].streamlit.app |
| **Supabase project** | `[your-project-ref]` — schema in [`db/migrations/`](db/migrations/) |
| **Service tier (local)** | `uvicorn api.main:app --port 8000` → http://127.0.0.1:8000/docs |
| **Service capture** | [link to your screen recording of the service serving `POST /act` and `GET /docs`] |
| **Environment** | `CartPole-v1` (`gymnasium[classic-control]`) |
| **Deployed policy** | 4 → 64 → 64 → 2 MLP, softmax output, ~18 KB NumPy `.npz` |
| **Author** | [name] |

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.11–3.13. Not 3.14.
pip install -r requirements-train.txt                 # laptop / Colab
cp .env.example .env                                  # then fill in your keys

# 1. apply the schema — paste BOTH migrations into the Supabase SQL editor,
#    001 first. 002 adds the gradient_stats table this product is about.
python -m db.seed

# 2. train the deployable agent (writes telemetry as it goes, then exports)
python -m train.train --episodes 400 --seeds 3

# 3. the evidence: the 2x2 ablation. See "Budgets" below before you run it.
python -m train.ablation --episodes 1000 --seeds 3

# 4. serve, and demonstrate it serving
uvicorn api.main:app --reload --port 8000

# 5. the user interface
streamlit run ui/app.py

# 6. the gate
pytest -q && ruff check .
```

### Budgets

| | command | wall clock | what it is for |
|---|---|---|---|
| sandbox | `python -m train.ablation --episodes 120 --seeds 3` | ~40 s | finding bugs cheaply; **not** a result |
| report | `python -m train.ablation --episodes 1000 --seeds 3` | ~1–2 h on a laptop | the twelve runs you report |

The defaults in `train/ablation.py` are the **sandbox** budget, deliberately, so
that a fresh fork completes in a lunch break and you discover your bugs before
you spend two hours on them. At 120 episodes the four arms have barely separated
and any conclusion drawn from them is noise wearing a chart. The graded run is
**1,000 episodes × 3 seeds × 4 arms = twelve runs**, and all twelve must be rows
in `experiments`. Say in your Quantitative Analysis which command produced the
numbers you are quoting.

## What is where

```
api/        the service tier. Owns the policy. Imports NumPy, never a framework.
  main.py     the six standing endpoints, plus /gradient_stats and /episodes
  forward.py  THE NUMPY FORWARD PASS — the file the equivalence test compares
  policy.py   artifact discovery, checksums, and the act() contract
ui/         the presentation tier. No policy code, no training code, no writes.
  app.py      Concepts / Ablation / Gradient Variance / Play / Run History / Model Card
  service.py  the one switch between in-process and HTTP service calls
train/      the training tier. Imports torch. Never deployed, never imported by api/.
  policy.py     pi_theta(a|s): the network, sampling, log-probs, entropy
  baseline.py   V_phi(s), the MSE fit, and A = R - V(s)
  vpg.py        the loop, the gradient statistics, and the off-policy update
  ablation.py   the 2x2, >= 3 seeds per arm
  train.py      one command: train, evaluate, export, register
  export.py     PyTorch -> .npz. The seam between the tiers.
shared/     the contracts. Pydantic models, settings, the data-tier interface,
            and preprocess.py — every transformation, in one importable place.
envs/       CartPole-v1, exposed as make_env().
db/         migrations 001 (standing schema) and 002 (gradient_stats).
tests/      the standing four, plus the required equivalence test.
policies/   exported artifacts. Committed, because they are what you deployed.
```

## Build-step checklist

Tick these off in order. Each one is a rubric line.

- [ ] **Fork the template, reuse your Supabase project.** Apply
      `001_init.sql`, then `002_gradient_stats.sql` — the new table is
      `(experiment_id, update index, gradient norm, gradient variance, policy
      entropy)` plus the importance-weight columns and an index.
- [ ] **Instantiate CartPole-v1** from `gymnasium[classic-control]` in
      `envs/make_env()`. Never `gymnasium[all]`.
- [ ] **Policy network** in `train/policy.py`: fully connected, observation →
      softmax over actions, with stochastic sampling and log-probability.
- [ ] **Value network** in `train/baseline.py`, trained with MSE, and the
      advantage `A(s,a) = R − V(s)`.
- [ ] **VPG loop** in `train/vpg.py`: sample trajectories, compute discounted
      returns, estimate the advantage, update the value network, ascend the
      policy gradient. One `episodes` row per episode, one `gradient_stats` row
      per update.
- [ ] **Importance sampling** for off-policy updates: reweight experience from
      an older policy, modify the update accordingly, and **log the
      distribution of the weights**.
- [ ] **The 2×2 ablation**, ≥ 1,000 episodes and ≥ 3 seeds per arm. Twelve runs,
      twelve rows in `experiments`.
- [ ] **Export the best policy** with `train/export.py`, register it in
      `policies` with size and checksum, and implement the forward pass in
      `api/forward.py` using NumPy alone.
- [ ] **The equivalence test passes** — `tests/test_equivalence.py`, not a
      manual check. Quote the measured maximum absolute difference below.
- [ ] **Algorithm recommendation memo** (~400 words) in this README.
- [ ] **FastAPI service**: `POST /act`, `POST /rollout`, `GET /runs`,
      `GET /policies`, `GET /healthz`, `GET /version`. Run it under uvicorn and
      record it.
- [ ] **Streamlit frontend** with the six tabs.
- [ ] **Pytest**: the equivalence test plus the standing four.
- [ ] **Deploy** the Streamlit app, confirm the Supabase project is active, and
      link the service tier running locally.

## Architecture — two clouds, three tiers

**Presentation** (Streamlit Community Cloud, deployed) → **Service** (FastAPI,
in this repository, run under uvicorn locally and imported in-process in
production) → **Data** (Supabase Postgres, deployed).

The service tier is a real application with real Pydantic contracts, exercised
over HTTP by the test suite. In the deployed app the Streamlit tier imports the
same handlers instead of crossing the network; `SERVICE_MODE` is the only thing
that changes.

**Why a database tier at all.** A supervised product can be judged from one
number on a held-out set. A reinforcement learning product cannot: the
deliverable is a comparison of learning curves, and every honest claim has the
form *configuration A reached return R in N episodes and configuration B did
not*. That is a query, not a plot. This topic sharpens the point — the headline
claim is about the **variance of an estimator across twelve runs**, which is
not a number you can eyeball off a chart at all.

## The no-PyTorch-in-serving rule — and why this is the topic where it bites

`import torch` alone occupies roughly **490 MB** of resident memory against
Streamlit Community Cloud's **690 MB** guarantee. The entire deployed stack here
measures **82 MB**.

Until this topic the rule was free: nobody imports a framework to index a
Q-table. Now the deployed policy is a network, and two things change.

* The forward pass can be **silently wrong**. A transposed weight matrix is
  still a matrix, and NumPy multiplies it without complaint; the symptom is an
  agent that scored 500 in training and behaves randomly in the deployed app,
  with no error anywhere. Hence `tests/test_equivalence.py`.
* The test suite now needs torch, while this process must not have it. The
  answer is a subprocess (`run_torch_script` in `tests/conftest.py`), not a
  weaker guard. `sys.modules` is per-process, so one stray `import torch` would
  make `tests/test_no_torch.py` fail for a reason that has nothing to do with
  the deployment.

`gymnasium[classic-control]` is now in `requirements-serve.txt` — a few
megabytes, and genuinely a serving dependency because `POST /rollout` runs
episodes server-side. `torch` is not, and never will be. See
[`docs/no-torch.md`](docs/no-torch.md).

## Free-Tier Notes

*Required section. State how this product handles each limit.*

| Limit | Value | How this product handles it |
|---|---|---|
| Streamlit Cloud memory | 690 MB guaranteed | [82 MB measured; NumPy forward pass, no framework in the serving path] |
| Streamlit Cloud sleep | after 12 h idle | [wakes on first request; note the cold start in your demo] |
| Supabase storage | 500 MB | [12,000 episode rows + ~1,200 gradient_stats rows per full ablation ≈ [M] MB. Importance weights are stored as a 20-bin histogram, not raw — the raw form would be ~200k floats per run] |
| Supabase projects | 2 active per person | [one project reused across every topic] |
| Supabase pause | after 1 week idle | [UI degrades visibly; see the health banner] |
| Python version | 3.11–3.13 | [pinned in CI; 3.14 has no Box2D wheels for Topic 4] |
| Artifact size | reviewable in a PR | [~18 KB float32 `.npz`; if yours is megabytes you exported the optimiser state] |

## Theoretical Brief

*350–600 words, mirrored in the Streamlit "Concepts" tab, which carries the full
derivations. Summarise here; do not duplicate at length.*

The policy stops being a table because CartPole's observation is four real
numbers and there are no cells to write into. We parameterise it instead:
`pi_theta(a|s) = softmax(f_theta(s))`.

The **policy gradient theorem** comes from the likelihood-ratio identity
`grad p = p · grad log p`, which moves the parameter out of the distribution and
into the integrand. Expanding `log p(tau)` shows the environment's transition
model differentiating away, leaving

    grad J = E[ sum_t grad log pi(a_t|s_t) · G_t ]

which is why the method is model-free: we need to sample the dynamics, never to
know them.

Set beside the **maximum likelihood** gradient `grad sum_i log pi(a_i|s_i)`, the
two are the same expression with the return as a weight. Policy gradient
learning is maximum likelihood on the agent's own behaviour, where the label is
soft, noisy, and inferred from what happened next. Where labels exist, behaviour
cloning is faster and easier to debug; RL earns its cost when you can only score
outcomes.

The **baseline** exploits the fact that `E_a[grad log pi(a|s) · b(s)] = b(s) ·
grad sum_a pi(a|s) = b(s) · grad 1 = 0`. Subtracting any state-dependent `b`
leaves the estimator unbiased and changes only the second moment; the
variance-minimising choice is close to `V(s)`. Intuitively: reinforce actions
that beat their own state's average, not every action taken somewhere good.

**Importance sampling** reweights old data by `pi_new/pi_old`. It buys sample
reuse and pays in variance that grows with policy mismatch; the exact
trajectory-level weight is a product over the horizon whose variance grows
exponentially, so this implementation uses per-step ratios — biased, bounded,
and the same approximation PPO makes.

[Write your own 350–600 words. A brief that is a paraphrase of this one is not
a brief.]

## Quantitative Analysis

*Every number must be reproducible from `experiments`, `episodes` and
`gradient_stats`, and the seeds must be named. A learning curve with no seed and
no row count is not evidence.*

### The four arms

| Configuration | Seeds | Episodes | Mean return, last 100 (± SE across seeds) | Median gradient variance | Query |
|---|---|---|---|---|---|
| no baseline, on-policy | | | | | |
| baseline, on-policy | | | | | |
| no baseline, importance sampled | | | | | |
| baseline + importance sampled | | | | | |

```sql
-- the comparison, as a query rather than as a screenshot
select e.algorithm,
       count(distinct e.seed)                    as seeds,
       avg(g.gradient_variance)                  as mean_gradient_variance,
       avg(g.policy_entropy)                     as mean_entropy
from gradient_stats g
join experiments e on e.id = g.experiment_id
where g.off_policy = false and g.update_index < 20   -- compare EARLY; see below
group by e.algorithm
order by mean_gradient_variance;
```

### The baseline's effect on gradient variance — the headline claim

**Read the raw variance chart with one caveat in front of you.** In CartPole the
return *is* the episode length, so an arm that learns faster sums more terms
into each gradient and shows higher raw variance for a reason that has nothing
to do with its estimator. Two defensible comparisons, and you should make both:

1. Compare at the same **update index, early**, before the arms separate in
   return (the SQL above does this).
2. Compare in a **controlled single batch** — the same trajectories, the same
   policy parameters, only the advantage changing. That is
   `train.vpg.compare_baseline_variance`, and it is what
   `tests/test_baseline_reduces_variance.py` asserts on every commit.

Reference numbers from this template, measured with the controlled comparison
(`python -c "from train.vpg import compare_baseline_variance as c; print(c(seed=0))"`),
12 episodes per batch:

| Seed | Var(g) without baseline | Var(g) with baseline | Reduction | Explained variance of V |
|---|---|---|---|---|
| 0 | 6.92e3 | 1.11e3 | **×6.23** | 0.69 |
| 1 | 1.87e3 | 5.58e2 | **×3.36** | 0.72 |
| 2 | 2.72e3 | 8.76e2 | **×3.10** | 0.65 |

Quote *your* numbers, from *your* seeds. And quote explained variance alongside:
a baseline that reduces variance while explaining nothing about the return is
reducing it by being a constant, and the honest name for that method is
"mean-centred returns", not "a learned value baseline".

### NumPy / PyTorch equivalence

Required, measured, and stated as a number rather than as a claim that it
passed:

    max |NumPy − PyTorch| = [your measured value]   (tolerance 1e-5, 256 observations)

This template measures **8.9e-08**. `make equivalence` prints it. The residual
is float32 weights evaluated in float64; anything at 1e-3 or above is a bug —
almost always a transposed weight matrix or a missing bias. The diagnostic
recipe is at the bottom of `api/forward.py`.

### One deliberate deviation from common practice

Advantages here are **not** standardised, although
`shared/preprocess.normalise_returns` is available and most published
implementations do it. Rescaling the advantage rescales the gradient, so a
normalised no-baseline arm and a normalised with-baseline arm would report
similar gradient variances no matter what the baseline did — the exact quantity
this product exists to measure would have been normalised away. If you turn
standardisation on, say so and expect the headline chart to flatten.

## Algorithm recommendation memo

*≈400 words. Required. Recommend an algorithm for each of the three problems
below, justify the recommendation **against the evidence in your own
`gradient_stats` and `evaluations` tables**, and state what evidence would
change your mind. A memo that cites the literature but not your own numbers has
not done the assignment.*

**Problem A — a discrete-action control task with a cheap simulator.** Millions
of environment steps are essentially free; engineer time is not.
*Consider: which arm of your ablation reached threshold in the fewest episodes,
and did the difference exceed the standard error across your three seeds? If
samples are free, is the baseline's variance reduction worth the second network
and the extra hyperparameters?*

**Problem B — a continuous-action robotics task with an expensive simulator.**
Each episode costs minutes of wall clock and hardware wear; the action space is
real-valued, so a softmax over actions does not apply.
*Consider: what your importance-sampling arms showed about reusing a batch —
specifically the effective sample size you logged, and how fast it decayed as
the policies drifted. What would you need to change about this codebase to
handle a continuous action space at all, and what does that tell you about which
family of methods to reach for?*

**Problem C — only logged data from a previous policy is available.** No
simulator, no further interaction, one fixed dataset collected under a behaviour
policy you did not choose.
*Consider: your importance-weight histograms. What happened to the weight
distribution as the target policy moved away from the behaviour policy, and what
does that imply about how far from the logging policy you can safely optimise?
What is the support condition, and how would you check whether your logged data
satisfies it?*

For each: **what evidence would change your mind?** Name a specific measurement
— a number from a specific column of a specific table — that would flip your
recommendation. "More experiments" is not an answer.

[Your ~400 words here.]

## AI-Assistance Disclosure

*Required. What did you generate, with which tool, and how did you verify it?
Generated code must be read, understood and tested by you; blind paste-through
is not acceptable. For this product in particular: if a tool wrote your forward
pass, say so — and say that the equivalence test is how you know it is right.*

## Limitations & Responsible Use

*At least four concrete limitations, each with how you would test whether it
binds in a deployment scenario. Then foreseeable misuse, reward-specification
risk, and the worldview reflection your topic calls for.*

Four that are genuinely true of this artifact and worth taking seriously:

1. **The deployed policy is the best of N seeds**, selected on greedy
   evaluation. Its return is therefore an optimistic estimate of what a fresh
   seed would give you. *Test: train three new seeds and compare their mean
   against the selected one's.*
2. **The off-policy updates use a biased per-step importance ratio**, not the
   exact trajectory weight. *Test: compute both on the same batch and compare
   the resulting gradients' direction and magnitude.*
3. **The value baseline is fitted on a batch of ten episodes** and its explained
   variance moves substantially between batches. *Test: log explained variance
   per update and look at its spread, not its mean.*
4. **CartPole terminates at a fixed pole angle**, so the agent has never been
   asked to recover from a state past that threshold. *Test: reset the
   environment to an out-of-distribution state and watch what the policy does —
   `POST /act` accepts any four numbers you like.*

**Reward specification.** CartPole's reward is +1 per surviving step, so this
agent optimises survival and nothing else: not smoothness, not energy, not
staying centred. An agent that maximises the wrong reward competently is more
dangerous than one that fails visibly, and the variance instrumentation in this
product tells you nothing about whether the objective was the right one.
