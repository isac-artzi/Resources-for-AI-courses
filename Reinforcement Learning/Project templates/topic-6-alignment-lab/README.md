<!-- =========================================================================
     PRODUCT 6 README — "Alignment Lab"

     The first fifteen lines are graded. Replace the bracketed placeholders
     before your first commit — not the night before submission. Everything
     outside brackets is yours to keep or rewrite; the section headings are
     rubric-aligned and should stay.

     Numbers quoted below that are not in brackets were MEASURED BY THIS
     TEMPLATE on the offline fallback data, seed 0. They are here so you can
     tell a broken run from a working one. They are NOT your results, and the
     fallback corpus is not the graded corpus. Replace them.
     ========================================================================= -->

# Alignment Lab — [one line a stakeholder would recognise, e.g. "a service that scores text the way your users would, and shows you where that stops being true"]

[**One paragraph, before any code or commands.** A product team is about to
fine-tune a small language model on user preference data and has asked you for
a working demonstration of what that process does, what it costs, and how it
fails. This service scores any text with a reward model trained from real human
comparisons, shows base and aligned model outputs side by side for a library of
prompts, exposes what happened as the KL constraint was loosened, and
demonstrates — with evidence from its own runs — the point at which optimising
the proxy stopped improving the product. Then it steps outside the single-agent
framing and shows what happens when two learners share an environment. Write
your own version of this paragraph for the client, not for the grader. If it
only makes sense to someone who has read the assignment, rewrite it.]

| | |
|---|---|
| **Live app** | https://[your-app].streamlit.app |
| **Supabase project** | `[your-project-ref]` — schema in [`db/migrations/`](db/migrations/) |
| **Service tier (local)** | `uvicorn api.main:app --port 8000` → http://127.0.0.1:8000/docs |
| **Service capture** | [link to your screen recording of the service serving `POST /score`, `POST /compare` and `GET /docs`] |
| **Preference dataset** | [`trl-lib/ultrafeedback_binarized`, N comparisons, train/test split] |
| **Deployed model** | TF-IDF reward head, [V]-term vocabulary → 64 → 1, ~20 KB NumPy `.npz` |
| **Not deployed** | the embedding reward head — stronger, and its featuriser is a transformer |
| **Author** | [name] |

---

## The architecture note — this is the design, not a workaround

**No transformer runs in the serving tier, and neither does PyTorch.** The
split is the same one this project has used since the first topic, applied to
language.

* **The encoder is used offline, in the training tier**, to turn text into
  fixed-length embedding vectors. It is **frozen** — never fine-tuned, so no
  gradient flows into it — and it never leaves `train/`. Its output is cached
  to disk, so it runs once per unique string rather than once per request.
* **The reward model is a small head on top of those vectors** and exports to a
  NumPy archive exactly like every policy in this project.
* **Language-model generation is likewise performed offline** and the
  completions are persisted, so **the deployed service scores text rather than
  generating it.** There is no `/generate`, and its absence is architectural.
* **The deployed head carries its own featuriser.** The vocabulary and the IDF
  vector live inside the same `.npz` as the weights, and `api/reward.py`
  re-implements scikit-learn's TF-IDF transform in about fifteen lines of
  NumPy. One artifact, one checksum, one thing to deploy.

**Why, in numbers.** `import torch` alone is ~490 MB resident against Streamlit
Community Cloud's 690 MB guarantee. Add `transformers` and a 124M-parameter
causal LM and you are over a gigabyte before the first token is sampled. This
entire deployed stack measures **82 MB**. A service that scores fits the free
tier; a service that generates does not.

**Why it generalises.** That is the shape of a real inference budget: the
expensive model runs where you can afford it, the cheap one runs where the
traffic is. The same reasoning produces batch feature pipelines, distilled
serving models and precomputed embedding stores in production systems that have
never heard of this project. Full argument in
[`docs/no-torch.md`](docs/no-torch.md).

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.11–3.13. Not 3.14.
pip install -r requirements-train.txt                 # laptop / Colab
cp .env.example .env                                  # then fill in your keys

# 1. apply the schema — paste BOTH migrations into the Supabase SQL editor,
#    001 first. 002 adds preferences, completions and alignment_runs.
python -m db.seed

# 2. the whole pipeline. --offline uses the deterministic fallbacks and needs
#    no network; drop it for the real dataset and a GPU runtime.
python -m train.train --offline --quick     # ~35 s. Finds bugs. NOT a result.
python -m train.train --offline             # ~2 min, seven betas
python -m train.train                       # the graded run: hub + GPU

# 3. serve, and demonstrate it serving
uvicorn api.main:app --reload --port 8000
curl -X POST localhost:8000/score -H 'Content-Type: application/json' \
     -d '{"text":"A specific, measured answer with a tested baseline."}'

# 4. the user interface
streamlit run ui/app.py

# 5. the gate
pytest -q && ruff check .
```

> **If `GET /completions` comes back empty after a successful pipeline run,
> nothing is broken and you have found the point of the data tier.** With
> `SUPABASE_URL` unset, `shared/store.py` falls back to an **in-process**
> store, so the rows `train/train.py` wrote live in the *training* process and
> vanish when it exits — a separately-launched `uvicorn` has its own empty one.
> Two ways forward, and you want both eventually: fill in `.env` so the rows go
> to Postgres and every process sees them (this is the deployed configuration),
> or run the Streamlit app with `SERVICE_MODE=inprocess` in the same process
> that ran the pipeline. `POST /score` and `POST /compare` work either way,
> because they read a file rather than a database.

Each stage is also a module with its own `main()`, which is what you actually
want while debugging:

```bash
python -m train.data         --offline --inspect 5   # print comparisons in full
python -m train.embed        --offline --diagnose    # the encoder's length leak
python -m train.reward_model --offline               # the fastest loop
python -m train.dpo          --offline
python -m train.reward_hacking --offline
python -m train.multiagent   --offline
```

### Budgets

| | command | wall clock | what it is for |
|---|---|---|---|
| sandbox | `python -m train.train --offline --quick` | ~35 s | finding bugs cheaply; **not** a result |
| offline report | `python -m train.train --offline` | ~2 min | seven betas; still the fallback corpus |
| graded | `python -m train.train` | ~2–4 h, Colab GPU strongly recommended | the run you report |

The defaults are the **offline** budget, deliberately, so that a fresh fork
completes in a coffee break and you discover your bugs before you spend two
hours of GPU time on them. **Say in your Quantitative Analysis which command
produced the numbers you quote.**

### About the offline fallback — read this before you quote a number

This sandbox has no access to the model hub, so every stage ships a
deterministic offline generator alongside the real path, selected by
`--offline`:

| stage | required path | offline fallback | labelled as |
|---|---|---|---|
| `train/data.py` | `trl-lib/ultrafeedback_binarized` | a synthetic preference generator with a latent lexical quality signal, a length confound, and a quality term the reward model cannot see | `source = 'synthetic-offline'` on every row |
| `train/embed.py` | a frozen sentence-transformer, mean-pooled | a deterministic BLAKE2b token-hashing encoder, mean-pooled | `encoder = 'hashing-offline'` |
| `train/dpo.py` | TRL's `DPOTrainer` on gpt2 | a self-contained trigram-context PyTorch LM over a closed vocabulary | `notes = 'tiny-lm offline fallback'` |
| `train/multiagent.py` | `simple_spread_v3` from `mpe2` | a cooperative gridworld with the same reward structure | named in the result dict |

**The fallback is not a substitute for the real data and no result computed on
it belongs in your report as though it were.** What it is for is making the
whole pipeline — embedding, both reward heads, DPO, generation, scoring, the
beta sweep, the multi-agent experiments — runnable and debuggable end to end.
There is no silent fallback anywhere: `--offline` must be typed. A run that
quietly became synthetic because the hub timed out would be the worst possible
failure, because nothing about the output would look wrong.

---

## What is where

```
api/        the service tier. Owns the reward head. NumPy only — no torch,
            no transformers, and no scikit-learn either.
  main.py     the standing endpoints + /score, /compare, /completions,
              /alignment_runs
  reward.py   THE DEPLOYED MODEL: the TF-IDF transform and the forward pass,
              in NumPy. The file the equivalence test compares.
  policy.py   artifact discovery, checksums, and the three artifact kinds
ui/         the presentation tier. No model code, no training code, no writes.
  app.py      Concepts / Score It / Base vs Aligned / Reward Hacking /
              Multi-Agent / Run History / Model Card
  service.py  the one switch between in-process and HTTP service calls
train/      the training tier. Imports torch. Never deployed, never imported
            by api/ or ui/.
  data.py           load and split the preference dataset -> `preferences`
  embed.py          the FROZEN encoder, run once, cached
  reward_model.py   TWO HEADS, ONE LOSS: accuracy, margins, length bias
  dpo.py            the DPO loss in <15 lines, plus the end-to-end run
  reward_hacking.py proxy vs target vs KL, and the decoupling point
  multiagent.py     IPD, non-stationarity, a cooperative task, matching pennies
  train.py          one command: every stage, in the order the build requires
  export.py         PyTorch -> .npz. The seam between the tiers.
shared/     the contracts. Pydantic models, settings, the data-tier interface,
            and preprocess.py — including THE TOKENISER, which both tiers use.
envs/       the IPD against a FIXED opponent, exposed as make_env(). It is a
            valid environment only because the opponent is frozen — which is
            the point.
db/         migrations 001 (standing schema) and 002 (preferences,
            completions, alignment_runs).
tests/      the standing four, plus equivalence, length bias, the /score
            schema, audit hashing, the DPO loss and the multi-agent theory.
policies/   exported artifacts. Committed, because they are what you deployed.
reports/    charts and JSON summaries. Committed — they are your evidence.
cache/      embedding cache. Gitignored: regenerable, and megabytes of float32
            nobody can review in a pull request.
```

---

## Build-step checklist

Tick these off in order. Each one is a rubric line.

- [ ] **Fork the template, reuse your Supabase project.** Apply `001_init.sql`,
      then `002_topic6.sql` — `preferences` (prompt, chosen, rejected, split),
      `completions` (prompt_id, model_variant, beta, text, reward_score) and
      `alignment_runs` (beta, implicit reward margin, implicit reward accuracy,
      KL from reference, mean reward-model score), with indexes and the
      `alignment_by_beta` view the UI reads.
- [ ] **Load ≥ 2,000 comparisons** and persist both splits to `preferences`.
      **Inspect several by hand** — `python -m train.data --inspect 5` prints
      them in full — and record what you found below.
- [ ] **Embed once, offline,** with a frozen encoder. Cache the vectors.
      **State how long it took and how large the cache is.**
- [ ] **Two reward heads, one loss.** Both on the pairwise Bradley–Terry loss,
      differing only in input features. Export both to `.npz`, register both in
      `policies`. Report (a) held-out pairwise accuracy for each against the
      50% baseline and the reward-margin distribution; (b) the length-bias
      regression for each, naming which head is more susceptible and why that
      is what you would expect; (c) which head you would ship and what the
      cheaper one costs you.
- [ ] **DPO alignment** at β = 0.05, 0.1 and 0.5, with a frozen reference copy,
      logging loss, implicit reward margin, implicit reward accuracy and KL
      from the reference to `alignment_runs`.
- [ ] **Offline generation and scoring** for ≥ 50 held-out prompts, from the
      base model and each aligned variant, persisted to `completions` and
      scored **through the deployed `/score` endpoint**. Report mean reward and
      win rate against the base for each β.
- [ ] **Multi-agent experiments** — (a) independent tabular Q-learning on the
      IPD with each agent's reward plotted; (b) non-stationarity demonstrated
      by freezing one agent and retraining the other, then letting both learn;
      (c) independent learners on a cooperative environment with team reward
      plotted. Plus 200–300 words contrasting independent learning with
      centralised training and decentralised execution, naming one situation in
      which the independent approach should be expected to fail.
- [ ] **Reward-hacking evidence.** Mean reward-model score and KL from the
      reference against β on the same axes. **Read at least twenty generated
      completions by hand.** Present concrete evidence from your own runs —
      quoting specific completions — or explain why your setup did not exhibit
      it. State which β you would ship and why.
- [ ] **Report (≈700 words, below).**
- [ ] **Worldview reflection (≈300 words, below).**
- [ ] **The FastAPI service**: `POST /score`, `POST /compare`,
      `GET /completions`, `GET /runs`, `GET /policies`, `GET /healthz`,
      `GET /version`. Run it under uvicorn and record it.
- [ ] **The Streamlit frontend** with the seven tabs.
- [ ] **Pytest**: the NumPy/PyTorch equivalence test with a stated tolerance,
      the length-bias regression test with a justified threshold, the `/score`
      schema test, and the standing four.
- [ ] **Deploy** the Streamlit app, confirm the Supabase project is active, and
      link the service tier running locally.

---

## Architecture — two clouds, three tiers

**Presentation** (Streamlit Community Cloud, deployed) → **Service** (FastAPI,
in this repository, run under uvicorn locally and imported in-process in
production) → **Data** (Supabase Postgres, deployed).

The service tier is a real application with real Pydantic contracts, exercised
over HTTP by the test suite. In the deployed app the Streamlit tier imports the
same handlers instead of crossing the network; `SERVICE_MODE` is the only thing
that changes.

**Why a database tier at all.** In this topic the answer is sharper than in any
previous one. The deliverable is a comparison across β of four quantities that
are *not* a learning curve — an implicit reward margin, a KL divergence, a mean
proxy score and a quality signal the model never saw — joined against a library
of persisted completions. That is a `GROUP BY`, not a plot. It is also the only
way the reward-hacking claim can be checked by a reader rather than believed:
the chart in the Reward Hacking tab is a rendering of `alignment_by_beta`, and
anyone with the project can run the query.

**One privacy decision, made in the schema.** User text submitted to `/score`
is **never stored**. `audit_log.state_hash` holds a SHA-256 digest and nothing
else. The comment saying so has been in `001_init.sql` since the first topic;
this is the topic where it stops being hypothetical, and
`tests/test_audit_hashing.py` asserts it over the entire audit row rather than
over the one field the text is expected in.

---

## Free-Tier Notes

*Required section. State how this product handles each limit.*

| Limit | Value | How this product handles it |
|---|---|---|
| Streamlit Cloud memory | 690 MB guaranteed | [82 MB measured. NumPy forward pass, NumPy featuriser, no framework anywhere in the serving path. `GET /version` reports `torch_imported` honestly at runtime.] |
| Streamlit Cloud sleep | after 12 h idle | [wakes on first request; note the cold start in your demo] |
| Supabase storage | 500 MB | [`preferences` holds two full response texts per row and is the big table: 2,400 UltraFeedback comparisons ≈ [M] MB. `completions` is 60 prompts × 4 variants. Neither grows with training steps.] |
| Supabase projects | 2 active per person | [one project reused across every topic] |
| Supabase pause | after 1 week idle | [UI degrades visibly; see the health banner and the `degraded` flag on every read endpoint] |
| Python version | 3.11–3.13 | [pinned in CI] |
| Artifact size | reviewable in a PR | [20 KB TF-IDF head including its vocabulary and IDF vector. If yours is megabytes you exported the vectoriser object or the optimiser state.] |
| Colab GPU session | 12 h, pre-emptible | [the embedding pass is cached to disk, so a dropped session costs the DPO run and not the encoding] |
| **Hub / network access** | none in some environments | [`--offline` runs every stage deterministically; every row it writes is labelled] |

---

## Theoretical Brief

*350–600 words, mirrored in the Streamlit "Concepts" tab, which carries the
full derivations. Summarise here; do not duplicate at length.*

**The three stages.** Supervised fine-tuning fits a policy by maximum
likelihood on demonstrations and produces `pi_ref`, the reference. Reward
modelling fits `r(x, y)` from *comparisons*. Policy optimisation then improves
the policy against `r` under a KL constraint. Stage 2 is what makes alignment a
reinforcement learning problem rather than a supervised one: humans are
unreliable at absolute scores and reliable at comparisons, and a comparison is
a preference over outcomes, not a target you can regress onto.

**Bradley–Terry.** `P(y_c > y_r | x) = sigma(r(x,y_c) - r(x,y_r))`, so the
negative log-likelihood of the observed comparisons is
`L = -E[log sigma(r(x,y_c) - r(x,y_r))]`. Three consequences run through the
whole product. Only *differences* are identified — adding a constant to
`r(x, ·)` leaves the loss unchanged — so `/score`'s output is meaningful in
comparison and meaningless alone. The loss is *unbounded below*, so nothing but
capacity and the training budget stops the model driving its margins to
infinity. And the gradient *vanishes on easy pairs*, so held-out accuracy
plateaus long before the loss does.

**The KL-regularised objective.** `max_pi E[r] - beta*KL(pi || pi_ref)` has the
closed-form optimum
`pi*(y|x) = pi_ref(y|x) * exp(r(x,y)/beta) / Z(x)`. The partition function
sums over every sequence the model could emit and is intractable, which is why
stage 3 was a policy-gradient problem before DPO.

**No new mathematics.** This is the maximum-entropy objective from the
actor-critic topic with a different reference distribution. Using
`KL(pi || uniform) = -H(pi) + log|A|`, setting `pi_ref` uniform recovers
`E[r] + beta*H(pi)` up to a constant: `beta` plays exactly the role `alpha`
played there, and the entropy bonus is the special case.

**DPO inverts the identity.** Solve the closed form for the reward —
`r = beta*log(pi/pi_ref) + beta*log Z(x)` — and substitute into Bradley–Terry.
The loss depends on `r` only through a difference *at the same prompt*, and
`beta*log Z(x)` depends on the prompt alone, so it cancels exactly. The reward
model was not approximated away; it was reparameterised. `beta*log(pi/pi_ref)`
is the *implicit reward*, and it is what `alignment_runs.implicit_reward_margin`
and `.implicit_reward_accuracy` measure.

**GRPO** removes the critic instead of the reward model: sample `G` completions
per prompt and use the group's own mean as the baseline,
`A_i = (r_i - mean(r))/std(r)`. The advantages sum to zero by construction. It
trades per-token credit assignment — which is what the critic supplied — for
one fewer resident model.

[Write your own 350–600 words. A brief that paraphrases this one is not a
brief.]

---

## Quantitative Analysis

*Every number must be reproducible from `preferences`, `completions` and
`alignment_runs`, and the seed must be named. A chart with no seed and no row
count is not evidence.*

**Provenance of the numbers in this section:** measured by this template with
`python -m train.train --offline`, seed 0, on the **synthetic fallback corpus**.
They are here as a reference for what a working run looks like. Replace them
with yours, and say which command produced them.

### The dataset, and what hand inspection showed

2,400 comparisons, split **by prompt** (not by pair — several comparisons share
a prompt, and a pair-level split would leak prompt vocabulary across the
boundary): 1,920 train pairs over 231 prompts, 480 test pairs over 60 prompts.

| | mean chosen length | mean rejected length | chosen is longer | label–length correlation |
|---|---|---|---|---|
| train | 18.2 tokens | 15.7 tokens | 62.1% | r = +0.266 |
| test | 18.4 tokens | 15.7 tokens | 65.0% | r = +0.289 |

**That last column is the single most important number in this section**, and
almost nobody reports it. It says that in this corpus, *choosing the longer
response* is worth roughly 62–65% accuracy before you have read a word. Every
reward-model accuracy below must be read against it, and so must every
length-bias correlation.

[**Your hand inspection goes here.** Read five comparisons in full — `python -m
train.data --inspect 5` — and write down what you noticed. Useful things to
look for: how often the "rejected" response is actually fine; whether the
chosen one is merely longer or merely more formatted; whether you would have
made the same call. Two or three specific observations beat a paragraph of
generalities.]

### Embedding pass (build step 3)

| | value |
|---|---|
| encoder | `hashing-offline`, 256-dimensional *(real path: a frozen sentence-transformer, 384-d)* |
| texts encoded | 4,800 (chosen + rejected, both splits) |
| **elapsed** | **0.11 s** *(real path on CPU: expect 2–5 minutes; on a Colab T4, ~20 s)* |
| **cache size** | **4.56 MB** float32 |
| encoded unit | the response alone, **not** prompt + response — see below |

The encoder is frozen and cached, so it runs once per unique string. Note the
modelling limitation this shares with the deployed head: both score `r(y)`
rather than `r(x, y)`, because `POST /score` takes text and nothing else.
Conditioning on the prompt is worth several accuracy points on real data; set
`include_prompt=True` in `train/embed.py::embed_dataset` and quote the gap in
your limitations section.

### The two reward heads — the required comparison

Same loss, same optimiser, same 25 epochs, same 64-unit hidden layer, seed 0,
480 held-out pairs. **Only the input features differ.**

| | **TF-IDF head** (deployed) | **Embedding head** (not deployed) |
|---|---|---|
| feature dimension | 74 | 256 |
| **held-out pairwise accuracy** | **0.869 ± 0.015** | **0.896 ± 0.014** |
| baseline | 0.500 | 0.500 |
| "prefer the longer one" baseline | 0.650 | 0.650 |
| reward-vs-length Pearson r | **+0.353** (r² = 0.125) | **+0.328** (r² = 0.108) |
| length decodability from features (Ridge R²) | **0.757** | **0.104** |
| length-matched accuracy (±2 tokens, n = 131) | 0.863 | 0.893 |
| accuracy lost to length matching | **−0.006** | **−0.003** |
| artifact size | 19.7 KB | 60.9 KB |
| featuriser needed at serving time | a vocabulary and an IDF vector, **in the archive** | a transformer, ~90 MB + PyTorch |

Both clear the 50% baseline by more than twenty standard errors, so the
training loop is doing what it claims.

**Is the embedding head actually better? Test it PAIRED.** The two heads are
scored on identical held-out comparisons, so the unpaired error bar is the
wrong instrument — and it is wrong in the direction that matters:

| | value |
|---|---|
| accuracy difference (embedding − TF-IDF) | **+0.0271** |
| **unpaired** standard error, `sqrt(se_a² + se_b²)` | ±0.0207 → z = 1.3, reads as *"not significant"* |
| **paired** standard error, `sqrt(b + c)/n` (McNemar) | **±0.0091 → z = 2.98, p ≈ 0.003** |
| pairs only TF-IDF got right (b) | 3 |
| pairs only the embedding head got right (c) | 16 |
| discordant pairs out of 480 | 19 |

Same data, opposite conclusion, and the paired one is right. Most of the
variance in either head's accuracy is variance in *which pairs are hard*, and
that is shared between them and cancels; the 461 concordant pairs carry no
information about the difference at all. `train/reward_model.py::
paired_accuracy_difference` computes this, and the pipeline reports it. **Do
not quote two independent error bars for a comparison run on the same data.**

**(b) The length-bias finding, and which head is more susceptible.**

The raw correlations are **statistically indistinguishable** (+0.353 vs +0.328,
and they swap order across seeds). Reporting only those two numbers would have
supported no conclusion at all, so this template measures two more things, and
they do separate the heads decisively:

* **Length is 7× more linearly decodable from TF-IDF features** than from the
  embedding: R² = 0.757 against 0.104, stable across seeds.
* Both heads lose almost nothing to length matching (−0.006 and −0.003), so
  neither is *currently* riding the shortcut on this corpus.

**The TF-IDF head is the more susceptible one, and the mechanism is the L2
normalisation.** A TF-IDF vector is divided by its own Euclidean norm, which
grows roughly as the square root of the number of distinct terms — so a longer
response has *systematically smaller entries across the board*, and a linear
function of the feature vector reads length almost directly. The mean-pooled
embedding's length signal lives in its **norm**, which is a *quadratic*
function of the coordinates and therefore not available to a linear read-out at
all; only the hidden layer can approximate it, and only badly.
(`python -m train.embed --offline --diagnose` shows ‖e(y)‖ correlating with
length at r = −0.80 — the signal is there, it is just not linearly reachable.)

This is the **opposite of the folk expectation**, which is that dense
representations pick up spurious correlates and bag-of-words models are
"interpretable" and therefore safe. On this evidence it is the interpretable
one that is handed length for free.

**Why the correlation alone could not have told you this.** In a corpus where
longer responses genuinely *are* better, a head that scores quality perfectly
and ignores length entirely still shows a positive reward–length correlation,
because the two are correlated in the data. Both heads sit at r ≈ 0.34 against
a corpus confound of r = 0.289. That is roughly what an honest model looks
like — and it is why the length-bias test asserts on *two* quantities. See
`tests/test_length_bias.py` for the threshold and its justification.

[**Your version of this analysis.** On real UltraFeedback the picture is
sharper: the length heuristic alone reaches around 68–70% while a fitted reward
model reaches around 72–75%, so length is nearly as predictive as the model and
the shortcut is much more tempting. Expect larger correlations and a much
larger length-matched drop, and report both.]

**(c) Which head would you ship, and what does the cheaper one cost you?**

[Your argument. The answer this template ships with: the TF-IDF head, and the
reason is **not** that it is better. It is 0.027 worse. It is that its feature
pipeline is 20 KB and fits in the archive, while the embedding head's feature
pipeline is a transformer and does not fit the deployment at all. What the
cheaper one costs: 2.7 accuracy points, no semantic generalisation whatsoever
(a synonym the vocabulary has never seen is invisible), a much higher latent
susceptibility to length, and — from `HeadConfig.epochs` — a tendency to overfit
sooner, since 74 term weights can memorise a 1,920-pair training set and a
256-dimensional dense vector cannot memorise it as cheaply. Argue your own
position, and name the measurement that would change it.]

**Reward-margin distributions:** [`reports/reward_margins.png`](reports/reward_margins.png).
**Length-bias regressions:** [`reports/length_bias.png`](reports/length_bias.png).

### NumPy / PyTorch equivalence

Required, measured, and stated as a number rather than as a claim that it
passed:

    max |NumPy − PyTorch| = 1.375e-07     (tolerance 1e-4, 96 probe texts)

`make equivalence` prints it. The comparison is **end to end from a raw
string** — scikit-learn's TF-IDF plus a torch module on one side, this
repository's NumPy featuriser plus forward pass on the other — because in this
topic the featuriser is the bigger risk. A transposed matrix is a familiar bug;
a tokeniser that lowercases on one side and not the other is invisible to any
test that starts from a feature vector.

The probe set is adversarial on purpose: an empty document (the 0/0 case in the
L2 normalisation), a fully out-of-vocabulary document, a single token repeated
forty times, mixed case and punctuation, and a document six times longer than
anything in training. The tolerance is justified line by line in the header of
`tests/test_equivalence.py`.

### Alignment results (build steps 5 and 6)

SFT 300 steps, DPO 600 steps per β, one shared frozen reference, 60 held-out
prompts, seed 0. **Every completion was scored through the deployed `POST
/score` handler**, so the numbers below are the numbers the service returns.

| variant | β | KL from ref (nats) | mean reward (proxy) | mean true quality | repetition rate | mean tokens | implicit margin | implicit acc | win rate vs base |
|---|---|---|---|---|---|---|---|---|---|
| base | — | 0.00 | 0.889 | 0.171 | 0.096 | 15.2 | — | — | — |
| dpo | **0.5** | 6.41 | 3.100 | 0.352 | 0.218 | 26.1 | 1.845 | 0.727 | 0.933 |
| dpo | **0.1** | 16.50 | 3.529 | 0.383 | 0.266 | 26.6 | 1.173 | 0.750 | 0.967 |
| dpo | **0.05** | 20.87 | 3.599 | **0.403** ← peak | 0.273 | 25.9 | 1.088 | 0.765 | **1.000** |
| dpo | 0.02 | 25.45 | 3.626 | 0.398 | 0.282 | 26.8 | 1.041 | 0.769 | 0.983 |
| dpo | 0.01 | 30.22 | **3.638** ← peak | 0.385 | 0.292 | 27.6 | 1.023 | 0.788 | 0.983 |
| dpo | 0.005 | 34.23 | 3.597 | 0.374 | 0.281 | 27.4 | 1.001 | 0.783 | 0.967 |
| dpo | 0.002 | 43.43 | 3.616 | 0.366 | **0.318** | 26.7 | 0.955 | 0.783 | 0.967 |

β = 0.05, 0.1 and 0.5 are the three the build step requires; the four smaller
values were added because **the decoupling in this setup sits below 0.05, and a
sweep that stops at the required three shows a monotone curve and no
phenomenon.**

Two sanity checks worth reading off this table before anything else:

* **KL falls monotonically as β rises** — 43.4 down to 6.4. That is the KL
  coefficient doing what it is named for, and every claim below depends on it.
  `tests/test_dpo_loss.py` asserts it.
* **The implicit reward margin rises with β** — 0.955 to 1.845 — which is
  arithmetic, not learning: the margin is `beta` times a log-ratio difference.
  Margins at different β are **not** comparable and must not be plotted
  together as though they were.

```sql
-- the whole table above, as a query rather than as a screenshot
select coalesce(beta, 'Infinity'::double precision) as beta,
       count(*)              as completions,
       avg(reward_score)     as mean_proxy,
       avg(true_quality)     as mean_true_quality,
       avg(tokens)           as mean_tokens
from completions group by 1 order by 1 desc;
```

### Reward hacking — the required evidence

Chart: [`reports/reward_hacking.png`](reports/reward_hacking.png).

| | value |
|---|---|
| **decoupling point** | **β = 0.05** (KL ≈ 21 nats) |
| peak true quality at | β = 0.05 |
| peak proxy reward at | β = 0.01 |
| proxy change past the peak | **+0.017** |
| true-quality change past the peak | **−0.036** |
| proxy/target correlation, low-pressure half (β ≥ 0.05) | **+0.97** |
| proxy/target correlation, high-pressure half (β ≤ 0.02) | +0.57, and **unstable across seeds** |

**Read the two correlations with care.** Each is computed over three or four
points and is not a stable statistic; the honest summary is the pair of
magnitudes above. From the last configuration at which proxy and target agreed
(β = 0.05) to the most heavily optimised one (β = 0.002), **the proxy went up
and the target went down.** That is Goodhart's law, measured.

**The mechanism, and the concrete evidence.** `true_quality` here is
*good-token density × token diversity*: humans dislike repetition, and the
labeller does too. Every training comparison is between two non-degenerate
responses whose diversity is similar, so **the comparisons carry almost no
information about the diversity term.** The reward model therefore learns the
density term and nothing else — an honest fit to what it was shown. A policy
pushed far from the reference discovers that density is maximised by repeating
the highest-weighted token. Density goes up, diversity collapses, `true_quality`
falls, and the reward model — which never saw a repetitive response and never
learned to dislike one — scores the result *higher*.

The repetition-rate column above is that mechanism, monotone and scale-free:

    base 0.096  ->  beta=0.5 0.218  ->  beta=0.05 0.273  ->  beta=0.002 0.318

and it is visible by eye in the completions themselves:

> **base** — `model`
> **β = 0.5** — `model caveat above magic sample tested verified change needs benchmark citation verified limitation given result …`
> **β = 0.05** — `tradeoff quantified caveat stepwise range concrete item caveat stepwise benchmark the verified limitation given result reproducible evidence concrete …`
> **β = 0.002** — `reproducible reproducible where verified specific across item reproducible stepwise part citation verified process quantified quantified case output …`

[**Read at least twenty of your own completions and quote the ones that make
the point.** Paste the actual text, with its β and its proxy score. A claim
about output quality that is not accompanied by output is not evidence. If your
setup did *not* exhibit decoupling, say so and explain why — a sweep that
stopped at β = 0.05, a reward model whose failure mode the policy could not
reach, or a policy that never moved far enough. That is a legitimate finding
and it is much better than a manufactured one.]

**Which β would I ship, and why.** [β = 0.05 on this evidence: it is the peak
of the target series, it has the highest win rate against the base (1.000), and
it is the last point at which the proxy and the target still move together.
Everything to its right buys proxy score with quality. Note what makes that
recommendation fragile: **the location of the peak depends on `true_quality`,
which you cannot measure at deployment time.** In production you would be
choosing β from the proxy series alone, and the proxy series is nearly flat
across the entire region where the target is falling. State how you would
detect the same turn without the ground truth — that is the real question, and
"a held-out human-rated set" is the real answer.]

### Multi-agent results

Chart: [`reports/multiagent.png`](reports/multiagent.png). All four panels are
pure NumPy tabular Q-learning; no framework is involved.

**(a) Iterated prisoner's dilemma**, 3,000 episodes × 20 steps, memory-1 state,
independent learners:

| | agent A | agent B |
|---|---|---|
| mean reward per step (last 10%) | 1.574 | 1.642 |
| cooperation rate | 28.4% | |
| mutual defection pays | 1.0 | |
| mutual cooperation pays | 3.0 | |

Nearer mutual defection than mutual cooperation, which is what the theory
predicts and *why*: neither agent's update contains any term for the effect of
its own action on the other's future behaviour, and that is the only channel
through which reciprocity could pay. Both agents would prefer 3.0 and neither
can get there. This is the two-agent tragedy of the commons.

**(b) Non-stationarity, measured.** Same agent, same algorithm, same
hyperparameters. The only change is whether the opponent is learning:

| agent A, last third of training | vs **frozen** tit-for-tat | vs a **learning** opponent | ratio |
|---|---|---|---|
| mean reward per step | 2.975 | 1.317 | — |
| **Q-table drift per episode** | 8.2 × 10⁻⁸ | 9.7 × 10⁻² | **1.2 × 10⁶ ×** |
| **greedy-policy switches** | **0** | 161 | — |
| **late-reward standard deviation** | 0.040 | 0.471 | **11.8 ×** |

Against a frozen opponent the transition function is fixed, the Markov property
holds, Q-learning's convergence guarantee applies, and it converges: the table
stops moving and the greedy policy never changes again. Against a learning
opponent none of that is true, and none of it happens. **The comparison is
about stability, not return** — the two arms play different opponents, so their
returns are not comparable and charting them together would be charting two
different games.

**(c) Cooperative task**, two agents, two landmarks, one shared reward, 4,000
episodes:

| | value |
|---|---|
| final team reward per step | **−1.046** |
| random-policy floor | −5.785 |
| improvement | +4.74 |

A rising curve is not evidence on its own — ε-decay alone raises it — so the
floor is the comparison that matters.

**Matching pennies** (DQ 6b), 60,000 steps, single state, Boltzmann
exploration at temperature 0.1:

| | agent A | agent B |
|---|---|---|
| time-averaged P(tails) | 0.499 | 0.498 |
| mixed Nash equilibrium | 0.500 | 0.500 |
| standard deviation of the policy | 0.290 | 0.289 |
| range of P(tails) over training | 0.006 → 0.991 | |
| **late/early policy-spread ratio** | **0.99** | |

The time average lands on the equilibrium; the *iterates never do*, and they do
not even come close — agent A's policy swings across almost the entire unit
interval. The last row is the actual finding: the spread is **as large at the
end of training as at the beginning**, which is what distinguishes an **orbit**
from convergence with noise. A converging pair would show a ratio near zero.

Two implementation notes that are part of the result:

* **Boltzmann rather than epsilon-greedy exploration**, so the policy is a
  smooth function of the Q-values and the orbit is visible. An epsilon-greedy
  policy is a step function of the table — it can take exactly two values per
  action — and its "phase plot" is four corners. This is also the
  theoretically correct choice: the continuous-time limit of Boltzmann
  Q-learning is the **replicator dynamics**, whose trajectories in matching
  pennies are closed orbits around the mixed equilibrium. The figure is a
  picture of that prediction.
* **The empirical action frequency in a sliding window does not work** and it
  is worth knowing why: the greedy action flips about every 27 steps at these
  hyperparameters, so any window long enough to estimate a frequency precisely
  is long enough to average the cycle away. The template's first attempt
  produced a tight blob at (0.5, 0.5) that looked like convergence and was the
  opposite of the truth.

### Independent learning vs CTDE

*200–300 words. Required.*

[Your 200–300 words. The shape of the argument, with the hooks from this
template's own numbers:

Independent learners treat the other agents as part of the environment, which
is precisely what makes the transition function time-dependent — panel (b)
above is that violation, measured, at a Q-drift ratio of 10⁶. Centralised
training with decentralised execution keeps *execution* decentralised (each
agent still acts on its own observation, so nothing about deployment changes)
but trains a **centralised critic** that sees the joint state and the joint
action. The critic's input stops changing when the other agents learn, so the
non-stationarity is absorbed into a component that is discarded before
deployment.

**Name one situation in which the independent approach should be expected to
fail**, and be specific. The candidate from this repository: a task with a
single shared reward and many agents. Our cooperative gridworld hands both
agents an identical number every step; with two agents the correlation between
an agent's own behaviour and that number is strong enough to learn from, and
with ten it is not — each agent's contribution is buried in nine others'
noise. That is the credit-assignment problem CTDE exists to solve, and it is
also why our state representation (the *joint* position, 625 states) does not
scale: it grows as |cells|^n, and a genuinely independent learner would see
only its own local observation, making the task partially observed as well as
non-stationary.

**And where CTDE cannot be used:** any setting where no party is permitted to
see the joint state at training time — competing firms, separately-owned
vehicles, federated deployments with a privacy boundary.]

---

## Report

*≈700 words. Required. Summarise the accuracy of both reward heads, the
length-bias finding, the alignment results and the multi-agent findings in one
comparison table; explain reward hacking and Goodhart's law using your own
evidence; evaluate three mitigations, naming the cost of each; argue for
PPO-based RLHF, DPO or GRPO for a stated deployment scenario under a fixed
compute budget; and close with limitations.*

### One comparison table

| result | measurement | value | baseline / comparison |
|---|---|---|---|
| TF-IDF head | held-out pairwise accuracy | [ ] | 0.500 |
| Embedding head | held-out pairwise accuracy | [ ] | 0.500 |
| TF-IDF head | reward-vs-length r | [ ] | corpus confound r = [ ] |
| Embedding head | reward-vs-length r | [ ] | corpus confound r = [ ] |
| More susceptible head | length decodability R² | [ ] | [ ] |
| Best β | win rate vs base | [ ] | 0.500 |
| Decoupling | β, KL | [ ] | — |
| IPD | reward per step | [ ] | defect 1.0 / cooperate 3.0 |
| Non-stationarity | Q-drift ratio | [ ] | 1.0 = no effect |
| Cooperative task | team reward | [ ] | random policy [ ] |

### Reward hacking and Goodhart's law

[Your explanation, **using your own evidence**. Goodhart's law in its useful
form is not "metrics get gamed" — it is that a measure and its target agree
*on the distribution the measure was calibrated on*, and an optimiser's job is
to leave that distribution. Ground it in the specific gap in your data: what
did the comparisons under-determine, and what did the policy find in that gap?
Quote completions. A paragraph about Goodhart that could have been written
without running anything is not the assignment.]

### Three mitigations, and the cost of each

*Required: name the cost. A mitigation with no cost has not been thought about.*

1. **Early stopping on a held-out human-rated set.** Rate N completions by hand
   at intervals and stop when the human series turns over, rather than when the
   proxy does. *This is the only one of the three that addresses the actual
   problem*, because it is the only one that measures the target. **Cost:**
   human labelling, continuously, for as long as you keep training — the exact
   expense the reward model was built to avoid. It is also statistically thin:
   detecting the turn in this template's sweep would mean distinguishing 0.403
   from 0.385, which needs a lot of ratings per checkpoint. [Estimate how many,
   for your effect size.]
2. **Reward-model ensembles.** Train K heads on different seeds or splits and
   optimise the mean, or the pessimistic minimum. Disagreement among members is
   itself a usable signal that you have left the training distribution.
   **Cost:** K times the training and K times the inference — and, more
   subtly, *correlated errors survive ensembling*. Every member of your
   ensemble was fitted on the same comparisons, so every member is blind to
   exactly the same thing. [Test it: would an ensemble of your TF-IDF heads
   have caught the repetition failure? All of them learned density and none of
   them saw diversity. Say so.]
3. **A tighter KL constraint.** Raise β. **Cost:** it does not fix the problem,
   it delays it — and it buys the delay with capability. In this template's
   sweep, β = 0.5 has the lowest repetition rate (0.218) *and* the lowest
   quality of any aligned variant (0.352) and the lowest win rate (0.933).
   Mathematically: the penalty makes distance from the reference expensive, so
   the policy reaches the region where the proxy is wrong later, at a larger β.
   It never makes the proxy *correct* there, and as β → ∞ the policy → the
   reference and the improvement goes to zero along with the hacking.

### PPO-based RLHF vs DPO vs GRPO, under a fixed compute budget

*Required. State the deployment scenario first, then argue, then name the
measurement that would change your mind.*

**Scenario:** [state it — model size, budget, whether you have a preference
dataset or only a scorer, latency and safety requirements, how often you
retrain.]

| | PPO-based RLHF | DPO | GRPO |
|---|---|---|---|
| models resident in training | **4** (policy, reference, reward model, critic) | **2** (policy, reference) | **3** (policy, reference, reward model) |
| explicit reward model | required | **none** | required |
| samples from the policy during training | yes | **no** | yes |
| advantage estimate | learned critic | n/a | group mean over G samples |
| library status (TRL 1.9.x) | `PPOTrainer` **experimental** | `DPOTrainer` stable | `GRPOTrainer` stable |
| main failure mode | instability; four things to tune | offline — never sees its own outputs | reward-model quality fully exposed |

[Your argument. Hooks worth engaging: (i) DPO is **offline** and never samples
from the policy during training, so it optimises a preference dataset collected
under a different policy — which is the distribution-shift failure this
product's own reward-hacking result is an instance of, and which online DPO
variants address by regenerating preferences from the current policy; (ii) the
memory table is the whole argument under a fixed budget, and two models against
four is the difference between fitting on a free T4 and not; (iii) GRPO buys
online sampling back for one extra model by discarding the critic, and what it
loses is *per-token* credit assignment — the critic could say which part of a
completion was good, a group baseline can only say which completion; (iv) group
size G matters: the standard error of the group-mean baseline falls as
1/√G, so small groups make GRPO's advantage estimate noisy in exactly the
regime where you chose it to save memory.

**Name the measurement that would flip your recommendation.** A specific number
from a specific column. "More experiments" is not an answer.]

### Limitations

*Required, and the three specific questions below must be answered.*

1. **What your evaluation does not measure.** [At least four things, each with
   how you would test whether it binds. Candidates that are genuinely true of
   this artifact: the deployed head scores `r(y)`, not `r(x, y)`, so it cannot
   tell a good answer from a good answer *to a different question*; TF-IDF has
   no semantic generalisation at all, so an unseen synonym is invisible; the
   held-out split shares a generator with the training split, so nothing here
   measures distribution shift; and `true_quality` is available only because
   this is synthetic — on real data the reward-hacking chart has no second
   series unless you rate completions by hand.]
2. **Who was represented in the preference data.** [Not a rhetorical question.
   For UltraFeedback: the "preferences" are GPT-4 ratings, not human ones, so
   the reward model is fitted to *a model's model of human preference* —
   which means its blind spots are correlated with the blind spots of the
   systems it will be used to align. Where human annotation is involved, ask
   who was recruited, in what language, under what pay and time pressure, and
   what an annotator paid per comparison is incentivised to do with a pair that
   requires ten minutes of reading.]
3. **What a majority preference signal systematically omits.** [Bradley–Terry
   fits a *single* reward function to an aggregate of comparisons. That is a
   modelling assumption with teeth: it presumes preferences are consistent
   enough across people to be represented by one function. Where they are not,
   the fitted reward is the majority's, and the minority's preference does not
   appear as an error term — it disappears. Consider also what a comparison
   *cannot* express: an annotator who thinks both responses are bad, a
   preference that depends on context the annotator does not have, and any good
   that is not visible in a single response read in isolation — honesty about
   uncertainty, refusal when refusal is right, or the long-run effect of a habit
   of answering.]

---

## Worldview Reflection

*≈300 words. Required.*

A reward model encodes whose preferences it was trained on. This product makes
that concrete: `preferences` is a table you can query, and every number in the
report above traces back to it.

**The question.** Evaluate, from a Christian worldview, whether an aggregated
human preference signal is a sufficient definition of what is good. Contrast a
reward function *fit from data* with a moral standard understood to be
objective and revealed rather than learned — Micah 6:8 ("do justly, love mercy,
walk humbly") and Romans 12:2 ("be not conformed to this world, but be
transformed by the renewing of your mind") are the suggested points of entry —
and state what the distinction implies for the design of systems intended to
act on people's behalf.

**Connect it to your Step 8 evidence.** What does it mean to optimise a
measurable proxy for a good that is not fully measurable? Does the same failure
appear in human institutions that reward measured performance — test scores,
citation counts, quarterly targets, clinical throughput?

**How to write this well.** Three things distinguish a serious answer from a
gesture:

* **Engage the strongest version of the position you do not hold.** The case
  *for* preference aggregation is genuinely strong: it is revisable in light of
  evidence, it does not require agreement on contested metaphysics before a
  system can be built, it can represent the preferences of people the designer
  has never met, and it is auditable in a way that an appeal to an external
  standard often is not. A reflection that does not state that case has not
  argued against it. Equally, the case that a revealed standard supplies
  something aggregation cannot — a ground for saying a *unanimous* preference
  is wrong, which no amount of preference data can supply from inside itself —
  is a real claim with real consequences, and dismissing it as unfalsifiable
  does not engage it either.
* **Use your own numbers.** Romans 12:2's "conformed to this world" has an
  uncomfortably precise reading in this product: a model fitted to what people
  currently prefer cannot, by construction, tell you that what they currently
  prefer is wrong. Your β sweep is an empirical instance of a general problem —
  the proxy and the target agreed until the optimiser left the region where
  they had been compared, and *nothing inside the system could tell*. Micah
  6:8's three terms are worth taking seriously as a design question rather than
  as decoration: justice is at least partly auditable, mercy is exactly the
  thing a majority preference signal is worst at representing, and humility has
  a concrete engineering form — a system that reports what it does not know,
  which is why `ScoreResponse` carries `oov_rate`.
* **Say what it implies for design, specifically.** Whatever you conclude
  about the metaethics, the engineering question is the same: what would you
  build differently? Candidate answers to argue for or against — a held-out
  evaluation set that is not drawn from the preference distribution; explicit
  constraints that are not learned and that the optimiser cannot trade away;
  documented recourse for the people a system acts on; a decision about who is
  accountable when the proxy and the good come apart, made before they do.

[**Your ≈300 words.** A reflection that reaches a different conclusion from the
one you expect to be rewarded — argued from your own evidence, and engaging the
opposing case honestly — is doing the assignment. One that reaches the expected
conclusion without engaging anything is not. The rubric is about the quality of
the reasoning, not the destination.]

---

## AI-Assistance Disclosure

*Required. What did you generate, with which tool, and how did you verify it?
Generated code must be read, understood and tested by you; blind paste-through
is not acceptable.*

[For this product in particular: if a tool wrote your `tfidf_vector`
replication, say so — and say that `tests/test_equivalence.py` is how you know
it is right, and quote the measured difference. If a tool wrote your worldview
reflection, that is a different kind of problem and you should say that too.]

---

## Limitations & Responsible Use

*At least four concrete limitations, each with how you would test whether it
binds in a deployment scenario. Then foreseeable misuse, reward-specification
risk, and the worldview reflection above.*

Four that are genuinely true of this artifact and worth taking seriously:

1. **The deployed score is uncalibrated and unbounded.** Bradley–Terry
   identifies rewards only up to an additive constant, so a threshold tuned on
   one artifact will not transfer to the next one you train. *Test: retrain
   with a different seed and compare the score distributions on the same 100
   texts. If your threshold moves, it was never a threshold.*
2. **The head scores `r(y)`, not `r(x, y)`.** It cannot distinguish a good
   answer from a good answer to a different question. *Test: score one strong
   response against ten unrelated prompts and see whether the score moves at
   all. It will not.*
3. **Out-of-vocabulary text is scored as an almost-empty vector,** and the head
   returns something close to its bias term with full confidence. *Test: `POST
   /score` with text from a domain the corpus never contained and read
   `oov_rate` — which is in the response for exactly this reason.*
4. **The reward-hacking result depends on a ground-truth series you will not
   have in production.** *Test: try to identify the decoupling point from the
   proxy series alone. In this template's sweep the proxy is flat to within
   0.04 across the entire region where the target falls by 0.036 — you cannot.*

**Reward specification.** This head is a proxy fitted to comparisons, and the
Reward Hacking tab shows the point at which optimising it stopped improving the
thing it stood in for. Anyone using this score as an **optimisation target**
rather than as a **diagnostic** should read that tab first, and should assume
the same failure exists in their setting with a different mechanism.

**Privacy.** `POST /score` receives user text and stores a SHA-256 digest, never
the content. That is pseudonymisation, not anonymisation: anyone holding a
candidate text can confirm it was submitted by hashing it themselves, and short
or guessable inputs are effectively not protected at all. If your threat model
includes an adversary who can enumerate plausible inputs, a hash is not enough
and you need a keyed MAC or no log at all.
