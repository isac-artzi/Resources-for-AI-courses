# MoodLens — Topic 4 starter skeleton

> **A sentiment and aspect-based sentiment service on the three-cloud stack.**
> Streamlit Community Cloud (UI) ⇄ FastAPI on Render.com (API) ⇄ Supabase Postgres (data).
>
> This is a **skeleton, not a solution.** The infrastructure is finished and
> deployable as-is. The NLP layer — every function in `api/nlp.py` — is yours.

The layout is the Topic 1 template with the middle box swapped: same three
tiers, same schema-first contract, same "make the contract tests pass" workflow.
If you understood TokenForge, you already know where everything lives here.

What is new in this topic is that the product now makes a **claim about a
person's writing** and logs it. That is why there is a second table, why every
prediction carries a model version, and why one of the six tabs is a bias audit
that is graded on its own.

---

## Live deployment URLs (fill these in)

| Tier | Platform | URL |
|------|----------|-----|
| **UI** | Streamlit Community Cloud | `https://<your-app>.streamlit.app` |
| **API** | Render.com | `https://<your-api>.onrender.com` |
| **Data** | Supabase | `https://<your-project-ref>.supabase.co` |

Replace these with your real URLs before you submit. Graders open them.

---

## What is given, and what is yours

| | Given, working | Yours to write |
|---|---|---|
| **API** | `api/main.py`: all five routes, CORS, error handling, batch logging | every function in `api/nlp.py` |
| **Data** | `api/db.py`, the two-table schema in `db/migrations/001_init.sql`, RLS policies | nothing — but read the SQL comments |
| **UI** | `ui/app.py`: all six tabs, API client, charts, audit view | the **Concepts** tab content |
| **Ops** | `render.yaml`, `.env.example`, secrets example, `.gitignore` | your actual secrets (never committed) |
| **Tests** | infrastructure + schema tests that pass on a fresh fork | make the `contract` tests pass |
| **Docs** | this file, `MODEL_CARD.md` scaffold | fill in the model card, including the graded failure table |

If you catch yourself building a database helper, a health endpoint or a chart,
stop — it already exists, and the time belongs to the modelling and the audit.

### The endpoints

| Method | Path | What it does |
|--------|------|--------------|
| POST | `/predict` | one review → document sentiment + aspect breakdown, logged |
| POST | `/predict_batch` | up to 64 reviews in one forward pass, logged in one round trip |
| GET | `/audit` | recent served predictions, newest first, hashes only |
| GET | `/healthz` | process up, database reachable, model loaded |
| GET | `/version` | service, git SHA, model version, base model |

### The tabs

Concepts · Score Text · Aspect Breakdown · Model Performance · Bias Audit · Model Card

Scoring goes through the API. The metrics tabs read the `runs` table directly
with the anon key — the UI never computes a metric, it renders one that a
training run wrote down. That split is deliberate: a number recomputed in the
browser is a number that can quietly disagree with the one in your report.

---

## Get it running (about 25 minutes)

### 1. Fork and install

```bash
git clone <your-fork-url> && cd Topic_4_moodlens
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

That pulls torch, which is large. If you only want to read the code and run the
infrastructure tests first, `pip install fastapi uvicorn pydantic supabase
pytest httpx` is enough for `pytest -m "not contract"`.

### 2. Prove the skeleton works before you change anything

```bash
pytest -m "not contract"
```

Everything should pass. If it doesn't, fix your environment now — debugging a
broken install while also debugging your first fine-tuning run is twice the work.

### 3. See what you have to build

```bash
pytest -m contract
pytest -m "contract and not network"    # the half that needs no downloads
```

Every failure is a function in `api/nlp.py` with a docstring explaining what it
must return. That list is your assignment.

### 4. Run the two tiers locally

```bash
uvicorn api.main:app --reload            # terminal 1 → http://127.0.0.1:8000/docs
streamlit run ui/app.py                  # terminal 2 → http://localhost:8501
```

The UI works immediately. The product tabs will say "not implemented yet" until
you write the code — the API returns `501` for a `NotImplementedError`, and the
UI renders that as a message pointing at the function you still owe.

### 5. Wire up Supabase

1. Create a project at supabase.com (free tier is fine).
2. SQL Editor → New query → paste `db/migrations/001_init.sql` → Run.
3. `cp .env.example .env` and fill in the URL and the **service-role** key.
4. `python db/seed.py` to put one fixture training run and two demo predictions
   in the database, so the Model Performance and Bias Audit tabs render before
   you have trained anything.
5. `cp ui/.streamlit/secrets.toml.example ui/.streamlit/secrets.toml` and fill in
   the URL and the **anon** key. Streamlit looks for `.streamlit/secrets.toml`
   relative to the directory you run from, so if you start the UI from the
   repository root (which is what the command below does) put a copy in
   `.streamlit/secrets.toml` at the root as well — both paths are gitignored.
   On Streamlit Community Cloud there is no file at all; you paste the values
   into the Secrets box.

> Two different keys, on purpose. The service-role key can write and lives only
> on the API side. The anon key can only `SELECT`, because that is the only
> policy the migration creates, and it is the one the browser sees.

The seed row's numbers are all invented and the UI puts a red banner on them.
Delete it (`delete from runs where model_version like 'seed%';`) once you have a
real run.

### 6. Train — somewhere that is not Render

Fine-tune locally or in a free hosted notebook, then commit or upload the
weights and point `MODEL_DIR` at them. Render's free plan has neither the memory
nor the time; a build that tries to train will be killed halfway through and the
logs will not tell you why in a useful way.

Log the run with `api.db.insert_run(...)`. That row is what the Model
Performance and Bias Audit tabs render, and it is the only place your numbers
are written down in a form somebody else can check.

### 7. Deploy

**API → Render.** New → Blueprint → point at your fork. `render.yaml` does the
rest. Afterwards, set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in the service's
environment settings.

**UI → Streamlit Community Cloud.** New app → your fork → main file
`ui/app.py`. Paste the three keys into the Secrets box.

Confirm both `GET /healthz` and `GET /version` answer from the live Render URL,
then record all three URLs in the table at the top of this file.

---

## The order to implement in

`api/nlp.py` is written to be worked top to bottom, and the order is not
arbitrary.

1. `load_reviews` — get the corpus and the splits right first. A leak here
   poisons every number you produce afterwards, quietly and in your favour.
2. `fit_baseline`, `baseline_predict` — the TF-IDF baseline. It trains in
   seconds and it gives you the number the transformer has to beat.
3. `evaluate_documents` — write the metrics before you have a model to flatter.
   The contract tests for it run offline in milliseconds.
4. `fine_tune_transformer`, `load_classifier` — the Hugging Face side.
5. `fit_calibrator`, `predict_sentiment`, `predict_sentiment_batch` — serving,
   with an honest probability.
6. `extract_aspects` — per-aspect sentiment with evidence quoted from the input.
7. `evaluate_aspects`, `slice_of`, `evaluate_slices` — the numbers behind the
   Bias Audit tab.

---

## The bias audit is graded on its own

The Bias Audit tab has to show, from your own run:

- performance on **at least two slices** of the held-out data (review length is
  one you can measure; the second is yours to choose and document), and
- **at least three example failures**, each with the ethical risk it illustrates
  and the use limitation that follows from it.

The failures live in the **Documented failures** section of `MODEL_CARD.md`; the
tab reads that section directly so the audit and the model card cannot drift
apart. Keep the heading spelled exactly as it ships.

A limitation is only worth writing if someone could enforce it. "Be careful with
sarcasm" is not a limitation. "Predictions on messages under twenty words are
not used to route or escalate anything without a person reading the message"
is one, because you can point at the code that implements it.

---

## Things that will cost you an evening if nobody tells you

- **Sarcasm and negation are the classic failures, and fine-tuning does not fix
  them cheaply.** "Great, another update that breaks everything" is positive by
  every surface cue in the sentence. So is "I can't say I loved it". A larger
  model helps a little; sarcasm-labelled data helps more and you do not have
  any. Plan to *document* these failures rather than to eliminate them, and put
  the examples in the model card where a reader will find them.

- **Your aspect definitions are a modelling decision, not a fact.** Nobody
  labelled a review with "acting". You decided that "acting" exists, that
  "pacing" belongs under "plot", and that a sentence about the lead actor's
  script is one or the other. Every per-aspect number inherits that decision.
  Write the rules down before you annotate anything, or you will apply different
  ones on Tuesday than you did on Monday and never know it.

- **A calibration plot that comes out as a clean diagonal is usually a bug.**
  Bin the predictions by predicted probability, then plot the mean *predicted*
  probability against the *observed* fraction of positives in that bin. If you
  plot the mean predicted probability against itself you get a perfect diagonal
  that says nothing, and it looks exactly like the result you were hoping for. A
  fine-tuned transformer trained with cross-entropy is normally over-confident;
  if yours is perfectly calibrated before you calibrate it, be suspicious before
  you are pleased.

- **IMDb will flatter you.** It is long-form, balanced 50/50, and written by
  people who chose to sit down and write a review. Real input is a 12-word
  comment typed on a phone with no punctuation. High accuracy on this corpus is
  not evidence of a model that works on short, noisy text — it is evidence about
  long, clean text. Test on something short before you claim otherwise, and say
  which one your number came from.

- **Slicing on an attribute you inferred makes the audit circular.** If you
  assign each review a genre with a classifier or a keyword rule and then report
  accuracy by genre, a gap could be in your sentiment model, in your genre
  guesser, or in both, and the table cannot tell you which. Prefer attributes
  you can measure (length) or that the corpus ships. If you must infer, mark the
  slice `observed=False` — the UI will label it — and treat the result as a
  hypothesis, not a finding.

- **Read `n` before you read the gap.** A bucket of 23 reviews will happily show
  a twelve-point difference that is nothing but sample size. "We found bias" is a
  serious claim; make it on numbers that can carry it.

- **Do not train on Render.** Free tier, CPU, and a build timeout. Train
  elsewhere, ship the weights.

- **Load the model once.** Module-level cache (`_CLASSIFIER`). A per-request
  load means every keystroke re-reads hundreds of megabytes and the tab times
  out while your code looks correct.

- **Render free tier sleeps.** The first request after idle takes 30–60 seconds,
  and with a transformer to load it can be worse. Your UI is not broken; it is
  waking up. Say so in your demo, and check `/healthz` — it reports whether the
  model is in memory without loading it.

- **Do not put the service-role key in `secrets.toml`.** It ends up in the
  browser. Anyone can then write to your database — including rewriting the
  metrics the Bias Audit tab reports.

- **Do not disable RLS to make a tab work.** If a tab is empty, the policy is
  right and your insert is failing — check the API logs.

- **Check your label mapping.** Index 1 is not always "positive". Read the
  mapping off the model's own config. If you get it backwards, every metric in
  your report is exactly inverted and every one of them still looks plausible.

- **Commit `.env.example`, never `.env`.** Both are in this repo's `.gitignore`
  for the second one.

---

## What you submit

The assignment's deliverables list is in the course assignment policy. This
repository is where the README, the model card and the three live URLs live —
make sure they are current before you submit, because they are the first thing
opened.
