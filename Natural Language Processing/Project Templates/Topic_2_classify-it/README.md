# Classify-It — Topic 2 starter skeleton

> **A text-classification service on the three-cloud stack.**
> Streamlit Community Cloud (UI) ⇄ FastAPI on Render.com (API) ⇄ Supabase Postgres (data).
>
> This is a **skeleton, not a solution.** The infrastructure is finished and
> deployable as-is. The machine-learning layer — every function in `api/nlp.py` —
> is yours.

The product: a customer-operations team wants to classify short text and get back
a label, a probability they can act on, and a record of every answer the service
gave. You build it twice — once with TF-IDF and logistic regression, once by
fine-tuning a pretrained encoder — and you measure the difference instead of
assuming it.

This is the same layout as the Topic 1 template with the middle box swapped. If
you understood that one, you already know where everything lives.

---

## Live deployment URLs (fill these in)

| Tier | Platform | URL |
|------|----------|-----|
| **UI** | Streamlit Community Cloud | `https://<your-app>.streamlit.app` |
| **API** | Render.com | `https://<your-api>.onrender.com` |
| **Data** | Supabase | `https://<your-project-ref>.supabase.co` |

Replace these with your real URLs before you submit. Graders open them, and they
check `GET /healthz` and `GET /version`.

---

## What is given, and what is yours

| | Given, working | Yours to write |
|---|---|---|
| **API** | `api/main.py`: all six routes, CORS, `/healthz`, `/version`, error handling, prediction logging | every function in `api/nlp.py` |
| **Data** | `api/db.py`, the two-table schema in `db/migrations/001_init.sql`, RLS policies, `db/seed.py` | nothing — but read the SQL comments |
| **UI** | `ui/app.py`: all five tabs, API client, error states, both database reads | the **Concepts** tab, and the worked examples in the comparison tab |
| **Ops** | `render.yaml`, `.env.example`, secrets example, `.gitignore` | your actual secrets (never committed), getting the artifact onto Render |
| **Tests** | infrastructure + schema tests that pass on a fresh fork | make the `contract` tests pass |
| **Docs** | this file, `MODEL_CARD.md` scaffold | fill in the model card |

If you catch yourself building a database helper or a health endpoint, stop —
it already exists, and the time belongs to the modelling work instead.

### The endpoints, and which function backs each one

| Endpoint | Backed by | Notes |
|---|---|---|
| `POST /predict` | `nlp.predict` | logs one row to `predictions` |
| `POST /predict_batch` | `nlp.predict_batch` | up to 64 texts, one pass, order preserved |
| `GET /schema` | `nlp.label_schema` | the two labels, their definitions, the class balance |
| `GET /runs` | `api.db.latest_runs` | training runs, already working |
| `GET /healthz` | — | already working |
| `GET /version` | — | already working |

---

## Get it running (about 20 minutes, plus the torch download)

### 1. Fork and install

```bash
git clone <your-fork-url> && cd Topic_2_classify-it
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

That install pulls torch and is large. If you only want to start on the data and
metrics layer, you can skip it for now and run the tests that do not need a model.

### 2. Prove the skeleton works before you change anything

```bash
pytest -m "not contract"
```

Everything should pass. If it doesn't, fix your environment now — debugging a
broken install while also debugging your first fine-tune is twice the work.

### 3. See what you have to build

```bash
pytest -m contract                      # the whole to-do list
pytest -m "contract and not model" -x   # just the parts that need no artifact
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
2. SQL Editor → New query → paste `db/migrations/001_init.sql` → Run. This
   creates **two** tables: `runs` and `predictions`.
3. `cp .env.example .env` and fill in the URL and the **service-role** key.
4. `python db/seed.py` to put placeholder rows in both tables so you can see the
   Baseline vs. Transformer and Recent Predictions tabs render. Delete those rows
   once your own runs are landing — they are tagged `model_version = 'seed'`.
5. `cp ui/.streamlit/secrets.toml.example ui/.streamlit/secrets.toml` and fill in
   the URL and the **anon** key.

> Two different keys, on purpose. The service-role key can write and lives only
> on the API side. The anon key can only `SELECT`, because that is the only
> policy the migration creates, and it is the one the browser sees.

### 6. Train, somewhere with memory

Fine-tuning does not happen on Render and it does not happen inside a request.
Run `fit_baseline` and `fine_tune_transformer` from a notebook, a Colab session,
or a local script; write the run metadata to `runs`; save the artifacts; then get
the artifacts to the web service — commit the small baseline pickle, or push the
checkpoint to the Hugging Face Hub and pull it in the build command. Whichever
you choose, write down which in your model card.

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

`api/nlp.py` is written to be worked top to bottom. Data first. If your loader is
wrong you will spend an hour training on garbage before you find out.

1. `load_labeled_dataset` — read the corpus, drop empty rows, normalize the label
   column to your human label strings once and for all.
2. `validate_dataset` — enforce single-label with exactly two values, and look at
   the class balance it computes. That number determines how you read every
   metric for the rest of the project.
3. `stratified_split` — deterministic, disjoint, class proportions preserved,
   de-duplicated first.
4. `compute_metrics` — count the confusion matrix by hand once. Do this before
   you train anything, so you already know what the numbers mean when they
   arrive.
5. `fit_baseline` — TF-IDF + logistic regression. Minutes, not hours, and it sets
   the bar the transformer has to clear.
6. `fine_tune_transformer` — DistilBERT with the Trainer API, offline, 2–3
   epochs, learning rate around 2e-5.
7. `load_model`, `score_texts` — the serving path, with a module-level cache.
8. `calibrate` — the step that turns a score into a probability you can put in
   front of a stakeholder.
9. `predict`, `predict_batch` — one shared code path so they cannot disagree.
10. `label_schema` — the service's self-description, read from config, not
    hard-coded twice.

---

## Things that will cost you an evening if nobody tells you

- **Accuracy is the wrong headline number.** If your corpus is 90/10, answering
  with the majority label every time scores 0.90 and finds nothing. There is a
  contract test that asserts exactly that. Lead with F1 and recall on the
  positive class, and state the class balance next to every number you report.
- **Do not fine-tune on the free Render plan.** It has a few hundred megabytes of
  RAM; training wants gigabytes. The process gets killed and the logs just stop,
  which reads like a mysterious deploy failure. Fine-tune locally or in Colab and
  load the artifact. The web service loads and answers, nothing else.
- **The softmax output is not a calibrated probability.** Fine-tuned encoders are
  overconfident — 0.99 on inputs they get wrong. Fit a Platt or isotonic
  calibrator on held-out data, persist it with the model, and show the
  before/after reliability curve in your report. If you skip calibration, say
  "uncalibrated" in the model card instead of promising something else.
- **Fine-tuning learning rates are small.** 2e-5 to 5e-5. Adam's default of 1e-3
  will wreck the pretrained weights in a few hundred steps and give you a model
  that predicts one class for everything. If accuracy exactly equals your
  majority-class share, look here first.
- **Persist the vectorizer with the classifier.** Refitting TF-IDF at prediction
  time builds a different vocabulary, so the coefficients index features that no
  longer mean what they meant. Nothing crashes. The model just gets quietly
  worse, and you will blame the model.
- **Load the model once.** A module-level cache. Deserializing DistilBERT per
  request makes a correct service feel broken, and one Render worker holding two
  copies will run out of memory.
- **Do not evaluate on data you trained on.** That includes the rows you used to
  pick a threshold or fit the calibrator. De-duplicate before splitting, too:
  boilerplate support messages repeat, and duplicates straddling the split turn
  part of your held-out score into a memorization score.
- **Render free tier sleeps.** The first request after idle takes 30–60 seconds,
  and with a transformer to load it is closer to 60. Your UI is not broken; it is
  waking up. Say so in your demo.
- **Do not put the service-role key in `secrets.toml`.** It ends up in the
  browser. Anyone can then write to your database.
- **Do not disable RLS to make the Recent Predictions tab work.** If the tab is
  empty, the policy is right and your insert is failing — check the API logs.
- **Never log the input text.** Only `sha256_text(...)`. The corpus is other
  people's support messages, and the anon key can read the predictions table from
  the browser. A "temporary" debug column publishes all of it.
- **Commit `.env.example`, never `.env`.** Both are in this repo's `.gitignore`
  for the second one — and so are `artifacts/` and `*.csv`, so do not commit a
  400 MB checkpoint or someone else's dataset into a public fork.

---

## What you submit

The assignment's deliverables list is in the course assignment policy. This
repository is where the README, the model card, and the two live URLs live —
make sure they are current before you submit, because they are the first thing
opened.
