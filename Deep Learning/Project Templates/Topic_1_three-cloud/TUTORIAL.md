# Three-Cloud Tutorial
## Streamlit + FastAPI + Supabase — a reusable AI-product pattern

> A step-by-step guide to building and deploying a small machine-learning product
> split cleanly across three managed clouds. Work through it once and you can
> reuse the exact same skeleton for tabular classification, CNNs, sequence
> models, transformers, fine-tuning, and generative models. The other templates
> in this repository are worked examples of that reuse.

---

## Table of Contents
1. [Why three clouds? (Separation of Concerns)](#1-why-three-clouds-separation-of-concerns)
2. [The architecture at a glance](#2-the-architecture-at-a-glance)
3. [Prerequisites & accounts](#3-prerequisites--accounts)
4. [Project structure](#4-project-structure)
5. [Part A — Supabase (the data cloud)](#5-part-a--supabase-the-data-cloud)
6. [Part B — FastAPI (the model cloud)](#6-part-b--fastapi-the-model-cloud)
7. [Part C — Streamlit (the UI cloud)](#7-part-c--streamlit-the-ui-cloud)
8. [Part D — Run all three locally](#8-part-d--run-all-three-locally)
9. [Part E — Deploy: Supabase → Render → Streamlit](#9-part-e--deploy-supabase--render--streamlit)
10. [Part F — Testing with pytest](#10-part-f--testing-with-pytest)
11. [Reusing this pattern](#11-reusing-this-pattern)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Why three clouds? (Separation of Concerns)

A common first project ships a Streamlit app with the model *inside* the
Streamlit process and the data in a bundled SQLite file. That is fine for a demo
but it mixes three very different concerns into one process. This pattern splits
them onto three cooperating clouds:

| Concern | Lives on | Why it is separate |
|---|---|---|
| **User interface** | Streamlit Community Cloud | Changes often, must be cheap and public, holds no secrets beyond a read-only key |
| **Model / business logic** | FastAPI on Render.com | CPU-heavy, holds the model + the service-role key, versioned and health-checked independently |
| **Data / state** | Supabase (Postgres) | Durable source of truth for datasets, run history, and prediction logs; survives redeploys of the other two |

This is the software-engineering principle of **separation of concerns**. What it
buys you:

- **Observability** — each tier has its own logs, its own `/healthz`, its own deploy.
- **Auditability** — every training run and every prediction is a row in Postgres you can query months later.
- **Team workflow** — the UI person and the model person can work and deploy independently as long as the API contract (the Pydantic schemas) holds.

The rule that keeps the separation honest:

> **No model code in the Streamlit process, and no SQL writes in the Streamlit
> process.** The UI calls FastAPI over HTTPS for everything, and does at most a
> single *read-only* anon-key query for display.

---

## 2. The architecture at a glance

```
   Browser
      │  (HTTPS)
      ▼
┌─────────────────────┐     HTTPS POST /train, /predict, ...     ┌──────────────────────┐
│  Streamlit UI       │ ───────────────────────────────────────▶ │  FastAPI service     │
│  (Streamlit Cloud)  │ ◀─────────────────────────────────────── │  (Render.com)        │
│  thin client        │              JSON responses               │  PyTorch model here  │
└─────────┬───────────┘                                           └──────────┬───────────┘
          │ read-only anon-key SELECT (runs)                                  │ service-role read/write
          ▼                                                                   ▼
                          ┌───────────────────────────────────────────┐
                          │  Supabase Postgres                         │
                          │  datasets · runs · predictions             │
                          └───────────────────────────────────────────┘
```

The worked example in this template is **Regress-It**: an interactive
linear-regression trainer. But the *shape* — UI ⇄ API ⇄ DB — never changes.

---

## 3. Prerequisites & accounts

**Software**
- Python 3.11+ ([python.org](https://www.python.org/downloads/))
- Git + a GitHub account ([github.com](https://github.com/))
- VS Code or PyCharm

**Free accounts (create these first)**
- **Supabase** — [supabase.com](https://supabase.com/) (the Postgres data layer)
- **Render.com** — [render.com](https://render.com/docs/deploy-fastapi) (hosts the FastAPI service)
- **Streamlit Community Cloud** — [streamlit.io/cloud](https://streamlit.io/cloud) (hosts the UI)

**Knowledge**
- Basic Python, a little SQL, and familiarity with the FastAPI *First Steps* and
  Streamlit *Get Started* tutorials. It helps to build and deploy those sample
  apps once before starting here.

---

## 4. Project structure

This is the template you fork. Each tier has its **own** `requirements.txt` so
each cloud installs only what it needs.

```
three-cloud/
├── README.md                       # product README (fill in your 3 URLs at the top)
├── TUTORIAL.md                     # this file
├── MODEL_CARD.md                   # rendered in the UI 'Model Card' tab
├── render.yaml                     # Render blueprint for the API
├── .env.example                    # API secrets template (SUPABASE_URL / SERVICE_KEY)
├── requirements-dev.txt            # everything, for local dev + tests
│
├── shared/                         # imported by BOTH the api and db tiers
│   ├── schemas.py                  # Pydantic request/response models (the API contract)
│   └── data.py                     # synthetic dataset generator (known ground truth)
│
├── api/                            # Cloud #2 — FastAPI (Render.com)
│   ├── main.py                     # 6+ endpoints
│   ├── db.py                       # the ONLY Supabase access for the model tier
│   ├── training.py                 # the ONLY model code (PyTorch)
│   ├── configs/default.yaml
│   └── requirements.txt
│
├── db/                             # Cloud #3 — Supabase
│   ├── migrations/001_init.sql     # datasets · runs · predictions + RLS
│   └── seed.py                     # seed one default dataset
│
├── ui/                             # Cloud #1 — Streamlit
│   ├── app.py                      # 5 tabs, thin client
│   ├── requirements.txt            # no torch here on purpose
│   └── .streamlit/secrets.toml.example
│
└── tests/                          # pytest (schema, health, numerical, roundtrip)
```

---

## 5. Part A — Supabase (the data cloud)

### A.1 Create the project
1. Sign in at [supabase.com](https://supabase.com/) → **New project**.
2. Pick a name and a strong database password, choose the free plan, and wait for
   provisioning.
3. Copy three things from **Project Settings → API**:
   - **Project URL** → `https://<project-ref>.supabase.co`
   - **anon public key** → safe for the browser (the Streamlit UI uses this)
   - **service_role key** → **secret**, server-side only (the FastAPI service uses this)

> The `<project-ref>` in the URL is your Supabase project reference.

### A.2 Apply the migration
Supabase's REST API cannot run arbitrary DDL, so apply the schema by hand once:

1. Open **SQL Editor → New query**.
2. Paste the entire contents of [`db/migrations/001_init.sql`](db/migrations/001_init.sql).
3. Click **Run**.

That creates three tables and turns on Row Level Security:

- `datasets(id, name, slope, intercept, noise, n_points, xs, ys, created_at)`
- `runs(id, dataset_id, lr, batch_size, epochs, mse, mae, r2, weights_json, created_at)`
- `predictions(id, run_id, x, yhat, created_at)`

The migration also adds one RLS policy: **anonymous users may `SELECT` from
`runs`** and nothing else. That is exactly what the UI's "Run History" tab needs,
and it means the public anon key can never write or read the other tables. The
API uses the service-role key, which bypasses RLS, so all writes stay server-side.

### A.3 Seed a default dataset
From the repo root, with your API secrets in a local `.env` (see Part B.1):

```bash
python -m db.seed
```

This inserts one `default-linear` dataset (500 points from `y = 2.5x + 1 + noise`).

---

## 6. Part B — FastAPI (the model cloud)

The API is the only tier that touches the model and the only tier that *writes*
to Supabase.

### B.1 Local secrets
Copy the template and fill in your Supabase values (never commit `.env`):

```bash
cp .env.example .env
# edit .env:
#   SUPABASE_URL=https://<project-ref>.supabase.co
#   SUPABASE_SERVICE_KEY=<service_role key>
```

### B.2 Install and run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r api/requirements.txt python-dotenv
uvicorn api.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the auto-generated Swagger UI.

### B.3 The endpoints
This template ships seven:

| Method & path | What it does |
|---|---|
| `POST /datasets` | Generate a synthetic dataset and persist it to Supabase |
| `POST /train` | Read a `dataset_id` from Supabase, train, write the run row, return `run_id` + metrics |
| `POST /predict` | Given a `run_id` + `x`, predict `yhat` and **log it** to Supabase |
| `GET /runs` | Return the latest 50 runs from Supabase |
| `GET /runs/{run_id}` | Return one run (used by the fitted-line overlay) |
| `GET /healthz` | 200 when the model loader **and** the Supabase client are both reachable |
| `GET /version` | git SHA, PyTorch version, Supabase project ref |

Every request and response is a Pydantic type in
[`shared/schemas.py`](shared/schemas.py) — that module *is* the contract the UI
codes against.

### B.4 Where the concepts live
- [`api/training.py`](api/training.py) holds the **PyTorch** loop: `nn.Linear(1,1)`
  trained with mini-batch SGD, a 20% held-out split, and **MSE / MAE / R²**
  computed on that split. Learning rate, batch size, and epochs come from the
  request body — this is where the effect of learning rate on convergence,
  divergence, and oscillation becomes tangible.
- The metrics are computed **server-side** and persisted to `runs`; the UI never
  recomputes them.

Try it from the docs page: `POST /datasets`, then `POST /train` with `lr: 0.01`
(converges) and again with `lr: 1.5` (diverges → NaN/huge MSE). Both runs land in
the `runs` table so you can compare them later.

---

## 7. Part C — Streamlit (the UI cloud)

The UI is a **thin client**: it imports `requests` and `streamlit`, not `torch`.

### C.1 Local secrets
```bash
cp ui/.streamlit/secrets.toml.example ui/.streamlit/secrets.toml
# edit it:
#   API_URL           = "http://localhost:8000"        (for local dev)
#   SUPABASE_URL      = "https://<project-ref>.supabase.co"
#   SUPABASE_ANON_KEY = "<anon public key>"
```

### C.2 Run it
```bash
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

### C.3 The five tabs
- **Concepts** — LaTeX derivation of gradient descent and the chain rule.
- **Train** — creates a dataset, calls `/train`, shows MSE/MAE/R² and the fitted line.
- **Predict** — calls `/predict` on the most recent run.
- **Run History** — a *read-only* Supabase anon-key query against `runs`, sortable.
- **Model Card** — rendered from [`MODEL_CARD.md`](MODEL_CARD.md).

Notice the two different Supabase accesses in play: the **API** wrote the run with
the service-role key; the **UI** reads it back with the anon key. That is the
separation of concerns made concrete.

---

## 8. Part D — Run all three locally

Two terminals:

```bash
# Terminal 1 — API (model cloud, talking to real Supabase)
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000

# Terminal 2 — UI (thin client)
source .venv/bin/activate
streamlit run ui/app.py
```

Then in the browser: **Train** tab → create a dataset → run training → open the
**Run History** tab and watch the row appear (it round-tripped UI → API →
Supabase → UI). Try one converging and one diverging learning rate.

---

## 9. Part E — Deploy: Supabase → Render → Streamlit

Deploy in dependency order. Supabase is already live from Part A. Push your fork
to GitHub first.

### E.1 FastAPI on Render.com
1. Push the repo to GitHub.
2. In Render: **New → Blueprint**, point it at your repo. Render reads
   [`render.yaml`](render.yaml):
   - build: `pip install -r api/requirements.txt`
   - start: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - health check: `/healthz`
3. In the service's **Environment** settings add two secrets:
   - `SUPABASE_URL` = `https://<project-ref>.supabase.co`
   - `SUPABASE_SERVICE_KEY` = your service-role key
4. Deploy. When it is live, visit `https://<your-service>.onrender.com/healthz`
   and confirm `{"status":"ok", ...}`. Copy this base URL.

> **Free-tier note:** Render free services sleep after inactivity, so the first
> request after idle takes ~30–60s to wake. That is normal; the UI just needs a
> generous request timeout (already set to 120s in `ui/app.py`).

### E.2 Streamlit on Community Cloud
1. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → **New app**.
2. Repository = your fork, **Main file path = `ui/app.py`**.
3. Open **Advanced settings → Secrets** and paste (using your real values):
   ```toml
   API_URL = "https://<your-service>.onrender.com"
   SUPABASE_URL = "https://<project-ref>.supabase.co"
   SUPABASE_ANON_KEY = "<anon public key>"
   ```
4. Deploy. Streamlit installs `ui/requirements.txt`. Copy the public app URL.

### E.3 Verify the whole chain
On the deployed Streamlit app, run a training job. If the row shows up in **Run
History**, all three clouds are talking. You now have the three public URLs for
the UI, the API, and the database.

---

## 10. Part F — Testing with pytest

This template ships four kinds of test in [`tests/`](tests/):

| File | What it checks |
|---|---|
| `test_schema.py` | request-schema validation for `/predict` (422 on bad input) |
| `test_healthz.py` | smoke test that `/healthz` returns 200 |
| `test_training.py` | numerical test: the trained slope recovers the synthetic ground truth within tolerance |
| `test_supabase_roundtrip.py` | inserts a fixture dataset, trains, confirms a run row was written |

Run them:

```bash
pip install -r requirements-dev.txt
pytest -q
```

The schema/health/numerical tests stub Supabase with an in-memory fake (via
`tests/conftest.py`), so they pass offline. The round-trip test **auto-skips**
unless real credentials are present — run it against a throwaway project with:

```bash
SUPABASE_URL=... SUPABASE_SERVICE_KEY=... pytest tests/test_supabase_roundtrip.py
```

---

## 11. Reusing this pattern

Every later product is the **same three clouds** with the middle box swapped. Keep
`shared/schemas.py`, `api/db.py`, `api/main.py` (endpoint shape), `ui/app.py` (tab
shape), and the test skeleton — only the model and the tables change. Two worked
examples live alongside this one in the repository:

| Example | Model in `api/training.py` | Supabase tables | New endpoints |
|---|---|---|---|
| Income-Insight | PyTorch **MLP** + sklearn pipeline (tabular) | `datasets`, `runs`, `run_artifacts`, `predictions` | `/predict_batch`, `/schema`, `/audit` |
| See-Sense | **CNN** + Grad-CAM (image classification) | `datasets`, `runs`, `run_artifacts`, `image_metadata` | `/predict` (multipart), `/predict_sample`, `/classes` |

The same skeleton extends naturally to sequence models (LSTM/GRU + attention),
transformer pipelines, fine-tuning workflows, and generative models (VAE/DDPM).

The invariants that carry across every variant:
- UI is a thin client; **no model code, no SQL writes** in Streamlit.
- The API owns the model and is the only writer to Supabase.
- Every prediction/run is **logged** to Postgres for audit.
- Store **hashes, not raw payloads** for images/text/PII.
- `Concepts` (LaTeX) and `Model Card` tabs appear in every product.
- Keep `/healthz` and `/version`; keep the four test categories.

---

## 12. Troubleshooting

**`KeyError: 'SUPABASE_URL'` on API startup**
`.env` not loaded or the var not set. Locally, `db.py` reads `os.environ`; use a
`.env` + `python-dotenv`, or `export` the vars. On Render, set them in the
service's Environment settings.

**Streamlit "Run History" is empty but training works**
The anon RLS policy may not be applied. Re-run the `create policy "anon can read
runs"` block from `001_init.sql`. Confirm the UI uses the **anon** key, not the
service key.

**CORS error in the browser console**
The API must allow the Streamlit origin. `ALLOWED_ORIGINS` defaults to `*`; if you
tightened it, add your `https://<app>.streamlit.app` origin and redeploy.

**First request after idle times out**
Render free tier cold start. Retry after ~60s; the client timeout is already
generous. Hitting `/healthz` warms the service.

**`supabase` insert returns an empty `data` list**
You are probably using the anon key on the API tier. The API must use the
**service-role** key so RLS does not block the insert.

**Torch install is slow / large on Render**
`api/requirements.txt` pins CPU-only `torch`, which is the free-tier-friendly
choice. Do not add a GPU build.

---

**You have now built and deployed a three-cloud AI product.** Every other template
in this repository reuses exactly this skeleton.
