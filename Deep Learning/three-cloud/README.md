# Regress-It — Three-Cloud Reference Template

> A working, forkable template for a three-cloud architecture:
> **Streamlit (UI) ⇄ FastAPI (Model API) ⇄ Supabase (Data)**. Fork it, wire up
> your own accounts, and reuse the exact same pattern for the other templates in
> this repository.

## Live deployment URLs (fill these in)

| Tier | Platform | URL |
|------|----------|-----|
| **UI** | Streamlit Community Cloud | `https://<your-app>.streamlit.app` |
| **API** | Render.com | `https://<your-api>.onrender.com` |
| **Data** | Supabase | `https://<your-project-ref>.supabase.co` |

> Replace the placeholders above with your real URLs once deployed.

---

## What it does

Regress-It is an interactive teaching demo for 1-D linear regression. You pick a
learning rate, batch size, and epoch count; the API trains `y = w·x + b` with
PyTorch mini-batch SGD on synthetic data, reports held-out **MSE / MAE / R²**,
and persists every run. The UI lets you visualise convergence, make predictions,
and browse run history.

## Architecture

```
┌──────────────────────┐   HTTPS/JSON    ┌──────────────────────┐   service-role   ┌──────────────────┐
│  Streamlit Cloud     │ ──────────────► │  FastAPI on Render   │ ───────────────► │  Supabase        │
│  (ui/app.py)         │                 │  (api/main.py)       │   full access    │  Postgres        │
│  thin client, no ML  │                 │  PyTorch training    │                  │  datasets/runs/  │
│                      │ ◄────anon key,  │                      │                  │  predictions     │
│                      │   read-only ────┼─────────────────────┼──────────────────►│  (RLS: anon can  │
└──────────────────────┘   SELECT runs   └──────────────────────┘                  │   only SELECT)   │
                                                                                    └──────────────────┘
```

- **UI never touches the model or writes SQL.** It calls the API over HTTPS and
  performs one read-only `SELECT` on `runs` with the anon public key.
- **API owns the model and all writes**, using the Supabase **service-role** key.
- **Supabase is the single source of truth** for datasets, runs, and predictions.

See [`TUTORIAL.md`](./TUTORIAL.md) for the full step-by-step build and deploy guide.

## Project structure

```
three-cloud/
├── TUTORIAL.md               # Full build + deploy guide (start here)
├── README.md                 # This file
├── MODEL_CARD.md             # Model details, intended use, limitations
├── shared/                   # Code shared by both tiers
│   ├── schemas.py            # Pydantic API contract
│   └── data.py               # Synthetic linear data generator
├── api/                      # FastAPI tier (deploys to Render)
│   ├── main.py               # Endpoints
│   ├── training.py           # PyTorch linear regression
│   ├── db.py                 # Supabase (service-role) data access
│   ├── configs/default.yaml  # Default hyperparameters
│   └── requirements.txt
├── ui/                       # Streamlit tier (deploys to Streamlit Cloud)
│   ├── app.py                # 5-tab thin client
│   ├── requirements.txt      # No torch
│   └── .streamlit/secrets.toml.example
├── db/                       # Database tier (Supabase)
│   ├── migrations/001_init.sql
│   └── seed.py
├── tests/                    # pytest suite
├── render.yaml               # Render blueprint
├── requirements-dev.txt      # Both tiers + pytest (local dev)
└── .env.example
```

## Quickstart (local)

```bash
cd three-cloud

# 1. Install everything (both tiers + test tools)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# 2. Run the tests (6 pass; the live-Supabase test skips without creds)
pytest -q

# 3. Configure secrets
cp .env.example .env                                   # API: SUPABASE_URL + SERVICE key
cp ui/.streamlit/secrets.toml.example ui/.streamlit/secrets.toml

# 4. Run the API
uvicorn api.main:app --reload --port 8000

# 5. In another terminal, run the UI
streamlit run ui/app.py
```

To deploy to the three clouds, follow **Part E** of [`TUTORIAL.md`](./TUTORIAL.md):
apply `db/migrations/001_init.sql` in the Supabase SQL Editor → deploy the API
from `render.yaml` on Render → deploy the UI on Streamlit Community Cloud.

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/datasets` | Create a synthetic dataset |
| `POST` | `/train` | Train a model, persist the run, return metrics |
| `GET`  | `/runs/{run_id}` | Fetch one run |
| `GET`  | `/runs` | List recent runs |
| `POST` | `/predict` | Predict `ŷ` for an `x` using a fitted run |
| `GET`  | `/healthz` | Liveness / DB ping |
| `GET`  | `/version` | Build SHA + framework versions |

## Reusing this pattern

The three-cloud split and the file layout stay identical for every product. Swap
only the model in `api/training.py`, the Pydantic contract in `shared/schemas.py`,
the tables in `db/migrations/`, and the UI tabs — the UI stays a thin client and
Supabase stays the single source of truth. Two worked examples
(`income-insight`, `see-sense`) live alongside this one; see the
final section of [`TUTORIAL.md`](./TUTORIAL.md).

## Checklist

- [ ] Three live URLs listed at the top of this README
- [ ] `datasets`, `runs`, `predictions` tables in Supabase with RLS
- [ ] 6+ API endpoints
- [ ] 5 Streamlit tabs (Concepts, Train, Predict, Run History, Model Card)
- [ ] PyTorch training with held-out MSE/MAE/R²
- [ ] pytest suite passing
- [ ] `MODEL_CARD.md` completed
